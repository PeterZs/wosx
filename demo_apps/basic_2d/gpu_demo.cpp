/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file is the entry point for a 2D demo application demonstrating how to use WoSX.
// It reads a 'model problem' description from a JSON file, runs the GPUWalkOnSpheres,
// GPUWalkOnStars or GPUBoundaryValueCaching solvers, and writes the result to a PFM or PNG file.

#include "gpu_model_problem.h"
#include "grid.h"

template <size_t DIM>
void computeDistanceInfo(const std::vector<wosx::Vector<DIM>>& solveLocations,
                         std::shared_ptr<wosx::GPUGeometricQueries<DIM>> queries,
                         const std::string& deviceBackend,
                         bool domainIsWatertight,
                         bool solveDoubleSided,
                         bool solveExterior,
                         std::vector<DistanceInfo>& distanceInfo)
{
    // initialize GPU task handle
    std::filesystem::path wosxDirectoryPath = std::filesystem::current_path().parent_path();
    std::shared_ptr<wosx::GPUTaskHandle<float, DIM>> taskHandle =
        std::make_shared<wosx::GPUTaskHandle<float, DIM>>(wosxDirectoryPath.string(), "", deviceBackend);
    taskHandle->setGeometricQueries(queries);
    taskHandle->init();

    // compute distance info on the GPU
    std::vector<float> distToAbsorbingBoundary;
    std::vector<float> distToReflectingBoundary;
    wosx::GPUDistanceQueries<float, DIM> distanceQueries(taskHandle);
    distanceQueries.computeDistToBoundary(solveLocations, distToAbsorbingBoundary, distToReflectingBoundary);

    // determine solve locations in the valid solve region
    distanceInfo.resize(solveLocations.size());
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        float d1 = std::fabs(distToAbsorbingBoundary[i]);
        float d2 = std::fabs(distToReflectingBoundary[i]);
        float distToBoundary = d1 < d2 ? distToAbsorbingBoundary[i] : distToReflectingBoundary[i];
        bool insideDomain = !domainIsWatertight ? true : distToBoundary < 0.0;
        if (domainIsWatertight && solveExterior) insideDomain = !insideDomain;
        distanceInfo[i].inValidSolveRegion = insideDomain || solveDoubleSided;
        distanceInfo[i].distToAbsorbingBoundary = d1;
        distanceInfo[i].distToReflectingBoundary = d2;
    }
}

template <size_t DIM>
void createSamplePoints(const std::vector<wosx::Vector<DIM>>& solveLocations,
                        const std::vector<DistanceInfo>& distanceInfo,
                        std::vector<wosx::GPUSamplePoint<DIM>>& samplePts)
{
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        if (distanceInfo[i].inValidSolveRegion) {
            wosx::GPUSamplePoint<DIM> samplePt;
            if constexpr (DIM == 2) {
                samplePt.pt = wosx::float2{solveLocations[i][0],
                                           solveLocations[i][1]};

            } else if constexpr (DIM == 3) {
                samplePt.pt = wosx::float3{solveLocations[i][0],
                                           solveLocations[i][1],
                                           solveLocations[i][2]};
            }

            samplePt.type = wosx::SampleType::InDomain;
            samplePt.estimationQuantity = wosx::EstimationQuantity::Solution;
            samplePt.distToAbsorbingBoundary = distanceInfo[i].distToAbsorbingBoundary;
            samplePt.distToReflectingBoundary = distanceInfo[i].distToReflectingBoundary;
            samplePts.emplace_back(samplePt);
        }
    }
}

template <typename T, size_t DIM>
void runWalkOnSpheres(const json& solverConfig,
                      std::shared_ptr<wosx::GPUTaskHandle<T, DIM>> taskHandle,
                      std::shared_ptr<wosx::GPUGeometricQueries<DIM>> queries,
                      std::shared_ptr<wosx::GPUPDE> pde,
                      bool solveDoubleSided,
                      const std::vector<wosx::GPUSamplePoint<DIM>>& samplePts,
                      std::vector<wosx::GPUSampleStatistics<T, DIM>>& sampleStatistics)
{
    // load config settings
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);

    const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int nResidentThreads = getOptional<int>(solverConfig, "nResidentThreads", 131072);

    const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
    const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
    const bool ignoreAbsorbingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreSourceContribution = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    bool disablePersistentThreads = getOptional<bool>(solverConfig, "disablePersistentThreads", false);
    if (!disablePersistentThreads && taskHandle->getDeviceBackend() != "cuda") {
        std::cerr << "Persistent threads require CUDA backend, disabling" << std::endl;
        disablePersistentThreads = true;
    }

    // initialize GPU task handle
    taskHandle->setGeometricQueries(queries);
    taskHandle->setPDE(pde);
    taskHandle->init();

    // initialize GPU walk settings
    std::shared_ptr<wosx::GPUWalkSettings> walkSettings = std::make_shared<wosx::GPUWalkSettings>();
    walkSettings->epsilonShellForAbsorbingBoundary = epsilonShellForAbsorbingBoundary;
    walkSettings->epsilonShellForReflectingBoundary = 0.0f;
    walkSettings->silhouettePrecision = 0.0f;
    walkSettings->russianRouletteThreshold = russianRouletteThreshold;
    walkSettings->maxWalkLength = maxWalkLength;
    walkSettings->stepsBeforeUsingMaximalSpheres = 0;
    walkSettings->solveDoubleSided = solveDoubleSided ? 1 : 0;
    walkSettings->useGradientControlVariates = disableGradientControlVariates ? 0 : 1;
    walkSettings->useGradientAntitheticVariates = disableGradientAntitheticVariates ? 0 : 1;
    walkSettings->ignoreAbsorbingBoundaryContribution = ignoreAbsorbingBoundaryContribution ? 1 : 0;
    walkSettings->ignoreReflectingBoundaryContribution = 1;
    walkSettings->ignoreSourceContribution = ignoreSourceContribution ? 1 : 0;

    // initialize GPU walk on spheres solver
    wosx::GPUWalkOnSpheresSolver<T, DIM> walkOnSpheres(taskHandle, walkSettings, nResidentThreads,
                                                       !disablePersistentThreads, printLogs);

    if (disablePersistentThreads) {
        // compute workload parameters--the number of sample copies and number of
        // walks per copy--to help improve GPU occupancy when running the solver
        std::pair<uint32_t, uint32_t> workloadParameters =
            walkOnSpheres.computeWorkloadParameters(samplePts.size(), nWalks);

        // populate sample points on the GPU
        walkOnSpheres.populateSamplePoints(samplePts, workloadParameters.first, false);

        // run solver on the GPU
        walkOnSpheres.solve(1, workloadParameters.second);

    } else {
        // populate sample points on the GPU
        walkOnSpheres.populateSamplePoints(samplePts, 1, false);

        // run solver on the GPU
        walkOnSpheres.solve(nWalks);
    }

    // extract sample statistics from the GPU
    walkOnSpheres.getSampleStatistics(sampleStatistics);
}

template <typename T, size_t DIM>
void runWalkOnStars(const json& solverConfig,
                    std::shared_ptr<wosx::GPUTaskHandle<T, DIM>> taskHandle,
                    std::shared_ptr<wosx::GPUGeometricQueries<DIM>> queries,
                    std::shared_ptr<wosx::GPUPDE> pde,
                    bool solveDoubleSided,
                    const std::vector<wosx::GPUSamplePoint<DIM>>& samplePts,
                    std::vector<wosx::GPUSampleStatistics<T, DIM>>& sampleStatistics)
{
    // load config settings
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);

    const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);
    const int nResidentThreads = getOptional<int>(solverConfig, "nResidentThreads", 131072);

    const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
    const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
    const bool ignoreAbsorbingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreReflectingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreReflectingBoundaryContribution", false);
    const bool ignoreSourceContribution = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    bool disablePersistentThreads = getOptional<bool>(solverConfig, "disablePersistentThreads", false);
    if (!disablePersistentThreads && taskHandle->getDeviceBackend() != "cuda") {
        std::cerr << "Persistent threads require CUDA backend, disabling" << std::endl;
        disablePersistentThreads = true;
    }

    // initialize GPU task handle
    taskHandle->setGeometricQueries(queries);
    taskHandle->setPDE(pde);
    taskHandle->init();

    // initialize GPU walk settings
    std::shared_ptr<wosx::GPUWalkSettings> walkSettings = std::make_shared<wosx::GPUWalkSettings>();
    walkSettings->epsilonShellForAbsorbingBoundary = epsilonShellForAbsorbingBoundary;
    walkSettings->epsilonShellForReflectingBoundary = epsilonShellForReflectingBoundary;
    walkSettings->silhouettePrecision = silhouettePrecision;
    walkSettings->russianRouletteThreshold = russianRouletteThreshold;
    walkSettings->maxWalkLength = maxWalkLength;
    walkSettings->stepsBeforeUsingMaximalSpheres = stepsBeforeUsingMaximalSpheres;
    walkSettings->solveDoubleSided = solveDoubleSided ? 1 : 0;
    walkSettings->useGradientControlVariates = disableGradientControlVariates ? 0 : 1;
    walkSettings->useGradientAntitheticVariates = disableGradientAntitheticVariates ? 0 : 1;
    walkSettings->ignoreAbsorbingBoundaryContribution = ignoreAbsorbingBoundaryContribution ? 1 : 0;
    walkSettings->ignoreReflectingBoundaryContribution = ignoreReflectingBoundaryContribution ? 1 : 0;
    walkSettings->ignoreSourceContribution = ignoreSourceContribution ? 1 : 0;

    // initialize GPU walk on stars solver
    wosx::GPUWalkOnStarsSolver<T, DIM> walkOnStars(taskHandle, walkSettings, nResidentThreads,
                                                   !disablePersistentThreads, printLogs);

    if (disablePersistentThreads) {
        // compute workload parameters--the number of sample copies and number of
        // walks per copy--to help improve GPU occupancy when running the solver
        std::pair<uint32_t, uint32_t> workloadParameters =
            walkOnStars.computeWorkloadParameters(samplePts.size(), nWalks);

        // populate sample points on the GPU
        walkOnStars.populateSamplePoints(samplePts, workloadParameters.first, false);

        // run solver on the GPU
        walkOnStars.solve(1, workloadParameters.second);

    } else {
        // populate sample points on the GPU
        walkOnStars.populateSamplePoints(samplePts, 1, false);

        // run solver on the GPU
        walkOnStars.solve(nWalks);
    }

    // extract sample statistics from the GPU
    walkOnStars.getSampleStatistics(sampleStatistics);
}

template <typename T, size_t DIM>
void getSolution(const std::vector<DistanceInfo>& distanceInfo,
                 const std::vector<wosx::GPUSampleStatistics<T, DIM>>& sampleStatistics,
                 std::vector<T>& solution)
{
    solution.resize(distanceInfo.size(), T(0.0f));
    int counter = 0;
    for (int i = 0; i < (int)distanceInfo.size(); i++) {
        if (distanceInfo[i].inValidSolveRegion) {
            solution[i] = sampleStatistics[counter++].getEstimatedSolution();
        }
    }
}

template <typename T, size_t DIM>
void createBvcEvaluationPoints(const std::vector<wosx::Vector<DIM>>& solveLocations,
                               std::vector<wosx::GPUBVCEvaluationPoint<DIM>>& evalPts)
{
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        wosx::GPUBVCEvaluationPoint<DIM> evalPt;
        if constexpr (DIM == 2) {
            evalPt.pt = wosx::float2{solveLocations[i][0],
                                     solveLocations[i][1]};
            evalPt.normal = wosx::float2{0.0f, 0.0f};

        } else if constexpr (DIM == 3) {
            evalPt.pt = wosx::float3{solveLocations[i][0],
                                     solveLocations[i][1],
                                     solveLocations[i][2]};
            evalPt.normal = wosx::float3{0.0f, 0.0f, 0.0f};
        }

        evalPt.type = wosx::SampleType::InDomain;
        evalPts.emplace_back(evalPt);
    }
}

template <typename T, size_t DIM>
void runBoundaryValueCaching(const json& solverConfig,
                             const std::pair<wosx::Vector<DIM>, wosx::Vector<DIM>>& boundingBox,
                             const std::vector<wosx::Vector<DIM>>& absorbingBoundaryPositions,
                             const std::vector<wosx::Vectori<DIM>>& absorbingBoundaryIndices,
                             const std::vector<wosx::Vector<DIM>>& reflectingBoundaryPositions,
                             const std::vector<wosx::Vectori<DIM>>& reflectingBoundaryIndices,
                             std::shared_ptr<wosx::GPUTaskHandle<T, DIM>> taskHandle,
                             std::shared_ptr<wosx::GPUGeometricQueries<DIM>> queries,
                             std::shared_ptr<wosx::GPUPDE> pde,
                             bool solveDoubleSided,
                             const std::vector<wosx::GPUBVCEvaluationPoint<DIM>>& evalPts,
                             std::vector<wosx::GPUBVCEvaluationOutputs<T, DIM>>& evalOutputs)
{
    // load config settings for wost
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);

    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);
    const int nResidentThreads = getOptional<int>(solverConfig, "nResidentThreads", 131072);

    const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
    const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
    const bool ignoreAbsorbingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreReflectingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreReflectingBoundaryContribution", false);
    const bool ignoreSourceContribution = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    bool disablePersistentThreads = getOptional<bool>(solverConfig, "disablePersistentThreads", false);
    if (!disablePersistentThreads && taskHandle->getDeviceBackend() != "cuda") {
        std::cerr << "Persistent threads require CUDA backend, disabling" << std::endl;
        disablePersistentThreads = true;
    }

    // load config settings for boundary value caching
    const int nWalksForCachedSolutionEstimates = getOptional<int>(solverConfig, "nWalksForCachedSolutionEstimates", 128);
    const int nWalksForCachedGradientEstimates = getOptional<int>(solverConfig, "nWalksForCachedGradientEstimates", 640);
    const int absorbingBoundaryCacheSize = getOptional<int>(solverConfig, "absorbingBoundaryCacheSize", 1024);
    const int reflectingBoundaryCacheSize = getOptional<int>(solverConfig, "reflectingBoundaryCacheSize", 1024);
    int domainCacheSize = getOptional<int>(solverConfig, "domainCacheSize", 1024);

    const float normalOffsetForAbsorbingBoundary = getOptional<float>(solverConfig, "normalOffsetForAbsorbingBoundary",
                                                                      5.0f*epsilonShellForAbsorbingBoundary);
    const float normalOffsetForReflectingBoundary = getOptional<float>(solverConfig, "normalOffsetForReflectingBoundary", 0.0f);
    const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 0.0f);
    const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

    // initialize boundary samplers
    std::function<bool(const wosx::Vector<DIM>&)> insideBoundingDomain =
        [&boundingBox](const wosx::Vector<DIM>& x) -> bool {
        return (x.array() >= boundingBox.first.array()).all() &&
               (x.array() <= boundingBox.second.array()).all();
    };

    std::shared_ptr<wosx::GPUBoundarySampler> absorbingBoundarySampler =
        wosx::createUniformBoundarySampler(queries, absorbingBoundaryPositions,
                                           absorbingBoundaryIndices, insideBoundingDomain,
                                           normalOffsetForAbsorbingBoundary, solveDoubleSided);
    std::shared_ptr<wosx::GPUBoundarySampler> reflectingBoundarySampler =
        wosx::createUniformBoundarySampler(queries, reflectingBoundaryPositions,
                                           reflectingBoundaryIndices, insideBoundingDomain,
                                           normalOffsetForReflectingBoundary, solveDoubleSided);

    // initialize solve region and domain sampler
    std::shared_ptr<wosx::GPUSolveRegion> solveRegion = nullptr;
    if (solveDoubleSided) {
        solveRegion = std::make_shared<wosx::GPUBoundingBoxSolveRegion<DIM>>(
            boundingBox.first, boundingBox.second);

    } else {
        float signedVolumeAbsorbingBoundary = wosx::computeSignedVolume<DIM>(
            absorbingBoundaryPositions, absorbingBoundaryIndices);
        float signedVolumeReflectingBoundary = wosx::computeSignedVolume<DIM>(
            reflectingBoundaryPositions, reflectingBoundaryIndices);
        float regionVolume = std::fabs(signedVolumeAbsorbingBoundary + signedVolumeReflectingBoundary);
        solveRegion = std::make_shared<wosx::GPUWatertightDomainSolveRegion<DIM>>(
            queries, boundingBox.first, boundingBox.second, regionVolume);
    }

    std::shared_ptr<wosx::GPUDomainSampler> domainSampler =
        std::make_shared<wosx::GPUUniformDomainSampler<DIM>>(solveRegion, queries);
    if (ignoreSourceContribution) domainCacheSize = 0;

    // initialize GPU task handle
    taskHandle->setGeometricQueries(queries);
    taskHandle->setAbsorbingBoundarySampler(absorbingBoundarySampler);
    taskHandle->setReflectingBoundarySampler(reflectingBoundarySampler);
    taskHandle->setDomainSampler(domainSampler);
    taskHandle->setPDE(pde);
    taskHandle->init();

    // initialize GPU walk settings
    std::shared_ptr<wosx::GPUWalkSettings> walkSettings = std::make_shared<wosx::GPUWalkSettings>();
    walkSettings->epsilonShellForAbsorbingBoundary = epsilonShellForAbsorbingBoundary;
    walkSettings->epsilonShellForReflectingBoundary = epsilonShellForReflectingBoundary;
    walkSettings->silhouettePrecision = silhouettePrecision;
    walkSettings->russianRouletteThreshold = russianRouletteThreshold;
    walkSettings->maxWalkLength = maxWalkLength;
    walkSettings->stepsBeforeUsingMaximalSpheres = stepsBeforeUsingMaximalSpheres;
    walkSettings->solveDoubleSided = solveDoubleSided ? 1 : 0;
    walkSettings->useGradientControlVariates = disableGradientControlVariates ? 0 : 1;
    walkSettings->useGradientAntitheticVariates = disableGradientAntitheticVariates ? 0 : 1;
    walkSettings->ignoreAbsorbingBoundaryContribution = ignoreAbsorbingBoundaryContribution ? 1 : 0;
    walkSettings->ignoreReflectingBoundaryContribution = ignoreReflectingBoundaryContribution ? 1 : 0;
    walkSettings->ignoreSourceContribution = ignoreSourceContribution ? 1 : 0;

    // initialize GPU boundary value caching solver
    wosx::GPUBoundaryValueCachingSolver<T, DIM> boundaryValueCaching(
        taskHandle, walkSettings, nResidentThreads, !disablePersistentThreads, printLogs);

    // populate evaluation points on the GPU
    boundaryValueCaching.populateEvaluationPoints(evalPts);

    // generate boundary and domain samples
    boundaryValueCaching.generateSamples(absorbingBoundaryCacheSize,
                                         reflectingBoundaryCacheSize,
                                         domainCacheSize);

    // compute sample estimates on the boundary
    boundaryValueCaching.computeSampleEstimates(nWalksForCachedSolutionEstimates,
                                                nWalksForCachedGradientEstimates);

    // splat boundary sample estimates and domain data to evaluation points
    boundaryValueCaching.splat(radiusClampForKernels, regularizationForKernels);

    // extract evaluation outputs from the GPU
    boundaryValueCaching.getEvaluationOutputs(evalOutputs);
}

template <typename T, size_t DIM>
void getSolution(const std::vector<wosx::GPUBVCEvaluationOutputs<T, DIM>>& evalOutputs,
                 std::vector<T>& solution)
{
    solution.resize(evalOutputs.size(), T(0.0f));
    for (int i = 0; i < (int)evalOutputs.size(); i++) {
        solution[i] = evalOutputs[i].solution;
    }
}

template <typename T, size_t DIM>
void runSolver(const std::string& solverType,
               const std::string& deviceBackend,
               const json& solverConfig,
               const std::pair<wosx::Vector<DIM>, wosx::Vector<DIM>>& boundingBox,
               const std::vector<wosx::Vector<DIM>>& absorbingBoundaryPositions,
               const std::vector<wosx::Vectori<DIM>>& absorbingBoundaryIndices,
               const std::vector<wosx::Vector<DIM>>& reflectingBoundaryPositions,
               const std::vector<wosx::Vectori<DIM>>& reflectingBoundaryIndices,
               std::shared_ptr<wosx::GPUGeometricQueries<DIM>> queries,
               std::shared_ptr<wosx::GPUPDE> pde,
               bool solveDoubleSided,
               const std::vector<wosx::Vector<DIM>>& solveLocations,
               const std::vector<DistanceInfo>& distanceInfo,
               std::vector<T>& solution)
{
    // create GPU task handle
    std::filesystem::path wosxDirectoryPath = std::filesystem::current_path().parent_path();
    std::filesystem::path wosxPdeDirectoryPath = wosxDirectoryPath / "demo_apps" / "basic_2d";
    std::shared_ptr<wosx::GPUTaskHandle<T, DIM>> taskHandle =
        std::make_shared<wosx::GPUTaskHandle<T, DIM>>(wosxDirectoryPath.string(),
                                                      wosxPdeDirectoryPath.string(),
                                                      deviceBackend);

    if (solverType == "wos") {
        // create sample points to estimate solution at
        std::vector<wosx::GPUSamplePoint<DIM>> samplePts;
        createSamplePoints<DIM>(solveLocations, distanceInfo, samplePts);

        // run walk on spheres
        std::vector<wosx::GPUSampleStatistics<T, DIM>> sampleStatistics;
        runWalkOnSpheres<T, DIM>(solverConfig, taskHandle, queries, pde,
                                 solveDoubleSided, samplePts, sampleStatistics);

        // extract solution from sample points
        getSolution<T, DIM>(distanceInfo, sampleStatistics, solution);

    } else if (solverType == "wost") {
        // create sample points to estimate solution at
        std::vector<wosx::GPUSamplePoint<DIM>> samplePts;
        createSamplePoints<DIM>(solveLocations, distanceInfo, samplePts);

        // run walk on stars
        std::vector<wosx::GPUSampleStatistics<T, DIM>> sampleStatistics;
        runWalkOnStars<T, DIM>(solverConfig, taskHandle, queries, pde,
                               solveDoubleSided, samplePts, sampleStatistics);

        // extract solution from sample points
        getSolution<T, DIM>(distanceInfo, sampleStatistics, solution);

    } else if (solverType == "bvc") {
        // create evaluation points to estimate solution at
        std::vector<wosx::GPUBVCEvaluationPoint<DIM>> evalPts;
        createBvcEvaluationPoints<T, DIM>(solveLocations, evalPts);

        // run boundary value caching
        std::vector<wosx::GPUBVCEvaluationOutputs<T, DIM>> evalOutputs;
        runBoundaryValueCaching<T, DIM>(solverConfig, boundingBox, absorbingBoundaryPositions,
                                        absorbingBoundaryIndices, reflectingBoundaryPositions,
                                        reflectingBoundaryIndices, taskHandle, queries, pde,
                                        solveDoubleSided, evalPts, evalOutputs);

        // extract solution from evaluation points
        getSolution<T, DIM>(evalOutputs, solution);

    } else {
        std::cerr << "Unknown solver type: " << solverType << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename T>
int runDemo(const json& config)
{
    // load config settings
    const std::string solverType = getOptional<std::string>(config, "solverType", "wost");
    const json modelProblemConfig = getRequired<json>(config, "modelProblem");
    const json solverConfig = getRequired<json>(config, "solver");
    const json outputConfig = getRequired<json>(config, "output");
    std::filesystem::path wosxDirectoryPath = std::filesystem::current_path().parent_path();
    const std::string deviceBackend = getOptional<std::string>(config, "deviceBackend", "cuda");

    // initialize the model problem
    GPUModelProblem<T> modelProblem(modelProblemConfig, wosxDirectoryPath.string());
    const std::pair<Vector2, Vector2>& boundingBox = modelProblem.getBoundingBox();
    const std::vector<Vector2i>& absorbingBoundaryIndices = modelProblem.getAbsorbingBoundaryIndices();
    const std::vector<Vector2i>& reflectingBoundaryIndices = modelProblem.getReflectingBoundaryIndices();
    std::shared_ptr<wosx::GPUGeometricQueries<2>> queries = modelProblem.getGeometricQueries();
    bool domainIsWatertight = modelProblem.domainIsWatertight();
    bool solveDoubleSided = modelProblem.solveDoubleSided();
    bool solveExterior = modelProblem.solveExterior();

    // create solve locations on a grid for this demo
    std::vector<Vector2> solveLocations;
    std::vector<DistanceInfo> distanceInfo;
    createGridPoints(outputConfig, boundingBox, solveLocations);
    computeDistanceInfo<2>(solveLocations, queries, deviceBackend, domainIsWatertight,
                           solveDoubleSided, solveExterior, distanceInfo);

    // solve the model problem
    std::vector<T> solution;
    if (solveExterior) {
        const wosx::KelvinTransform<T, 2>& kelvinTransform = modelProblem.getKelvinTransform();
        const std::pair<Vector2, Vector2>& invertedBoundingBox = modelProblem.getInvertedBoundingBox();
        const std::vector<Vector2>& invertedAbsorbingBoundaryPositions = modelProblem.getInvertedAbsorbingBoundaryPositions();
        const std::vector<Vector2>& invertedReflectingBoundaryPositions = modelProblem.getInvertedReflectingBoundaryPositions();
        std::shared_ptr<wosx::GPUPDE> pdeInvertedDomain = modelProblem.getPDEInvertedDomain();
        std::shared_ptr<wosx::GPUGeometricQueries<2>> queriesInvertedDomain = modelProblem.getGeometricQueriesInvertedDomain();

        // invert the solve locations and update the distance info
        int nSolveLocations = (int)solveLocations.size();
        std::vector<Vector2> invertedSolveLocations(nSolveLocations, Vector2::Zero());
        for (int i = 0; i < nSolveLocations; i++) {
            invertedSolveLocations[i] = kelvinTransform.transformPoint(solveLocations[i]);
        }
        std::vector<DistanceInfo> distanceInfoInvertedDomain;
        computeDistanceInfo<2>(invertedSolveLocations, queriesInvertedDomain,
                               deviceBackend, domainIsWatertight, solveDoubleSided,
                               false, distanceInfoInvertedDomain);

        // run the solver on the inverted domain
        runSolver<T, 2>(solverType, deviceBackend, solverConfig, invertedBoundingBox,
                        invertedAbsorbingBoundaryPositions, absorbingBoundaryIndices,
                        invertedReflectingBoundaryPositions, reflectingBoundaryIndices,
                        queriesInvertedDomain, pdeInvertedDomain, solveDoubleSided,
                        invertedSolveLocations, distanceInfoInvertedDomain, solution);

        // map the solution values back to the exterior domain
        for (int i = 0; i < nSolveLocations; i++) {
            solution[i] = kelvinTransform.transformSolutionEstimate(solution[i], invertedSolveLocations[i]);
        }

    } else {
        const std::vector<Vector2>& absorbingBoundaryPositions = modelProblem.getAbsorbingBoundaryPositions();
        const std::vector<Vector2>& reflectingBoundaryPositions = modelProblem.getReflectingBoundaryPositions();
        std::shared_ptr<wosx::GPUPDE> pde = modelProblem.getPDE();

        // run the solver on the input domain
        runSolver<T, 2>(solverType, deviceBackend, solverConfig, boundingBox,
                        absorbingBoundaryPositions, absorbingBoundaryIndices,
                        reflectingBoundaryPositions, reflectingBoundaryIndices,
                        queries, pde, solveDoubleSided, solveLocations,
                        distanceInfo, solution);
    }

    // save the solution to disk
    saveGridValues<T>(outputConfig, wosxDirectoryPath.string(), distanceInfo, solution);
    return 0;
}

int main(int argc, const char *argv[])
{
    if (argc != 2) {
        std::cerr << "must provide config filename" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::ifstream configFile(argv[1]);
    if (!configFile.is_open()) {
        std::cerr << "Error opening file: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    json config = json::parse(configFile);
    const json modelProblemConfig = getRequired<json>(config, "modelProblem");
    const int channels = getOptional<int>(modelProblemConfig, "channels", 1);
    if (channels == 1) {
        return runDemo<float>(config);

    } else if (channels == 4) {
        return runDemo<Array4>(config);
    }

    std::cerr << "C++ basic_2d GPU demo only supports channels = 1 or channels = 4" << std::endl;
    return EXIT_FAILURE;
}
