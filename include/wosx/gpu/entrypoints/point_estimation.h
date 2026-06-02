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

#pragma once

#include <wosx/gpu/entrypoints/interop/task_handle.h>

namespace wosx {

template <typename T, size_t DIM>
class GPUPointEstimator {
public:
    // constructor
    GPUPointEstimator(std::shared_ptr<GPUTaskHandle<T, DIM>> handle_,
                      GPUPointEstimatorType pointEstimatorType_,
                      uint32_t nResidentThreads_,
                      bool usePersistentThreads_,
                      bool printLogs_);

    // utility function to help improve GPU occupancy when running the solver:
    // given the number of input sample points and number of walks per point,
    // computes the number of sample copies and number of walks per copy
    std::pair<uint32_t, uint32_t> computeWorkloadParameters(uint32_t nInputSamplePoints,
                                                            uint32_t nWalks) const;

    // populates sample points
    void populateSamplePoints(const std::vector<GPUSamplePoint<DIM>>& inputSamplePoints,
                              uint32_t nSampleCopies=1, bool computeBoundaryDistance=true);

    // updates the boundary distance for populated sample points
    void updatePopulatedSamplePointsBoundaryDistance();

    // runs point estimator; nDispatchCalls is a utility parameter to help
    // reduce GPU divergence when random walks have varying lengths: setting
    // nDispatchCalls > 1 (and nWalks := nWalksPerSampleCopy / nDispatchCalls)
    // will run the solver in multiple dispatch calls
    void solve(uint32_t nWalks, uint32_t nDispatchCalls=1);

    // resets sample statistics
    void resetSampleStatistics();

    // returns sample statistics (resized to be the same size as input sample points)
    void getSampleStatistics(std::vector<GPUSampleStatistics<T, DIM>>& outputStatistics);

    // returns true if using persistent threads
    bool usingPersistentThreads() const;

    // enables/disables logging
    void enableLogging(bool enable);

protected:
    // members
    std::shared_ptr<GPUTaskHandle<T, DIM>> handle;
    GPUContext& context;
    std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries;
    uint32_t nOutputSamplePoints;
    uint32_t nSamplePoints;
    GPUBuffer samplePoints;
    GPUBuffer sampleStatistics;
    GPUBuffer globalCounter;
    GPUBuffer locks;
    ComputeShader computeSampleBoundaryDistanceShader;
    ComputeShader seedRngsShader;
    ComputeShader runPointEstimatorShader;
    ComputeShader resetSampleStatisticsShader;
    ComputeShader getSampleStatisticsShader;
    GPUSeedRngs seedRngsEntryPoint;
    std::function<void(const ComputeShader&, const ShaderCursor&)> bindComputeSampleBoundaryDistanceResources;
    std::function<void(const ComputeShader&, const ShaderCursor&)> bindRunPointEstimatorResources;
    uint32_t nResidentThreads;
    uint32_t nThreadsPerGroup;
    bool usePersistentThreads;
    bool printLogs;
};

template <typename T, size_t DIM>
class GPUWalkOnSpheresSolver: public GPUPointEstimator<T, DIM> {
public:
    // constructor
    GPUWalkOnSpheresSolver(std::shared_ptr<GPUTaskHandle<T, DIM>> handle_,
                           std::shared_ptr<GPUWalkSettings> walkSettings_,
                           uint32_t nResidentThreads_=131072,
                           bool usePersistentThreads_=true,
                           bool printLogs_=false);

private:
    // members
    GPUWalkOnSpheres<T, DIM> walkOnSpheres;
};

template <typename T, size_t DIM>
class GPUWalkOnStarsSolver: public GPUPointEstimator<T, DIM> {
public:
    // constructor
    GPUWalkOnStarsSolver(std::shared_ptr<GPUTaskHandle<T, DIM>> handle_,
                         std::shared_ptr<GPUWalkSettings> walkSettings_,
                         uint32_t nResidentThreads_=131072,
                         bool usePersistentThreads_=true,
                         bool printLogs_=false);

private:
    // members
    GPUWalkOnStars<T, DIM> walkOnStars;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
GPUPointEstimator<T, DIM>::GPUPointEstimator(std::shared_ptr<GPUTaskHandle<T, DIM>> handle_,
                                             GPUPointEstimatorType pointEstimatorType_,
                                             uint32_t nResidentThreads_,
                                             bool usePersistentThreads_,
                                             bool printLogs_):
handle(handle_),
context(handle->getContext()),
geometricQueries(handle->getGeometricQueries()),
nResidentThreads(nResidentThreads_),
nThreadsPerGroup(256),
usePersistentThreads(usePersistentThreads_),
printLogs(printLogs_)
{
    // initialize shader programs
    GPUModule mainModule;
    std::string pointEstimatorShaderName = pointEstimatorType_ == GPUPointEstimatorType::WalkOnSpheres ?
                                                                  "walk-on-spheres.cs.slang" :
                                                                  "walk-on-stars.cs.slang";
    mainModule.load(context, handle->getWosxGpuDirectoryPath() + "/entrypoints/" + pointEstimatorShaderName);
    const std::vector<GPUModule>& libraryModules = handle->getLibraryModules();
    computeSampleBoundaryDistanceShader.loadProgram(context, mainModule, libraryModules, "computeSampleBoundaryDistance");
    seedRngsShader.loadProgram(context, mainModule, libraryModules, "seedRngs");
    std::string runPointEstimatorShaderName = usePersistentThreads ? "runPersistentPointEstimator" : "runPointEstimator";
    runPointEstimatorShader.loadProgram(context, mainModule, libraryModules, runPointEstimatorShaderName);
    resetSampleStatisticsShader.loadProgram(context, mainModule, libraryModules, "resetSampleStatistics");
    getSampleStatisticsShader.loadProgram(context, mainModule, libraryModules, "getSampleStatistics");

    // bind shader resources
    bindComputeSampleBoundaryDistanceResources = [this](const ComputeShader& shader,
                                                        const ShaderCursor& cursor) {
        ComPtr<IShaderObject> geometricQueriesShaderObject = shader.createShaderObject(
            context, geometricQueries->getReflectionType());
        ShaderCursor geometricQueriesShaderCursor(geometricQueriesShaderObject);
        geometricQueries->setResources(geometricQueriesShaderCursor, printLogs);
        cursor.getPath("gGeometricQueries").setObject(geometricQueriesShaderObject);
    };
}

template <typename T, size_t DIM>
std::pair<uint32_t, uint32_t> GPUPointEstimator<T, DIM>::computeWorkloadParameters(uint32_t nInputSamplePoints,
                                                                                   uint32_t nWalks) const
{
    // decide how many copies of every sample we need: we want at least
    // ceil(nResidentThreads / nInputSamplePoints) threads per sample point,
    // but never more than the number of walks we eventually want
    uint32_t nSampleCopies = 1;
    if (!usePersistentThreads) {
        if (nInputSamplePoints > 0 && nInputSamplePoints < nResidentThreads) {
            nSampleCopies = (nResidentThreads + nInputSamplePoints - 1)/nInputSamplePoints;
            nSampleCopies = std::min(nSampleCopies, nWalks);
        }
    }

    // decide how many walks does each sample copy performs
    uint32_t nWalksPerSampleCopy = (nWalks + nSampleCopies - 1)/nSampleCopies;

    // print logs
    if (printLogs) {
        std::cout << "Input workload: " << std::to_string(nInputSamplePoints*nWalks) << std::endl;
        std::cout << "\tnInputSamplePoints: " << std::to_string(nInputSamplePoints) << std::endl;
        std::cout << "\tnWalksPerInput: " << std::to_string(nWalks) << std::endl;
        std::cout << "Output workload: " << std::to_string(nInputSamplePoints*nSampleCopies*nWalksPerSampleCopy) << std::endl;
        std::cout << "\tnSamplePoints: " << std::to_string(nInputSamplePoints*nSampleCopies) << std::endl;
        std::cout << "\tnSampleCopies: " << std::to_string(nSampleCopies) << std::endl;
        std::cout << "\tnWalksPerCopy: " << std::to_string(nWalksPerSampleCopy) << std::endl;
    }

    // return the pair <copies, walks-per-copy>
    return { nSampleCopies, nWalksPerSampleCopy };
}

template <typename T, size_t DIM>
void GPUPointEstimator<T, DIM>::populateSamplePoints(const std::vector<GPUSamplePoint<DIM>>& inputSamplePoints,
                                                     uint32_t nSampleCopies, bool computeBoundaryDistance)
{
    // allocate sample points and statistics
    nOutputSamplePoints = (uint32_t)inputSamplePoints.size();
    nSamplePoints = nOutputSamplePoints*nSampleCopies;
    samplePoints.allocate<GPUSamplePoint<DIM>>(context, computeBoundaryDistance, inputSamplePoints);
    sampleStatistics.allocate<GPUSampleStatistics<T, DIM>>(context.device, true, nullptr, nSamplePoints);

    // allocate locks buffer for persistent threads
    if (usePersistentThreads) {
        std::vector<uint32_t> locksData(nOutputSamplePoints, 0);
        locks.allocate<uint32_t>(context, true, locksData);
    }

    if (computeBoundaryDistance) {
        updatePopulatedSamplePointsBoundaryDistance();
    }

    // run seed rngs shader: persistent threads need one RNG per thread,
    // non-persistent needs one RNG per sample point
    uint32_t nRngs = usePersistentThreads ? nResidentThreads : nSamplePoints;
    seedRngsEntryPoint.allocate(context, nRngs);
    uint32_t nThreadGroups = countThreadGroups(nRngs, nThreadsPerGroup, printLogs);
    runShader<GPUSeedRngs>(context, seedRngsShader, seedRngsEntryPoint,
                           {}, nThreadGroups, 1, printLogs);

    // reset sample statistics
    resetSampleStatistics();
}

template <typename T, size_t DIM>
void GPUPointEstimator<T, DIM>::updatePopulatedSamplePointsBoundaryDistance()
{
    // initialize entry point data
    GPUComputeSampleBoundaryDistance entryPoint(samplePoints, nOutputSamplePoints);

    // run shader
    uint32_t nThreadGroups = countThreadGroups(nOutputSamplePoints, nThreadsPerGroup, printLogs);
    runShader<GPUComputeSampleBoundaryDistance>(context, computeSampleBoundaryDistanceShader,
                                                entryPoint, bindComputeSampleBoundaryDistanceResources,
                                                nThreadGroups, 1, printLogs);
}

template <typename T, size_t DIM>
void GPUPointEstimator<T, DIM>::solve(uint32_t nWalks, uint32_t nDispatchCalls)
{
    if (usePersistentThreads) {
        uint64_t nTotalWorkItems = (uint64_t)(nOutputSamplePoints)*nWalks;
        uint32_t nPersistentThreads = nResidentThreads;

        // reset global counter to 0 (locks are already zero from
        // populateSamplePoints or the previous solve call)
        std::vector<uint64_t> counterData = {0ULL};
        globalCounter.allocate<uint64_t>(context, true, counterData);

        // initialize entry point data
        GPURunPersistentPointEstimator entryPoint(samplePoints, seedRngsEntryPoint.getBuffer(),
                                                  sampleStatistics, globalCounter, locks,
                                                  nTotalWorkItems, nOutputSamplePoints,
                                                  nPersistentThreads);

        // run shader
        uint32_t nThreadGroups = countThreadGroups(nPersistentThreads, nThreadsPerGroup, printLogs);
        runShader<GPURunPersistentPointEstimator>(context, runPointEstimatorShader,
                                                  entryPoint, bindRunPointEstimatorResources,
                                                  nThreadGroups, 1, printLogs);

    } else {
        // initialize entry point data
        GPURunPointEstimator entryPoint(samplePoints, seedRngsEntryPoint.getBuffer(),
                                        sampleStatistics, nOutputSamplePoints,
                                        nSamplePoints, nWalks);

        // run shader
        uint32_t nThreadGroups = countThreadGroups(nSamplePoints, nThreadsPerGroup, printLogs);
        runShader<GPURunPointEstimator>(context, runPointEstimatorShader,
                                        entryPoint, bindRunPointEstimatorResources,
                                        nThreadGroups, nDispatchCalls, printLogs);
    }
}

template <typename T, size_t DIM>
void GPUPointEstimator<T, DIM>::resetSampleStatistics()
{
    // initialize entry point data
    GPUResetSampleStatistics entryPoint(sampleStatistics, nSamplePoints);

    // run shader
    uint32_t nThreadGroups = countThreadGroups(nSamplePoints, nThreadsPerGroup, printLogs);
    runShader<GPUResetSampleStatistics>(context, resetSampleStatisticsShader,
                                        entryPoint, {}, nThreadGroups, 1, printLogs);
}

template <typename T, size_t DIM>
void GPUPointEstimator<T, DIM>::getSampleStatistics(std::vector<GPUSampleStatistics<T, DIM>>& outputStatistics)
{
    // allocate entry point data
    GPUGetSampleStatistics<T, DIM> entryPoint(sampleStatistics, nSamplePoints, nOutputSamplePoints);
    entryPoint.allocate(context);

    // run shader
    uint32_t nThreadGroups = countThreadGroups(nOutputSamplePoints, nThreadsPerGroup, printLogs);
    runShader<GPUGetSampleStatistics<T, DIM>>(context, getSampleStatisticsShader,
                                              entryPoint, {}, nThreadGroups, 1, printLogs);

    // read results from GPU
    entryPoint.read(context, outputStatistics);
}

template <typename T, size_t DIM>
bool GPUPointEstimator<T, DIM>::usingPersistentThreads() const
{
    return usePersistentThreads;
}

template <typename T, size_t DIM>
void GPUPointEstimator<T, DIM>::enableLogging(bool enable)
{
    printLogs = enable;
}

template <typename T, size_t DIM>
GPUWalkOnSpheresSolver<T, DIM>::GPUWalkOnSpheresSolver(std::shared_ptr<GPUTaskHandle<T, DIM>> handle_,
                                                       std::shared_ptr<GPUWalkSettings> walkSettings_,
                                                       uint32_t nResidentThreads_,
                                                       bool usePersistentThreads_,
                                                       bool printLogs_):
GPUPointEstimator<T, DIM>(handle_,
                          GPUPointEstimatorType::WalkOnSpheres,
                          nResidentThreads_,
                          usePersistentThreads_,
                          printLogs_),
walkOnSpheres(handle_->getPDE(),
              handle_->getGeometricQueries(),
              walkSettings_)
{
    // bind shader resources
    this->bindRunPointEstimatorResources = [this](const ComputeShader& shader,
                                                  const ShaderCursor& cursor) {
        ComPtr<IShaderObject> walkOnSpheresShaderObject = shader.createShaderObject(
            this->context, walkOnSpheres.getReflectionType());
        ShaderCursor walkOnSpheresShaderCursor(walkOnSpheresShaderObject);
        walkOnSpheres.setResources(walkOnSpheresShaderCursor, this->printLogs);
        cursor.getPath("gPointEstimator").setObject(walkOnSpheresShaderObject);
    };
}

template <typename T, size_t DIM>
GPUWalkOnStarsSolver<T, DIM>::GPUWalkOnStarsSolver(std::shared_ptr<GPUTaskHandle<T, DIM>> handle_,
                                                   std::shared_ptr<GPUWalkSettings> walkSettings_,
                                                   uint32_t nResidentThreads_,
                                                   bool usePersistentThreads_,
                                                   bool printLogs_):
GPUPointEstimator<T, DIM>(handle_,
                          GPUPointEstimatorType::WalkOnStars,
                          nResidentThreads_,
                          usePersistentThreads_,
                          printLogs_),
walkOnStars(handle_->getPDE(),
            handle_->getGeometricQueries(),
            walkSettings_)
{
    // bind shader resources
    this->bindRunPointEstimatorResources = [this](const ComputeShader& shader,
                                                  const ShaderCursor& cursor) {
        ComPtr<IShaderObject> walkOnStarsShaderObject = shader.createShaderObject(
            this->context, walkOnStars.getReflectionType());
        ShaderCursor walkOnStarsShaderCursor(walkOnStarsShaderObject);
        walkOnStars.setResources(walkOnStarsShaderCursor, this->printLogs);
        cursor.getPath("gPointEstimator").setObject(walkOnStarsShaderObject);
    };
}

} // wosx
