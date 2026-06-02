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

// This demo solves an electrostatics problem around a MEMS comb-drive actuator
// using WoSX's GPU walk on stars solver. The lower anchored comb is grounded,
// while the upper comb has a prescribed voltage and is driven by a sinusoidal
// displacement. For each frame, the demo estimates the electric potential and
// electric field on a slice plane and visualizes the result in Polyscope when
// 'visualizeProblem' is enabled in the config; otherwise, it saves the results
// as PNG images.

#include <wosx/wosx_gpu.h>
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "config.h"
#include "image.h"
#include "colormap.h"
#include <filesystem>
#include <fstream>
#include <limits>
#include <queue>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility structs and functions

struct MovableCombState {
    // members
    float voltage = 0.0f;
    float displacementAmplitude = 0.025f;
    int frameIndex = 0;
    int nFrames = 120;
    bool isMoving = false;

    // increment frame index and wrap around if necessary
    void incrementFrameIndex();
};

void MovableCombState::incrementFrameIndex()
{
    frameIndex += 1;
    if (frameIndex >= nFrames) {
        frameIndex -= nFrames;
    }
}

struct MeshData {
    // members
    std::vector<wosx::Vector3> basePositions;
    std::vector<wosx::Vector3> currentPositions;
    std::vector<wosx::Vector3i> indices;
    std::vector<uint32_t> isMovableFace;
    std::vector<bool> isMovableVertex;

    std::pair<wosx::Vector3, wosx::Vector3> boundingBox;
    std::vector<wosx::Vector3> boxPositions;
    std::vector<wosx::Vector3i> boxIndices;

    // helper functions
    float computeMovableCombDisplacement(const MovableCombState& combState);
    void updateMovableComb(float displacement);
};

float MeshData::computeMovableCombDisplacement(const MovableCombState& combState)
{
    float time = (float)combState.frameIndex/(float)combState.nFrames;
    return combState.displacementAmplitude*std::sin(2.0f*M_PI*time);
}

void MeshData::updateMovableComb(float displacement)
{
    currentPositions = basePositions;
    for (int i = 0; i < (int)currentPositions.size(); i++) {
        if (isMovableVertex[i]) {
            currentPositions[i] += displacement*wosx::Vector3::UnitY();
        }
    }
}

struct SlicePlaneData {
    // members
    std::vector<wosx::Vector3> positions;
    std::vector<std::vector<size_t>> indices;

    // builds slice plane mesh
    void build(const std::pair<wosx::Vector3, wosx::Vector3>& boundingBox,
               int resolutionPow2);

    // returns the cell centers of the slice plane
    std::vector<wosx::Vector3> getCellCenters() const;
};

void SlicePlaneData::build(const std::pair<wosx::Vector3, wosx::Vector3>& boundingBox,
                           int resolutionPow2)
{
    // compute positions and indices
    indices.clear();
    positions.clear();
    int size = 1 << resolutionPow2;

    for (size_t h = 0; h < size + 1; h++) {
        for (size_t w = 0; w < size + 1; w++) {
            size_t index = positions.size();
            positions.emplace_back(wosx::Vector3::Zero());
            positions[index](0) = w;
            positions[index](1) = h;

            if (h != size && w != size) {
                indices.emplace_back(std::vector<size_t>{index, index + 1,
                                                         index + size + 2,
                                                         index + size + 1});
            }
        }
    }

    // normalize, scale and shift positions
    wosx::Vector3 center = 0.5f*(boundingBox.first + boundingBox.second);
    wosx::Vector3 extent = boundingBox.second - boundingBox.first;
    center(2) = 0.0f;
    extent(2) = 0.0f;
    float scale = 0.125f*extent.norm();

    wosx::normalize<3>(positions);
    for (int i = 0; i < (int)positions.size(); i++) {
        positions[i] *= scale;
        positions[i] += center;
    }
}

std::vector<wosx::Vector3> SlicePlaneData::getCellCenters() const
{
    std::vector<wosx::Vector3> cellCenters;
    for (int i = 0; i < (int)indices.size(); i++) {
        const std::vector<size_t>& index = indices[i];
        wosx::Vector3 center = wosx::Vector3::Zero();
        int nIndex = (int)index.size();
        for (int j = 0; j < nIndex; j++) {
            center += positions[index[j]];
        }

        center = center/nIndex;
        cellCenters.emplace_back(center);
    }

    return cellCenters;
}

std::string getFilename(const std::string& directoryPath,
                        const json& config,
                        const std::string& filename,
                        bool isRequired=true,
                        const std::string& defaultValue="")
{
    return directoryPath + "/" + (isRequired ? getRequired<std::string>(config, filename) :
                                               getOptional<std::string>(config, filename, defaultValue));
}

std::string getFrameFilename(const std::string& filename, int frameIndex)
{
    std::filesystem::path path(filename);
    std::ostringstream frameFilename;
    frameFilename << path.stem().string() << "_"
                  << std::setw(4) << std::setfill('0') << frameIndex
                  << path.extension().string();

    return (path.parent_path() / frameFilename.str()).string();
}

void saveColormappedImage(const std::string& filename,
                          int imageHeight, int imageWidth,
                          const std::vector<float>& values,
                          const std::string& colormap,
                          float colormapMinVal,
                          float colormapMaxVal)
{
    // fill a temporary scalar image for colormapping
    std::shared_ptr<Image<1>> image = std::make_shared<Image<1>>(imageHeight, imageWidth);
    for (int i = 0; i < (int)values.size(); i++) {
        int row = imageHeight - i/imageWidth - 1;
        int col = i%imageWidth;
        image->get(row, col)[0] = values[i];
    }

    // write the colormapped image to disk
    std::filesystem::path path(filename);
    std::filesystem::create_directories(path.parent_path());
    getColormappedImage(image, colormap, colormapMinVal, colormapMaxVal)->write(filename);
}

void saveSolutionAndGradientNormImages(const std::string& directoryPath,
                                       const json& outputConfig,
                                       int frameIndex,
                                       int sliceResolutionPow2,
                                       const std::vector<float>& electricPotential,
                                       const std::vector<float>& electricFieldMagnitude)
{
    // get the filenames and colormap parameters for the electric potential
    const std::string electricPotentialFile = getFrameFilename(
        getFilename(directoryPath, outputConfig, "electricPotentialFile",
                    false, "solutions/electric_potential.png"), frameIndex);
    const std::string electricPotentialColormap = getOptional<std::string>(
        outputConfig, "electricPotentialColormap", "plasma");
    const float electricPotentialColormapMinVal = getOptional<float>(
        outputConfig, "electricPotentialColormapMinVal", 0.0f);
    const float electricPotentialColormapMaxVal = getOptional<float>(
        outputConfig, "electricPotentialColormapMaxVal", 100.0f);

    // get the filenames and colormap parameters for the electric field magnitude
    const std::string electricFieldMagnitudeFile = getFrameFilename(
        getFilename(directoryPath, outputConfig, "electricFieldMagnitudeFile",
                    false, "solutions/electric_field_magnitude.png"), frameIndex);
    const std::string electricFieldMagnitudeColormap = getOptional<std::string>(
        outputConfig, "electricFieldMagnitudeColormap", "turbo");
    const float electricFieldMagnitudeColormapMinVal = getOptional<float>(
        outputConfig, "electricFieldMagnitudeColormapMinVal", 0.0f);
    const float electricFieldMagnitudeColormapMaxVal = getOptional<float>(
        outputConfig, "electricFieldMagnitudeColormapMaxVal", 15000.0f);

    // save the images
    const int sliceResolution = 1 << sliceResolutionPow2;
    saveColormappedImage(electricPotentialFile, sliceResolution, sliceResolution,
                         electricPotential, electricPotentialColormap,
                         electricPotentialColormapMinVal, electricPotentialColormapMaxVal);
    saveColormappedImage(electricFieldMagnitudeFile, sliceResolution, sliceResolution,
                         electricFieldMagnitude, electricFieldMagnitudeColormap,
                         electricFieldMagnitudeColormapMinVal, electricFieldMagnitudeColormapMaxVal);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Problem specification - load mesh data, setup the GPU geometric queries and PDE objects

void loadMeshData(const std::string& directoryPath,
                  const json& problemConfig,
                  MeshData& meshData)
{
    // load the mesh
    std::string geometryFilename = getFilename(directoryPath, problemConfig, "geometry");
    wosx::loadBoundaryMesh<3>(geometryFilename, meshData.basePositions, meshData.indices);
    wosx::normalize<3>(meshData.basePositions); // normalize to a unit sphere

    // classify the mesh components into movable and fixed
    auto partitionIntoComponents = [](int nPositions,
                                      const std::vector<wosx::Vector3i>& indices) -> std::vector<uint32_t> {
        // build a map of adjacent faces for each vertex
        std::vector<std::vector<int>> adjacentFaces(nPositions);
        for (int t = 0; t < (int)indices.size(); t++) {
            for (int c = 0; c < 3; c++) {
                int vIndex = indices[t][c];
                adjacentFaces[vIndex].emplace_back(t);
            }
        }

        // partition the faces into components
        std::vector<int> components(indices.size(), -1);
        int component = 0;
        for (int t = 0; t < (int)indices.size(); t++) {
            if (components[t] >= 0) continue;

            std::queue<int> queue;
            queue.push(t);
            components[t] = component;
            while (!queue.empty()) {
                int fIndex = queue.front();
                queue.pop();

                for (int c = 0; c < 3; c++) {
                    int vIndex = indices[fIndex][c];
                    for (int neighborFace: adjacentFaces[vIndex]) {
                        if (components[neighborFace] < 0) {
                            components[neighborFace] = component;
                            queue.push(neighborFace);
                        }
                    }
                }
            }

            component++;
        }

        return std::vector<uint32_t>(components.begin(), components.end());
    };

    int nPositions = (int)meshData.basePositions.size();
    meshData.currentPositions = meshData.basePositions;
    meshData.isMovableFace = partitionIntoComponents(nPositions, meshData.indices);
    meshData.isMovableVertex.resize(nPositions, false);
    for (int t = 0; t < (int)meshData.indices.size(); t++) {
        if (meshData.isMovableFace[t] == 1) {
            for (int c = 0; c < 3; c++) {
                int vIndex = meshData.indices[t][c];
                meshData.isMovableVertex[vIndex] = true;
            }
        }
    }

    // compute a bounding box for the mesh
    meshData.boundingBox = wosx::computeBoundingBox<3>(meshData.basePositions, true, 1.5f);
    wosx::buildBoundingBoxMesh<3>(meshData.boundingBox.first, meshData.boundingBox.second,
                                  meshData.boxPositions, meshData.boxIndices);
}

std::shared_ptr<wosx::GPUGeometricQueries<3>> setupGeometricQueries(const MeshData& meshData)
{
    // setup an absorbing boundary handler for the mesh
    std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler =
        std::make_shared<wosx::GPUFcpwDirichletBoundaryHandler<3>>(
            meshData.currentPositions, meshData.indices);

    // setup a reflecting boundary handler for the mesh's bounding box
    std::function<bool(float, int)> ignoreCandidateSilhouette =
        wosx::getIgnoreCandidateSilhouetteCallback(false);
    std::shared_ptr<wosx::GPUReflectingBoundaryHandler> reflectingBoundaryHandler =
        std::make_shared<wosx::GPUFcpwNeumannBoundaryHandler<3>>(
            meshData.boxPositions, meshData.boxIndices, ignoreCandidateSilhouette);

    // create a geometric queries object from the handlers
    return std::make_shared<wosx::GPUGeometricQueries<3>>(
        absorbingBoundaryHandler, reflectingBoundaryHandler,
        meshData.boundingBox.first, meshData.boundingBox.second, true);
}

class GPUElectrostaticsPDE: public wosx::GPUPDE {
public:
    // constructor
    GPUElectrostaticsPDE(std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler> accelerationStructure_,
                         const std::vector<uint32_t>& isMovableFace_,
                         float movableCombVoltage_);

    // update acceleration structure
    void updateAccelerationStructure(std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler> accelerationStructure_);

    // update movable comb voltage
    void updateMovableCombVoltage(float movableCombVoltage_);

    // allocates and sets GPU resources, and returns type info
    void allocate(wosx::GPUContext& context);
    void setResources(const wosx::ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    wosx::GPUPDEType getType() const;

private:
    // members
    std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler> accelerationStructure;
    std::vector<uint32_t> isMovableFace;
    wosx::GPUBuffer isMovableFaceBuffer = {};
    float movableCombVoltage;
};

GPUElectrostaticsPDE::GPUElectrostaticsPDE(std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler> accelerationStructure_,
                                           const std::vector<uint32_t>& isMovableFace_,
                                           float movableCombVoltage_):
accelerationStructure(accelerationStructure_),
isMovableFace(isMovableFace_),
movableCombVoltage(movableCombVoltage_)
{
    // nothing to do
}

void GPUElectrostaticsPDE::updateAccelerationStructure(std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler> accelerationStructure_)
{
    accelerationStructure = accelerationStructure_;
}

void GPUElectrostaticsPDE::updateMovableCombVoltage(float movableCombVoltage_)
{
    movableCombVoltage = movableCombVoltage_;
}

void GPUElectrostaticsPDE::allocate(wosx::GPUContext& context)
{
    isMovableFaceBuffer.allocate<uint32_t>(context, false, isMovableFace);
}

void GPUElectrostaticsPDE::setResources(const wosx::ShaderCursor& cursor, bool printLogs) const
{
    accelerationStructure->setResources(cursor["mAccelerationStructure"], printLogs);
    cursor["mIsMovableFace"].setBinding(isMovableFaceBuffer.buffer);
    cursor["mMovableCombVoltage"].setData(movableCombVoltage);
    if (printLogs) wosx::printReflectionInfo(cursor, 3, getReflectionType());
}

std::string GPUElectrostaticsPDE::getReflectionType() const
{
    return "ElectrostaticsPDE";
}

wosx::GPUPDEType GPUElectrostaticsPDE::getType() const
{
    return wosx::GPUPDEType::Poisson;
}

void updateMovableCombPosition(MeshData& meshData, float displacement,
                               std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle,
                               std::shared_ptr<wosx::GPUGeometricQueries<3>> geometricQueries,
                               std::shared_ptr<GPUElectrostaticsPDE> pde)
{
    // update the movable comb position
    meshData.updateMovableComb(displacement);

    // update the absorbing boundary handler for the geometric queries object
    std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler =
        std::make_shared<wosx::GPUFcpwDirichletBoundaryHandler<3>>(
            meshData.currentPositions, meshData.indices, false);
    geometricQueries->reallocateAbsorbingBoundary(taskHandle->getContext(),
                                                  absorbingBoundaryHandler,
                                                  meshData.boundingBox.first,
                                                  meshData.boundingBox.second, true);

    // update the PDE acceleration structure
    pde->updateAccelerationStructure(absorbingBoundaryHandler);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Walk on stars solver - create sample points, run the solver and extract the solution and gradient

void createSamplePoints(const SlicePlaneData& slicePlaneData,
                        std::vector<wosx::GPUSamplePoint<3>>& samplePts)
{
    samplePts.clear();
    std::vector<wosx::Vector3> solveLocations = slicePlaneData.getCellCenters();
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        wosx::GPUSamplePoint<3> samplePt;
        samplePt.pt = wosx::float3{solveLocations[i][0],
                                   solveLocations[i][1],
                                   solveLocations[i][2]};
        samplePt.normal = wosx::float3{0.0f, 0.0f, 0.0f};
        samplePt.type = wosx::SampleType::InDomain;
        samplePt.estimationQuantity = wosx::EstimationQuantity::SolutionAndGradient;
        samplePts.emplace_back(samplePt);
    }
}

std::shared_ptr<wosx::GPUWalkOnStarsSolver<float, 3>> setupSolver(const json& solverConfig,
                                                                  std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle,
                                                                  const std::vector<wosx::GPUSamplePoint<3>>& samplePts,
                                                                  std::pair<uint32_t, uint32_t>& workloadParameters)
{
    // load config settings for walk on stars
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);

    const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 10000);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);
    const int nResidentThreads = getOptional<int>(solverConfig, "nResidentThreads", 131072);

    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    bool disablePersistentThreads = getOptional<bool>(solverConfig, "disablePersistentThreads", false);
    if (!disablePersistentThreads && taskHandle->getDeviceBackend() != "cuda") {
        std::cerr << "Persistent threads require CUDA backend, disabling" << std::endl;
        disablePersistentThreads = true;
    }

    // setup walk settings for the solver
    std::shared_ptr<wosx::GPUWalkSettings> walkSettings = std::make_shared<wosx::GPUWalkSettings>();
    walkSettings->epsilonShellForAbsorbingBoundary = epsilonShellForAbsorbingBoundary;
    walkSettings->epsilonShellForReflectingBoundary = epsilonShellForReflectingBoundary;
    walkSettings->silhouettePrecision = silhouettePrecision;
    walkSettings->russianRouletteThreshold = 0.0f;
    walkSettings->maxWalkLength = maxWalkLength;
    walkSettings->stepsBeforeUsingMaximalSpheres = stepsBeforeUsingMaximalSpheres;
    walkSettings->solveDoubleSided = 0;
    walkSettings->useGradientControlVariates = 1;
    walkSettings->useGradientAntitheticVariates = 1;
    walkSettings->ignoreAbsorbingBoundaryContribution = 0;
    walkSettings->ignoreReflectingBoundaryContribution = 1; // demo uses perfectly insulating (i.e., zero) Neumann conditions
    walkSettings->ignoreSourceContribution = 1;

    // initialize the solver
    std::shared_ptr<wosx::GPUWalkOnStarsSolver<float, 3>> walkOnStars =
        std::make_shared<wosx::GPUWalkOnStarsSolver<float, 3>>(
            taskHandle, walkSettings, nResidentThreads,
            !disablePersistentThreads, printLogs);

    // compute workload parameters--the number of sample copies and number of
    // walks per copy--to help improve GPU occupancy when running the solver
    if (disablePersistentThreads) {
        workloadParameters = walkOnStars->computeWorkloadParameters(samplePts.size(), nWalks);

    } else {
        workloadParameters = std::make_pair(1, nWalks);
    }

    // populate sample points on the GPU
    walkOnStars->populateSamplePoints(samplePts, workloadParameters.first, false);

    return walkOnStars;
}

void runSolver(std::shared_ptr<wosx::GPUWalkOnStarsSolver<float, 3>> walkOnStars,
               std::pair<uint32_t, uint32_t> workloadParameters,
               std::vector<wosx::GPUSampleStatistics<float, 3>>& sampleStatistics)
{
    // update the boundary distance for the populated sample points
    walkOnStars->updatePopulatedSamplePointsBoundaryDistance();

    // reset the sample statistics from the previous solve
    walkOnStars->resetSampleStatistics();

    // solve
    if (walkOnStars->usingPersistentThreads()) {
        walkOnStars->solve(workloadParameters.second);

    } else {
        walkOnStars->solve(1, workloadParameters.second);
    }

    // extract sample statistics from the GPU
    walkOnStars->getSampleStatistics(sampleStatistics);
}

void getSolutionAndGradientNorm(const std::vector<wosx::GPUSampleStatistics<float, 3>>& sampleStatistics,
                                std::vector<float>& electricPotential,
                                std::vector<float>& electricFieldMagnitude)
{
    int nSamples = (int)sampleStatistics.size();
    electricPotential.resize(nSamples);
    electricFieldMagnitude.resize(nSamples);

    for (int i = 0; i < nSamples; i++) {
        const wosx::GPUSampleStatistics<float, 3>& statistics = sampleStatistics[i];
        electricPotential[i] = statistics.getEstimatedSolution();
        const float gx = statistics.getEstimatedGradient(0);
        const float gy = statistics.getEstimatedGradient(1);
        const float gz = statistics.getEstimatedGradient(2);
        electricFieldMagnitude[i] = std::sqrt(std::sqrt(gx*gx + gy*gy + gz*gz)); // NOTE: compressing high-field peaks for visualization
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Problem Visualization

void setCameraView(const MeshData& meshData)
{
    const wosx::Vector3& pMin = meshData.boundingBox.first;
    const wosx::Vector3& pMax = meshData.boundingBox.second;
    wosx::Vector3 center = 0.5f*(pMin + pMax);
    float viewDistance = 0.6f*std::max(pMax[0] - pMin[0], pMax[1] - pMin[1]);

    glm::vec3 target(center[0], center[1], center[2]);
    glm::vec3 camera(center[0], center[1], center[2] + viewDistance);
    polyscope::view::lookAt(camera, target, glm::vec3(0.0f, 1.0f, 0.0f), false);
}

void plotCombVoltage(const json& outputConfig,
                     const MeshData& meshData,
                     float movableCombVoltage)
{
    auto mesh = polyscope::getSurfaceMesh("Mesh");
    std::vector<float> voltages(meshData.isMovableFace.size(), 0.0f);
    for (int t = 0; t < (int)meshData.isMovableFace.size(); t++) {
        voltages[t] = meshData.isMovableFace[t] == 1 ? movableCombVoltage : 0.0f;
    }
    auto voltageQuantity = mesh->addFaceScalarQuantity("Voltage", voltages);
    voltageQuantity->setColorMap("reds");
    const float voltageColormapMinVal = getOptional<float>(
        outputConfig, "electricPotentialColormapMinVal", 0.0f);
    const float voltageColormapMaxVal = getOptional<float>(
        outputConfig, "electricPotentialColormapMaxVal", 100.0f);
    voltageQuantity->setMapRange({voltageColormapMinVal, voltageColormapMaxVal});
    voltageQuantity->setEnabled(true);
}

void plotSolutionAndGradient(const json& outputConfig,
                             const std::vector<float>& electricPotential,
                             const std::vector<float>& electricFieldMagnitude)
{
    auto slicePlane = polyscope::getSurfaceMesh("Slice Plane");

    // plot electric potential
    bool potentialEnabled = true;
    if (auto existingQuantity = slicePlane->getQuantity("Electric Potential")) {
        potentialEnabled = existingQuantity->isEnabled();
    }
    auto potentialQuantity = slicePlane->addFaceScalarQuantity("Electric Potential", electricPotential);
    const std::string electricPotentialColormap = getOptional<std::string>(
        outputConfig, "electricPotentialColormap", "plasma");
    const float electricPotentialColormapMinVal = getOptional<float>(
        outputConfig, "electricPotentialColormapMinVal", 0.0f);
    const float electricPotentialColormapMaxVal = getOptional<float>(
        outputConfig, "electricPotentialColormapMaxVal", 100.0f);
    potentialQuantity->setColorMap(electricPotentialColormap);
    potentialQuantity->setMapRange({electricPotentialColormapMinVal,
                                    electricPotentialColormapMaxVal});
    potentialQuantity->setEnabled(potentialEnabled);

    // plot electric field magnitude
    bool fieldMagnitudeEnabled = false;
    if (auto existingQuantity = slicePlane->getQuantity("Electric Field Magnitude")) {
        fieldMagnitudeEnabled = existingQuantity->isEnabled();
    }
    auto fieldMagnitudeQuantity = slicePlane->addFaceScalarQuantity("Electric Field Magnitude",
                                                                    electricFieldMagnitude);
    const std::string electricFieldMagnitudeColormap = getOptional<std::string>(
        outputConfig, "electricFieldMagnitudeColormap", "turbo");
    const float electricFieldMagnitudeColormapMinVal = getOptional<float>(
        outputConfig, "electricFieldMagnitudeColormapMinVal", 0.0f);
    const float electricFieldMagnitudeColormapMaxVal = getOptional<float>(
        outputConfig, "electricFieldMagnitudeColormapMaxVal", 15000.0f);
    fieldMagnitudeQuantity->setColorMap(electricFieldMagnitudeColormap);
    fieldMagnitudeQuantity->setMapRange({electricFieldMagnitudeColormapMinVal,
                                         electricFieldMagnitudeColormapMaxVal});
    fieldMagnitudeQuantity->setEnabled(fieldMagnitudeEnabled);
}

void guiCallback(const json& outputConfig,
                 MeshData& meshData,
                 MovableCombState& combState,
                 std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle,
                 std::shared_ptr<wosx::GPUGeometricQueries<3>> geometricQueries,
                 std::shared_ptr<GPUElectrostaticsPDE> pde,
                 std::shared_ptr<wosx::GPUWalkOnStarsSolver<float, 3>> walkOnStars,
                 std::pair<uint32_t, uint32_t> workloadParameters)
{
    if (combState.isMoving) {
        if (ImGui::Button("Stop Movable Comb")) {
            combState.isMoving = false;
        }

    } else {
        if (ImGui::SliderFloat("Movable Comb Voltage", &combState.voltage, 0.0f, 100.0f)) {
            // remove the existing quantities from the slice plane
            auto slicePlane = polyscope::getSurfaceMesh("Slice Plane");
            slicePlane->removeQuantity("Electric Potential");
            slicePlane->removeQuantity("Electric Field Magnitude");

            // update the PDE and plot the voltage on the comb
            pde->updateMovableCombVoltage(combState.voltage);
            plotCombVoltage(outputConfig, meshData, combState.voltage);
        }

        if (ImGui::Button("Solve with Movable Comb")) {
            combState.isMoving = true;
        }
    }

    if (combState.isMoving) {
        // update the movable comb position
        float displacement = meshData.computeMovableCombDisplacement(combState);
        updateMovableCombPosition(meshData, displacement, taskHandle,
                                  geometricQueries, pde);

        // run the solver and get the sample statistics
        std::vector<wosx::GPUSampleStatistics<float, 3>> sampleStatistics;
        runSolver(walkOnStars, workloadParameters, sampleStatistics);

        // extract electric potential and field magnitude estimates
        std::vector<float> electricPotential;
        std::vector<float> electricFieldMagnitude;
        getSolutionAndGradientNorm(sampleStatistics, electricPotential, electricFieldMagnitude);

        // update the mesh positions in polyscope
        polyscope::getSurfaceMesh("Mesh")->updateVertexPositions(meshData.currentPositions);

        // plot the estimated results on the slice plane
        plotSolutionAndGradient(outputConfig, electricPotential, electricFieldMagnitude);

        // increment the frame index
        combState.incrementFrameIndex();
    }
}

void visualizeProblem(const json& outputConfig,
                      const SlicePlaneData& slicePlaneData,
                      MeshData& meshData,
                      MovableCombState& combState,
                      std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle,
                      std::shared_ptr<wosx::GPUGeometricQueries<3>> geometricQueries,
                      std::shared_ptr<GPUElectrostaticsPDE> pde,
                      std::shared_ptr<wosx::GPUWalkOnStarsSolver<float, 3>> walkOnStars,
                      std::pair<uint32_t, uint32_t> workloadParameters)
{
    // set a few options
    polyscope::options::programName = "electrostatics demo";
    polyscope::options::verbosity = 0;
    polyscope::options::usePrefsFile = false;
    polyscope::options::autocenterStructures = false;
    polyscope::options::groundPlaneEnabled = false;

    // initialize polyscope
    polyscope::init();

    // set camera view
    setCameraView(meshData);

    // register the mesh
    polyscope::registerSurfaceMesh("Mesh", meshData.currentPositions, meshData.indices);
    plotCombVoltage(outputConfig, meshData, combState.voltage);

    // register the slice plane
    polyscope::registerSurfaceMesh("Slice Plane", slicePlaneData.positions, slicePlaneData.indices);

    // bind the gui callback
    polyscope::state::userCallback = std::bind(&guiCallback, std::cref(outputConfig),
                                               std::ref(meshData), std::ref(combState),
                                               taskHandle, geometricQueries, pde,
                                               walkOnStars, workloadParameters);

    // give control to polyscope gui
    polyscope::show();
    polyscope::state::userCallback = nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Demo entry point - load config, setup the problem, run the solver, save the solution,
// or optionally visualize the problem

void runDemo(const json& config)
{
    // load config settings
    std::filesystem::path wosxDirectoryPath = std::filesystem::current_path().parent_path();
    std::filesystem::path demoDirectoryPath = wosxDirectoryPath / "demo_apps" / "electrostatics";
    std::string wosxDirectoryPathStr = wosxDirectoryPath.string();
    std::string demoDirectoryPathStr = demoDirectoryPath.string();
    const json problemConfig = getRequired<json>(config, "problem");
    const json solverConfig = getRequired<json>(config, "solver");
    const json outputConfig = getRequired<json>(config, "output");

    // load mesh data
    MeshData meshData;
    loadMeshData(demoDirectoryPathStr, problemConfig, meshData);

    // setup a geometric queries object for the mesh
    std::shared_ptr<wosx::GPUGeometricQueries<3>> geometricQueries =
        setupGeometricQueries(meshData);

    // setup the demo PDE
    MovableCombState combState;
    combState.voltage = getOptional<float>(problemConfig, "movableCombVoltage", 75.0f);
    combState.nFrames = std::max(1, getOptional<int>(problemConfig, "nFrames", 120));
    std::shared_ptr<GPUElectrostaticsPDE> pde = std::make_shared<GPUElectrostaticsPDE>(
        geometricQueries->getAbsorbingBoundaryHandler(), meshData.isMovableFace, combState.voltage);

    // setup a task handle to run the demo on the GPU
    const std::string deviceBackend = getOptional<std::string>(config, "deviceBackend", "cuda");
    std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle =
        std::make_shared<wosx::GPUTaskHandle<float, 3>>(wosxDirectoryPathStr,
                                                        demoDirectoryPathStr,
                                                        deviceBackend);
    taskHandle->setGeometricQueries(geometricQueries);
    taskHandle->setPDE(pde);
    taskHandle->init();

    // create sample points to run the solver on; here we use a slice plane through the mesh
    const int sliceResolutionPow2 = getRequired<int>(problemConfig, "sliceResolutionPow2");
    SlicePlaneData slicePlaneData;
    slicePlaneData.build(meshData.boundingBox, sliceResolutionPow2);
    std::vector<wosx::GPUSamplePoint<3>> samplePts;
    createSamplePoints(slicePlaneData, samplePts);

    // setup the walk on stars solver
    std::pair<uint32_t, uint32_t> workloadParameters;
    std::shared_ptr<wosx::GPUWalkOnStarsSolver<float, 3>> walkOnStars =
        setupSolver(solverConfig, taskHandle, samplePts, workloadParameters);

    const bool shouldVisualizeProblem = getOptional<bool>(outputConfig, "visualizeProblem", false);
    if (shouldVisualizeProblem) {
        // visualize the problem
        visualizeProblem(outputConfig, slicePlaneData, meshData, combState, taskHandle,
                         geometricQueries, pde, walkOnStars, workloadParameters);

    } else {
        // run the solver and save the results as images
        for (int i = 0; i < combState.nFrames; i++) {
            std::cout << "Solving frame " << combState.frameIndex << std::endl;

            // update the movable comb position
            float displacement = meshData.computeMovableCombDisplacement(combState);
            updateMovableCombPosition(meshData, displacement, taskHandle,
                                      geometricQueries, pde);

            // run the solver and get the sample statistics
            std::vector<wosx::GPUSampleStatistics<float, 3>> sampleStatistics;
            runSolver(walkOnStars, workloadParameters, sampleStatistics);

            // extract electric potential and field magnitude estimates
            std::vector<float> electricPotential;
            std::vector<float> electricFieldMagnitude;
            getSolutionAndGradientNorm(sampleStatistics, electricPotential, electricFieldMagnitude);

            // save the estimated results as frame-indexed colormapped images
            saveSolutionAndGradientNormImages(demoDirectoryPathStr, outputConfig,
                                              combState.frameIndex, sliceResolutionPow2,
                                              electricPotential, electricFieldMagnitude);

            // increment the frame index
            combState.incrementFrameIndex();
        }
    }
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
    runDemo(config);

    return 0;
}
