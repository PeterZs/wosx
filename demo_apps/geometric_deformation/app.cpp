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

// This demo computes stochastic harmonic coordinates for a triangle mesh
// embedded inside a closed triangular cage, following the stochastic barycentric
// coordinates method [De Goes and Desbrun 2024]. GPU Walk on Spheres generates
// terminal samples on the cage boundary without tetrahedralizing its interior.
// RKPM restores linear precision, and optional cotangent-Laplacian denoising
// reduces Monte Carlo noise while preserving precision. The resulting sparse
// coordinates can repeatedly transfer cage displacements to the embedded geometry;
// Polyscope visualizes the setup and provides identity and rigid-rotation tests
// for linear precision, as well as non-affine cage deformation via ARAP.

#include "harmonic_samples_generation.h"
#include "harmonic_coordinates_solver.h"
#include "arap.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/pick.h"
#include "polyscope/transformation_gizmo.h"
#include "config.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <limits>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Problem specification - load mesh data, setup the GPU geometric queries and PDE objects

void loadMeshData(const std::string& directoryPath,
                  const json& problemConfig,
                  const std::string& meshFilename,
                  MeshData& meshData)
{
    auto getFilename = [](const std::string& directoryPath,
                          const json& config,
                          const std::string& filename) -> std::string {
        return directoryPath + "/" + getRequired<std::string>(config, filename);
    };

    // load the mesh
    std::string geometryFilename = getFilename(directoryPath, problemConfig, meshFilename);
    wosx::loadBoundaryMesh<3>(geometryFilename, meshData.positions, meshData.indices);
}

void normalizeMeshData(MeshData& embeddedMeshData,
                       MeshData& cageMeshData)
{
    // normalize the cage mesh
    int V = (int)cageMeshData.positions.size();
    wosx::Vector3 cm = wosx::Vector3::Zero();
    for (int i = 0; i < V; i++) {
        cm += cageMeshData.positions[i];
    }

    cm /= V;
    float radius = 0.0f;
    for (int i = 0; i < V; i++) {
        cageMeshData.positions[i] -= cm;
        radius = std::max(radius, cageMeshData.positions[i].norm());
    }

    for (int i = 0; i < V; i++) {
        cageMeshData.positions[i] /= radius;
    }

    // normalize the embedded mesh using the cm and scale of the cage mesh
    V = (int)embeddedMeshData.positions.size();
    for (int i = 0; i < V; i++) {
        embeddedMeshData.positions[i] -= cm;
        embeddedMeshData.positions[i] /= radius;
    }
}

std::shared_ptr<wosx::GPUGeometricQueries<3>> setupGeometricQueries(const MeshData& meshData)
{
    // setup an absorbing boundary handler for the mesh
    std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler =
        std::make_shared<wosx::GPUFcpwDirichletBoundaryHandler<3>>(
            meshData.positions, meshData.indices);

    // setup an empty reflecting boundary handler
    std::shared_ptr<wosx::GPUReflectingBoundaryHandler> reflectingBoundaryHandler =
        std::make_shared<wosx::GPUEmptyReflectingBoundaryHandler>();

    // create a geometric queries object from the handlers
    std::pair<wosx::Vector3, wosx::Vector3> boundingBox =
        wosx::computeBoundingBox<3>(meshData.positions, true, 1.5f);
    return std::make_shared<wosx::GPUGeometricQueries<3>>(
        absorbingBoundaryHandler, reflectingBoundaryHandler,
        boundingBox.first, boundingBox.second, true);
}

class GeometricDeformationPDE: public wosx::GPUPDE {
public:
    // constructor
    GeometricDeformationPDE() { /* nothing to do */ }

    // allocates and sets GPU resources, and returns type info
    void allocate(wosx::GPUContext& context) { /* nothing to do */ }
    void setResources(const wosx::ShaderCursor& cursor, bool printLogs) const { /* nothing to do */ }
    std::string getReflectionType() const { return "GeometricDeformationPDE"; }
    wosx::GPUPDEType getType() const { return wosx::GPUPDEType::Poisson; }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Solver setup - create query points, setup the harmonic samples generator, the harmonic coordinates solver
// for deforming the embedded mesh geometry, and the ARAP solver for deforming the cage mesh geometry

void createQueryPoints(const MeshData& embeddedMeshData,
                       std::vector<wosx::GPUSamplePoint<3>>& queryPoints)
{
    queryPoints.clear();
    for (int i = 0; i < (int)embeddedMeshData.positions.size(); i++) {
        wosx::GPUSamplePoint<3> queryPt;
        queryPt.pt = wosx::float3{embeddedMeshData.positions[i][0],
                                  embeddedMeshData.positions[i][1],
                                  embeddedMeshData.positions[i][2]};
        queryPt.normal = wosx::float3{0.0f, 0.0f, 0.0f};
        queryPt.type = wosx::SampleType::InDomain;
        queryPt.estimationQuantity = wosx::EstimationQuantity::Solution;
        queryPoints.emplace_back(queryPt);
    }
}

std::shared_ptr<GPUHarmonicSamplesGenerator> setupHarmonicSamplesGenerator(const json& solverConfig,
                                                                           std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle,
                                                                           const std::vector<wosx::GPUSamplePoint<3>>& queryPoints)
{
    // load config settings for walk on spheres
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const int nWalksPerBatch = getOptional<int>(solverConfig, "nWalksPerBatch", 100);
    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 10000);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);

    // setup walk settings (all PDE contributions can be ignored since we only care about sample generation)
    std::shared_ptr<wosx::GPUWalkSettings> walkSettings = std::make_shared<wosx::GPUWalkSettings>();
    walkSettings->epsilonShellForAbsorbingBoundary = epsilonShellForAbsorbingBoundary;
    walkSettings->epsilonShellForReflectingBoundary = 0.0f;
    walkSettings->silhouettePrecision = 0.0f;
    walkSettings->russianRouletteThreshold = 0.0f;
    walkSettings->maxWalkLength = maxWalkLength;
    walkSettings->stepsBeforeUsingMaximalSpheres = stepsBeforeUsingMaximalSpheres;
    walkSettings->solveDoubleSided = 0;
    walkSettings->useGradientControlVariates = 1;
    walkSettings->useGradientAntitheticVariates = 1;
    walkSettings->ignoreAbsorbingBoundaryContribution = 1;
    walkSettings->ignoreReflectingBoundaryContribution = 1;
    walkSettings->ignoreSourceContribution = 1;

    // initialize the harmonic samples generator
    std::shared_ptr<GPUHarmonicSamplesGenerator> harmonicSamplesGenerator =
        std::make_shared<GPUHarmonicSamplesGenerator>(taskHandle, walkSettings, printLogs);

    // populate the query points on the GPU
    harmonicSamplesGenerator->populateQueryPoints(queryPoints, nWalksPerBatch);

    return harmonicSamplesGenerator;
}

std::shared_ptr<HarmonicCoordinatesSolver> setupHarmonicCoordinatesSolver(const json& solverConfig,
                                                                          const MeshData& embeddedMeshData,
                                                                          const MeshData& cageMeshData)
{
    // load config settings for harmonic coordinates solver
    bool enableRKPM = getOptional<bool>(solverConfig, "enableRKPM", true);
    bool enableDenoising = getOptional<bool>(solverConfig, "enableDenoising", true);
    int nWalksPerBatch = getOptional<int>(solverConfig, "nWalksPerBatch", 100);

    // initialize the solver
    return std::make_shared<HarmonicCoordinatesSolver>(embeddedMeshData, cageMeshData,
                                                       enableRKPM, enableDenoising,
                                                       nWalksPerBatch);
}

std::vector<int> selectInitialHandleVertices(const Eigen::MatrixXf& cagePositions,
                                             const std::vector<int>& fixedVertexIndices,
                                             const std::vector<int>& nonFixedVertexIndices,
                                             int handleCount)
{
    // initialize distances to the fixed vertex set
    std::vector<float> minSquaredDistances(nonFixedVertexIndices.size(),
                                           std::numeric_limits<float>::max());
    for (int i = 0; i < (int)nonFixedVertexIndices.size(); i++) {
        for (int fixedVertexIndex: fixedVertexIndices) {
            minSquaredDistances[i] = std::min(
                minSquaredDistances[i],
                (cagePositions.row(nonFixedVertexIndices[i]) -
                 cagePositions.row(fixedVertexIndex)).squaredNorm());
        }
    }

    // select the highest non-fixed vertex first, then use greedy farthest-point selection
    std::vector<int> handleVertexIndices;
    for (int h = 0; h < handleCount; h++) {
        int farthestIndex = 0;
        if (h == 0) {
            for (int i = 1; i < (int)nonFixedVertexIndices.size(); i++) {
                if (cagePositions(nonFixedVertexIndices[i], 1) >
                    cagePositions(nonFixedVertexIndices[farthestIndex], 1)) {
                    farthestIndex = i;
                }
            }

        } else {
            for (int i = 1; i < (int)nonFixedVertexIndices.size(); i++) {
                if (minSquaredDistances[i] > minSquaredDistances[farthestIndex]) {
                    farthestIndex = i;
                }
            }
        }

        int handleVertexIndex = nonFixedVertexIndices[farthestIndex];
        handleVertexIndices.emplace_back(handleVertexIndex);
        for (int i = 0; i < (int)nonFixedVertexIndices.size(); i++) {
            float squaredDistance = (cagePositions.row(nonFixedVertexIndices[i]) -
                                     cagePositions.row(handleVertexIndex)).squaredNorm();
            minSquaredDistances[i] = std::min(minSquaredDistances[i], squaredDistance);
        }

        minSquaredDistances[farthestIndex] = -1.0f;
    }

    return handleVertexIndices;
}

std::shared_ptr<ARAPSolver> setupARAPSolver(const json& problemConfig,
                                            const MeshData& cageMeshData)
{
    // select the fixed cage vertices for the ARAP solver
    int handleCount = getOptional<int>(problemConfig, "handleCount", 1);
    float fixedVertexMaxY = getOptional<float>(problemConfig, "fixedVertexMaxY", -0.75f);
    std::vector<int> fixedVertexIndices;
    std::vector<int> nonFixedVertexIndices;
    Eigen::MatrixXf cagePositions = cageMeshData.convertPositionsToEigen();
    for (int i = 0; i < (int)cagePositions.rows(); i++) {
        if (cagePositions(i, 1) <= fixedVertexMaxY) {
            fixedVertexIndices.emplace_back(i);

        } else {
            nonFixedVertexIndices.emplace_back(i);
        }
    }

    // select spatially distributed initial handle vertices
    std::vector<int> handleVertexIndices = selectInitialHandleVertices(
        cagePositions, fixedVertexIndices, nonFixedVertexIndices, handleCount);

    // initialize the solver
    return std::make_shared<ARAPSolver>(cagePositions, cageMeshData.indices,
                                        fixedVertexIndices, handleVertexIndices);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Problem Visualization

struct RotationAnimationState {
    bool isPlaying = false;
    int frameIndex = 0;
    static constexpr int nFrames = 120;
};

struct ARAPDeformationState {
    Eigen::MatrixXf cagePositions;
    int activeHandleIndex;
    polyscope::TransformationGizmo* handleGizmo;
    polyscope::CurveNetworkNodeColorQuantity* cageVertexColorQuantity;
};

void updateARAPGizmo(ARAPDeformationState& arapState)
{
    int handleIndex = arapState.activeHandleIndex;
    Eigen::Vector3f handlePosition = arapState.cagePositions.row(handleIndex).transpose();
    arapState.handleGizmo->setPosition({handlePosition[0], handlePosition[1], handlePosition[2]});
}

void updateARAPVertexColors(const ARAPSolver& arapSolver,
                            ARAPDeformationState& arapState)
{
    std::vector<glm::vec3> cageVertexColors(arapState.cagePositions.rows(),
                                            glm::vec3{0.0f, 0.0f, 0.0f});
    for (int i = 0; i < (int)cageVertexColors.size(); i++) {
        if (arapSolver.isFixedVertex(i)) {
            cageVertexColors[i] = glm::vec3{1.0f, 0.0f, 0.0f};

        } else if (arapSolver.isHandleVertex(i)) {
            cageVertexColors[i] = glm::vec3{0.0f, 1.0f, 0.0f};
        }
    }

    arapState.cageVertexColorQuantity->updateData(cageVertexColors);
}

void resetARAPDeformationState(const MeshData& embeddedMeshData,
                               const MeshData& cageMeshData,
                               std::shared_ptr<ARAPSolver> arapSolver,
                               ARAPDeformationState& arapState)
{
    // reset the solver, cage positions, and gizmo
    arapSolver->reset();
    arapState.cagePositions = cageMeshData.convertPositionsToEigen();
    updateARAPGizmo(arapState);

    // update the cage wireframe and embedded mesh
    polyscope::getCurveNetwork("Cage Wireframe")->updateNodePositions(cageMeshData.positions);
    polyscope::getSurfaceMesh("Embedded Mesh")->updateVertexPositions(embeddedMeshData.positions);
}

void guiCallback(const MeshData& embeddedMeshData,
                 const MeshData& cageMeshData,
                 std::shared_ptr<GPUHarmonicSamplesGenerator> harmonicSamplesGenerator,
                 std::shared_ptr<HarmonicCoordinatesSolver> harmonicCoordinatesSolver,
                 std::shared_ptr<ARAPSolver> arapSolver,
                 RotationAnimationState& rotationAnimationState,
                 ARAPDeformationState& arapState)
{
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ImGui::TextUnformatted("Harmonic Coordinates (Precomputation)");
    ImGui::Indent();

    if (rotationAnimationState.isPlaying) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button("Generate Samples")) {
        std::vector<wosx::Vector3> hitPoints;
        std::vector<uint32_t> hitTriangleIds;

        // generate and accumulate a new batch of harmonic samples
        harmonicSamplesGenerator->generateSamples(hitPoints, hitTriangleIds);
        harmonicCoordinatesSolver->accumulateSamples(hitPoints, hitTriangleIds);

        // reset ARAP deformation state and remove identity reconstruction
        resetARAPDeformationState(embeddedMeshData, cageMeshData, arapSolver, arapState);
        polyscope::removeStructure("Identity Reconstruction");
    }

    if (ImGui::Button("Finalize Coordinates")) {
        // finalize harmonic coordinates and analyze their properties
        harmonicCoordinatesSolver->finalizeCoordinates();
        harmonicCoordinatesSolver->analyzeCoordinates();

        // reset ARAP deformation state
        resetARAPDeformationState(embeddedMeshData, cageMeshData, arapSolver, arapState);
    }

    if (ImGui::Button("Reset Coordinates")) {
        // reset the harmonic coordinates
        harmonicCoordinatesSolver->reset();

        // reset ARAP deformation state and remove identity reconstruction
        resetARAPDeformationState(embeddedMeshData, cageMeshData, arapSolver, arapState);
        polyscope::removeStructure("Identity Reconstruction");
    }

    ImGui::Text("Accumulated samples per query point: %d",
                harmonicCoordinatesSolver->getSamplesAccumulatedPerQueryPoint());

    bool enableRKPM = harmonicCoordinatesSolver->isRKPMEnabled();
    if (ImGui::Checkbox("Enable RKPM", &enableRKPM)) {
        // toggle RKPM correction and reset the harmonic coordinates
        harmonicCoordinatesSolver->toggleRKPM();

        // reset ARAP deformation state and remove identity reconstruction
        resetARAPDeformationState(embeddedMeshData, cageMeshData, arapSolver, arapState);
        polyscope::removeStructure("Identity Reconstruction");
    }

    bool enableDenoising = harmonicCoordinatesSolver->isDenoisingEnabled();
    if (!harmonicCoordinatesSolver->isRKPMEnabled()) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Checkbox("Enable Denoising", &enableDenoising)) {
        // toggle denoising and reset the harmonic coordinates
        harmonicCoordinatesSolver->toggleDenoising();

        // reset ARAP deformation state and remove identity reconstruction
        resetARAPDeformationState(embeddedMeshData, cageMeshData, arapSolver, arapState);
        polyscope::removeStructure("Identity Reconstruction");
    }
    if (!harmonicCoordinatesSolver->isRKPMEnabled()) {
        ImGui::EndDisabled();
    }

    if (rotationAnimationState.isPlaying) {
        ImGui::EndDisabled();
    }

    ImGui::Unindent();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    bool disableGeometricDeformation = !harmonicCoordinatesSolver->hasFinalizedCoordinates() ||
                                       rotationAnimationState.isPlaying;
    if (disableGeometricDeformation) {
        ImGui::BeginDisabled();
    }

    ImGui::NewLine();
    ImGui::TextUnformatted("Linear Precision");
    ImGui::Indent();

    if (ImGui::Button("Check Identity Reconstruction")) {
        // reset ARAP deformation state
        resetARAPDeformationState(embeddedMeshData, cageMeshData, arapSolver, arapState);

        // test linear precision by reconstructing the embedded mesh geometry
        // from the rest cage positions
        Eigen::MatrixXf restCagePositions = cageMeshData.convertPositionsToEigen();
        Eigen::MatrixXf reconstructedPositions =
            harmonicCoordinatesSolver->testLinearPrecision(restCagePositions);
        polyscope::registerSurfaceMesh("Identity Reconstruction",
                                       reconstructedPositions,
                                       embeddedMeshData.indices);
    }

    if (ImGui::Button("Check Rotated Reconstruction")) {
        // reset ARAP deformation state and start the rotation animation
        resetARAPDeformationState(embeddedMeshData, cageMeshData, arapSolver, arapState);
        rotationAnimationState.isPlaying = true;
        rotationAnimationState.frameIndex = 0;
    }

    ImGui::Unindent();

    ImGui::NewLine();
    ImGui::TextUnformatted("Geometric Deformation");
    ImGui::Indent();
    ImGui::TextWrapped("Red cage vertices are fixed and cannot be deformed.");
    ImGui::TextWrapped("Click a black cage vertex to add a deformation handle.");
    ImGui::TextWrapped("Click a green cage vertex to make it the active handle.");
    ImGui::TextWrapped("Drag the gizmo to move the active handle.");

    std::vector<int> handleVertexIndices = arapSolver->getHandleVertexSet();
    bool disableHandleRemoval = handleVertexIndices.size() == 1;
    if (disableHandleRemoval) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Remove Active Handle")) {
        // remove the active handle from the handle vertex set
        std::vector<int> updatedHandleVertexIndices;
        for (int index: handleVertexIndices) {
            if (index != arapState.activeHandleIndex) {
                updatedHandleVertexIndices.emplace_back(index);
            }
        }

        // update the handle vertex set and deform the cage and embedded geometry
        arapSolver->updateHandleVertexSet(updatedHandleVertexIndices);
        arapState.cagePositions = arapSolver->deform();
        Eigen::MatrixXf deformedEmbeddedPositions =
            harmonicCoordinatesSolver->deformEmbeddedGeometry(arapState.cagePositions);

        // update the active handle index, gizmo, and cage vertex colors
        arapState.activeHandleIndex = updatedHandleVertexIndices[0];
        updateARAPGizmo(arapState);
        updateARAPVertexColors(*arapSolver, arapState);

        // update the cage wireframe and embedded mesh
        polyscope::getCurveNetwork("Cage Wireframe")->updateNodePositions(arapState.cagePositions);
        polyscope::getSurfaceMesh("Embedded Mesh")->updateVertexPositions(deformedEmbeddedPositions);
        polyscope::removeStructure("Identity Reconstruction");
    }
    if (disableHandleRemoval) {
        ImGui::EndDisabled();
    }

    if (ImGui::Button("Reset Cage Deformation")) {
        resetARAPDeformationState(embeddedMeshData, cageMeshData, arapSolver, arapState);
    }

    if (disableGeometricDeformation) {
        ImGui::EndDisabled();
    }

    if (rotationAnimationState.frameIndex > RotationAnimationState::nFrames) {
        // stop the rotation animation and reset ARAP deformation state
        rotationAnimationState.isPlaying = false;
        rotationAnimationState.frameIndex = 0;
        resetARAPDeformationState(embeddedMeshData, cageMeshData, arapSolver, arapState);

    } else if (rotationAnimationState.isPlaying) {
        // test linear precision by reconstructing the embedded mesh geometry
        // from the rotated cage positions
        constexpr float twoPi = 6.28318530717958647692f;
        float angle = twoPi*(float)rotationAnimationState.frameIndex/
                            (float)RotationAnimationState::nFrames;
        Eigen::Matrix3f rotation = Eigen::AngleAxisf(
            angle, Eigen::Vector3f::UnitY()).toRotationMatrix();
        Eigen::MatrixXf rotatedCagePositions =
            cageMeshData.convertPositionsToEigen()*rotation.transpose();
        Eigen::MatrixXf reconstructedPositions =
            harmonicCoordinatesSolver->testLinearPrecision(rotatedCagePositions);

        polyscope::getCurveNetwork("Cage Wireframe")->updateNodePositions(rotatedCagePositions);
        polyscope::getSurfaceMesh("Embedded Mesh")->updateVertexPositions(reconstructedPositions);
        rotationAnimationState.frameIndex++;
    }

    // enable ARAP after harmonic coordinates have been finalized
    arapState.handleGizmo->setEnabled(!disableGeometricDeformation);

    // add a new handle or activate an existing one when a cage vertex is selected
    if (!disableGeometricDeformation && polyscope::haveSelection()) {
        polyscope::PickResult selection = polyscope::getSelection();
        auto cageWireframe = polyscope::getCurveNetwork("Cage Wireframe");
        if (selection.structure == cageWireframe) {
            polyscope::CurveNetworkPickResult cageSelection =
                cageWireframe->interpretPickResult(selection);
            if (cageSelection.elementType == polyscope::CurveNetworkElement::NODE) {
                int vertexIndex = (int)cageSelection.index;
                if (!arapSolver->isFixedVertex(vertexIndex)) {
                    std::vector<int> handleVertexIndices = arapSolver->getHandleVertexSet();
                    if (!arapSolver->isHandleVertex(vertexIndex)) {
                        handleVertexIndices.emplace_back(vertexIndex);
                        arapSolver->updateHandleVertexSet(handleVertexIndices);
                        updateARAPVertexColors(*arapSolver, arapState);
                    }

                    arapState.activeHandleIndex = vertexIndex;
                    updateARAPGizmo(arapState);
                }

                polyscope::resetSelection();
            }
        }
    }

    // apply the gizmo translation to the selected handle
    if (!disableGeometricDeformation) {
        int handleVertex = arapState.activeHandleIndex;
        glm::vec3 gizmoPosition = arapState.handleGizmo->getPosition();
        Eigen::Vector3f handlePosition(gizmoPosition.x, gizmoPosition.y, gizmoPosition.z);
        Eigen::Vector3f currentHandlePosition =
            arapState.cagePositions.row(handleVertex).transpose();
        if ((handlePosition - currentHandlePosition).squaredNorm() > 1e-12f) {
            // update the handle position and deform the cage and embedded geometry
            arapSolver->updateHandlePosition(handleVertex, handlePosition);
            arapState.cagePositions = arapSolver->deform();
            Eigen::MatrixXf deformedEmbeddedPositions =
                harmonicCoordinatesSolver->deformEmbeddedGeometry(arapState.cagePositions);

            // update the cage wireframe and embedded mesh
            polyscope::getCurveNetwork("Cage Wireframe")->updateNodePositions(arapState.cagePositions);
            polyscope::getSurfaceMesh("Embedded Mesh")->updateVertexPositions(deformedEmbeddedPositions);
            polyscope::removeStructure("Identity Reconstruction");
        }
    }

    ImGui::Unindent();
}

void visualizeProblem(const MeshData& embeddedMeshData,
                      const MeshData& cageMeshData,
                      std::shared_ptr<GPUHarmonicSamplesGenerator> harmonicSamplesGenerator,
                      std::shared_ptr<HarmonicCoordinatesSolver> harmonicCoordinatesSolver,
                      std::shared_ptr<ARAPSolver> arapSolver)
{
    // set a few options
    polyscope::options::programName = "geometric deformation demo";
    polyscope::options::verbosity = 0;
    polyscope::options::usePrefsFile = false;
    polyscope::options::autocenterStructures = false;
    polyscope::options::groundPlaneEnabled = false;

    // initialize polyscope
    polyscope::init();

    // register the embedded mesh
    polyscope::registerSurfaceMesh("Embedded Mesh", embeddedMeshData.positions, embeddedMeshData.indices);

    // register the cage mesh as a wireframe
    auto cageWireframe = polyscope::registerCurveNetwork("Cage Wireframe",
                                                         cageMeshData.positions,
                                                         cageMeshData.getEdges());
    cageWireframe->setColor({0.0f, 0.0f, 0.0f});
    cageWireframe->setRadius(0.00225f, false);

    // initialize the ARAP deformation state
    ARAPDeformationState arapState;
    arapState.cagePositions = cageMeshData.convertPositionsToEigen();
    arapState.activeHandleIndex = arapSolver->getHandleVertexSet()[0];

    // color fixed vertices red, free vertices black, and handles green
    std::vector<glm::vec3> cageVertexColors(cageMeshData.positions.size(),
                                            glm::vec3{0.0f, 0.0f, 0.0f});
    arapState.cageVertexColorQuantity = cageWireframe->addNodeColorQuantity(
        "Vertex Types: Fixed/Free/Handle", cageVertexColors);
    arapState.cageVertexColorQuantity->setEnabled(true);
    updateARAPVertexColors(*arapSolver, arapState);

    // create a translation gizmo for the ARAP vertex handle
    arapState.handleGizmo = polyscope::addTransformationGizmo("ARAP Handle");
    arapState.handleGizmo->setAllowTranslation(true);
    arapState.handleGizmo->setAllowRotation(false);
    arapState.handleGizmo->setAllowScaling(false);
    arapState.handleGizmo->setEnabled(false);
    updateARAPGizmo(arapState);

    // bind the gui callback
    RotationAnimationState rotationAnimationState;
    polyscope::state::userCallback = std::bind(&guiCallback,
                                               std::cref(embeddedMeshData),
                                               std::cref(cageMeshData),
                                               harmonicSamplesGenerator,
                                               harmonicCoordinatesSolver,
                                               arapSolver,
                                               std::ref(rotationAnimationState),
                                               std::ref(arapState));

    // give control to polyscope gui
    polyscope::show();
    polyscope::state::userCallback = nullptr;
    polyscope::removeTransformationGizmo(arapState.handleGizmo);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Demo entry point - load config and setup the problem

void runDemo(const json& config)
{
    // load config settings
    std::filesystem::path wosxDirectoryPath = std::filesystem::current_path().parent_path();
    std::filesystem::path demoDirectoryPath = wosxDirectoryPath / "demo_apps" / "geometric_deformation";
    std::string wosxDirectoryPathStr = wosxDirectoryPath.string();
    std::string demoDirectoryPathStr = demoDirectoryPath.string();
    const json problemConfig = getRequired<json>(config, "problem");
    const json solverConfig = getRequired<json>(config, "solver");

    // load embedded and cage mesh data, and normalize the mesh geometry
    MeshData embeddedMeshData, cageMeshData;
    loadMeshData(demoDirectoryPathStr, problemConfig, "embeddedMesh", embeddedMeshData);
    loadMeshData(demoDirectoryPathStr, problemConfig, "cageMesh", cageMeshData);
    normalizeMeshData(embeddedMeshData, cageMeshData);

    // setup a geometric queries object for the cage mesh
    std::shared_ptr<wosx::GPUGeometricQueries<3>> geometricQueries =
        setupGeometricQueries(cageMeshData);

    // setup the demo PDE (this is just an unused stub to instantiate the
    // walk-on-spheres-based harmonic samples generator)
    std::shared_ptr<GeometricDeformationPDE> pde = std::make_shared<GeometricDeformationPDE>();

    // setup a task handle to run the demo on the GPU
    const std::string deviceBackend = getOptional<std::string>(config, "deviceBackend", "cuda");
    std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle =
        std::make_shared<wosx::GPUTaskHandle<float, 3>>(wosxDirectoryPathStr,
                                                        demoDirectoryPathStr,
                                                        deviceBackend);
    taskHandle->setGeometricQueries(geometricQueries);
    taskHandle->setPDE(pde);
    taskHandle->init();

    // create query points from the embedded mesh
    std::vector<wosx::GPUSamplePoint<3>> queryPoints;
    createQueryPoints(embeddedMeshData, queryPoints);

    // setup the harmonic samples generator
    std::shared_ptr<GPUHarmonicSamplesGenerator> harmonicSamplesGenerator =
        setupHarmonicSamplesGenerator(solverConfig, taskHandle, queryPoints);

    // setup the harmonic coordinates solver
    std::shared_ptr<HarmonicCoordinatesSolver> harmonicCoordinatesSolver =
        setupHarmonicCoordinatesSolver(solverConfig, embeddedMeshData, cageMeshData);

    // setup the ARAP solver
    std::shared_ptr<ARAPSolver> arapSolver = setupARAPSolver(problemConfig, cageMeshData);

    // visualize the problem
    visualizeProblem(embeddedMeshData, cageMeshData, harmonicSamplesGenerator,
                     harmonicCoordinatesSolver, arapSolver);
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
