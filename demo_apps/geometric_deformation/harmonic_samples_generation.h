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

// This file implements a GPUHarmonicSamplesGenerator helper class for estimating
// harmonic coordinates by generating walk-on-spheres samples from query points
// embedded inside a watertight triangular cage. It does so by running a custom
// Slang shader defined in harmonic-samples-generation.cs.slang that records the
// terminal hit points and triangle ids on the cage for each walk-on-spheres sample.

#pragma once

#include <wosx/wosx_gpu.h>
#include <chrono>

using namespace wosx;

class GPUGenerateHarmonicSamples: public GPUShaderEntryPoint {
public:
    // constructor
    GPUGenerateHarmonicSamples(GPUBuffer& queryPoints_, GPUBuffer& rngs_);

    // allocate and set resources
    void allocate(GPUContext& context, uint32_t nQueryPoints_, uint32_t nWalksPerQueryPoint_);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

    // read results
    void read(GPUContext& context,
              std::vector<wosx::Vector3>& hitPointsData,
              std::vector<uint32_t>& hitTriangleIdsData) const;

private:
    // members
    GPUBuffer& queryPoints;
    GPUBuffer& rngs;
    GPUBuffer hitPoints = {};
    GPUBuffer hitTriangleIds = {};
    uint32_t nQueryPoints;
    uint32_t nWalksPerQueryPoint;
};

class GPUHarmonicSamplesGenerator {
public:
    // constructor
    GPUHarmonicSamplesGenerator(std::shared_ptr<GPUTaskHandle<float, 3>> handle_,
                                std::shared_ptr<GPUWalkSettings> walkSettings_,
                                bool printLogs_=false);

    // populates query points
    void populateQueryPoints(const std::vector<GPUSamplePoint<3>>& queryPoints_,
                             uint32_t nWalksPerQueryPoint_);

    // generates nQueryPoints * nWalksPerQueryPoint samples
    void generateSamples(std::vector<wosx::Vector3>& hitPoints,
                         std::vector<uint32_t>& hitTriangleIds);

    // enables/disables logging
    void enableLogging(bool enable);

private:
    // members
    std::shared_ptr<GPUTaskHandle<float, 3>> handle;
    GPUContext& context;
    std::shared_ptr<GPUGeometricQueries<3>> geometricQueries;
    GPUWalkOnSpheres<float, 3> walkOnSpheres;
    uint32_t nQueryPoints;
    uint32_t nWalksPerQueryPoint;
    GPUBuffer queryPoints;
    ComputeShader computeSampleBoundaryDistanceShader;
    ComputeShader seedRngsShader;
    ComputeShader generateHarmonicSamplesShader;
    GPUSeedRngs seedRngsEntryPoint;
    GPUGenerateHarmonicSamples generateHarmonicSamplesEntryPoint;
    std::function<void(const ComputeShader&, const ShaderCursor&)> bindComputeSampleBoundaryDistanceResources;
    std::function<void(const ComputeShader&, const ShaderCursor&)> bindGenerateHarmonicSamplesResources;
    uint32_t nThreadsPerGroup;
    bool printLogs;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

inline GPUGenerateHarmonicSamples::GPUGenerateHarmonicSamples(GPUBuffer& queryPoints_,
                                                              GPUBuffer& rngs_):
queryPoints(queryPoints_),
rngs(rngs_),
nQueryPoints(0),
nWalksPerQueryPoint(0)
{
    // do nothing
}

inline void GPUGenerateHarmonicSamples::allocate(GPUContext& context,
                                                 uint32_t nQueryPoints_,
                                                 uint32_t nWalksPerQueryPoint_)
{
    nQueryPoints = nQueryPoints_;
    nWalksPerQueryPoint = nWalksPerQueryPoint_;
    uint32_t nPoints = nQueryPoints*nWalksPerQueryPoint;
    hitPoints.allocate<wosx::Vector3>(context, true, std::vector<wosx::Vector3>(nPoints, wosx::Vector3::Zero()));
    hitTriangleIds.allocate<uint32_t>(context, true, std::vector<uint32_t>(nPoints, FCPW_GPU_UINT_MAX));
}

inline void GPUGenerateHarmonicSamples::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("queryPoints").setBinding(queryPoints.buffer);
    cursor.getPath("rngs").setBinding(rngs.buffer);
    cursor.getPath("hitPoints").setBinding(hitPoints.buffer);
    cursor.getPath("hitTriangleIds").setBinding(hitTriangleIds.buffer);
    cursor.getPath("nQueryPoints").setData(nQueryPoints);
    cursor.getPath("nWalksPerQueryPoint").setData(nWalksPerQueryPoint);
    if (printLogs) printReflectionInfo(cursor, 7, "generateHarmonicSamples");
}

inline void GPUGenerateHarmonicSamples::read(GPUContext& context,
                                             std::vector<wosx::Vector3>& hitPointsData,
                                             std::vector<uint32_t>& hitTriangleIdsData) const
{
    hitPoints.read<wosx::Vector3>(context, hitPointsData);
    hitTriangleIds.read<uint32_t>(context, hitTriangleIdsData);
}

inline GPUHarmonicSamplesGenerator::GPUHarmonicSamplesGenerator(std::shared_ptr<GPUTaskHandle<float, 3>> handle_,
                                                                std::shared_ptr<GPUWalkSettings> walkSettings_,
                                                                bool printLogs_):
handle(handle_),
context(handle->getContext()),
geometricQueries(handle->getGeometricQueries()),
walkOnSpheres(handle->getPDE(), handle->getGeometricQueries(), walkSettings_),
nQueryPoints(0),
nWalksPerQueryPoint(0),
generateHarmonicSamplesEntryPoint(queryPoints, seedRngsEntryPoint.getBuffer()),
nThreadsPerGroup(256),
printLogs(printLogs_)
{
    // initialize shader programs
    GPUModule mainModule;
    mainModule.load(context, handle->getWosxDirectoryPath() + "/demo_apps/geometric_deformation/harmonic-samples-generation.cs.slang");
    const std::vector<GPUModule>& libraryModules = handle->getLibraryModules();
    computeSampleBoundaryDistanceShader.loadProgram(context, mainModule, libraryModules, "computeSampleBoundaryDistance");
    seedRngsShader.loadProgram(context, mainModule, libraryModules, "seedRngs");
    generateHarmonicSamplesShader.loadProgram(context, mainModule, libraryModules, "generateHarmonicSamples");

    // bind shader resources
    bindComputeSampleBoundaryDistanceResources = [this](const ComputeShader& shader,
                                                        const ShaderCursor& cursor) {
        // bind geometric queries shader resources
        ComPtr<IShaderObject> geometricQueriesShaderObject = shader.createShaderObject(
            context, geometricQueries->getReflectionType());
        ShaderCursor geometricQueriesShaderCursor(geometricQueriesShaderObject);
        geometricQueries->setResources(geometricQueriesShaderCursor, printLogs);
        cursor.getPath("gGeometricQueries").setObject(geometricQueriesShaderObject);
    };
    bindGenerateHarmonicSamplesResources = [this](const ComputeShader& shader,
                                                  const ShaderCursor& cursor) {
        // bind walk on spheres shader resources
        ComPtr<IShaderObject> walkOnSpheresShaderObject = shader.createShaderObject(
            context, walkOnSpheres.getReflectionType());
        ShaderCursor walkOnSpheresShaderCursor(walkOnSpheresShaderObject);
        walkOnSpheres.setResources(walkOnSpheresShaderCursor, printLogs);
        cursor.getPath("gWalkOnSpheres").setObject(walkOnSpheresShaderObject);

        // bind acceleration structure shader resources
        std::shared_ptr<GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler =
            geometricQueries->getAbsorbingBoundaryHandler();
        ComPtr<IShaderObject> accelerationStructureShaderObject = shader.createShaderObject(
            context, absorbingBoundaryHandler->getReflectionType());
        ShaderCursor accelerationStructureCursor(accelerationStructureShaderObject);
        absorbingBoundaryHandler->setResources(accelerationStructureCursor, printLogs);
        cursor.getPath("gAccelerationStructure").setObject(accelerationStructureShaderObject);
    };
}

inline void GPUHarmonicSamplesGenerator::populateQueryPoints(const std::vector<GPUSamplePoint<3>>& queryPoints_,
                                                             uint32_t nWalksPerQueryPoint_)
{
    // allocate query points buffer
    nQueryPoints = (uint32_t)queryPoints_.size();
    nWalksPerQueryPoint = nWalksPerQueryPoint_;
    queryPoints.allocate<GPUSamplePoint<3>>(context, true, queryPoints_);

    // run compute boundary distance shader
    GPUComputeSampleBoundaryDistance computeSampleBoundaryDistanceEntryPoint(queryPoints, nQueryPoints);
    uint32_t nThreadGroups = countThreadGroups(nQueryPoints, nThreadsPerGroup, printLogs);
    runShader<GPUComputeSampleBoundaryDistance>(context, computeSampleBoundaryDistanceShader,
                                                computeSampleBoundaryDistanceEntryPoint,
                                                bindComputeSampleBoundaryDistanceResources,
                                                nThreadGroups, 1, printLogs);

    // run seed rngs shader
    uint32_t nRngs = nQueryPoints*nWalksPerQueryPoint;
    seedRngsEntryPoint.allocate(context, nRngs);
    nThreadGroups = countThreadGroups(nRngs, nThreadsPerGroup, printLogs);
    runShader<GPUSeedRngs>(context, seedRngsShader, seedRngsEntryPoint,
                           {}, nThreadGroups, 1, printLogs);

    // allocate generate harmonic samples entry point
    generateHarmonicSamplesEntryPoint.allocate(context, nQueryPoints, nWalksPerQueryPoint);
}

inline void GPUHarmonicSamplesGenerator::generateSamples(std::vector<wosx::Vector3>& hitPoints,
                                                         std::vector<uint32_t>& hitTriangleIds)
{
    auto start = std::chrono::high_resolution_clock::now();

    // run generate harmonic samples shader
    uint32_t nTotalWorkItems = nQueryPoints*nWalksPerQueryPoint;
    uint32_t nThreadGroups = countThreadGroups(nTotalWorkItems, nThreadsPerGroup, printLogs);
    runShader<GPUGenerateHarmonicSamples>(context, generateHarmonicSamplesShader,
                                          generateHarmonicSamplesEntryPoint,
                                          bindGenerateHarmonicSamplesResources,
                                          nThreadGroups, 1, printLogs);

    // read results from GPU
    generateHarmonicSamplesEntryPoint.read(context, hitPoints, hitTriangleIds);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "(GPU) Generated harmonic samples in " << elapsed.count() << " ms" << std::endl;
}

inline void GPUHarmonicSamplesGenerator::enableLogging(bool enable)
{
    printLogs = enable;
}
