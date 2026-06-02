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
class GPUDistanceQueries {
public:
    // constructor
    GPUDistanceQueries(std::shared_ptr<GPUTaskHandle<T, DIM>> handle_,
                       bool printLogs_=false);

    // compute distance to boundary
    void computeDistToBoundary(const std::vector<Vector<DIM>>& points,
                               std::vector<float>& distToAbsorbingBoundary,
                               std::vector<float>& distToReflectingBoundary);

    // compute intersection distance to boundary (returns -1.0 if no intersection)
    void intersectBoundary(const std::vector<Vector<DIM>>& origins,
                           const std::vector<Vector<DIM>>& directions,
                           std::vector<float>& distAlongRay,
                           std::vector<Vector<DIM>>& intersectionPoints,
                           std::vector<Vector<DIM>>& intersectionNormals);
    void intersectAbsorbingBoundary(const std::vector<Vector<DIM>>& origins,
                                    const std::vector<Vector<DIM>>& directions,
                                    std::vector<float>& distAlongRay,
                                    std::vector<Vector<DIM>>& intersectionPoints,
                                    std::vector<Vector<DIM>>& intersectionNormals);
    void intersectReflectingBoundary(const std::vector<Vector<DIM>>& origins,
                                     const std::vector<Vector<DIM>>& directions,
                                     std::vector<float>& distAlongRay,
                                     std::vector<Vector<DIM>>& intersectionPoints,
                                     std::vector<Vector<DIM>>& intersectionNormals);

private:
    // intersects a ray with the boundary
    void intersectBoundaryImpl(const ComputeShader& shader,
                               const std::vector<Vector<DIM>>& origins,
                               const std::vector<Vector<DIM>>& directions,
                               std::vector<float>& distAlongRay,
                               std::vector<Vector<DIM>>& intersectionPoints,
                               std::vector<Vector<DIM>>& intersectionNormals);

    // members
    std::shared_ptr<GPUTaskHandle<T, DIM>> handle;
    GPUContext& context;
    std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries;
    ComputeShader computeDistToBoundaryShader;
    ComputeShader intersectBoundaryShader;
    ComputeShader intersectAbsorbingBoundaryShader;
    ComputeShader intersectReflectingBoundaryShader;
    std::function<void(const ComputeShader&, const ShaderCursor&)> bindShaderResources;
    uint32_t nThreadsPerGroup;
    bool printLogs;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
GPUDistanceQueries<T, DIM>::GPUDistanceQueries(std::shared_ptr<GPUTaskHandle<T, DIM>> handle_,
                                               bool printLogs_):
handle(handle_),
context(handle->getContext()),
geometricQueries(handle->getGeometricQueries()),
nThreadsPerGroup(256),
printLogs(printLogs_)
{
    // initialize shader programs
    GPUModule mainModule;
    mainModule.load(context, handle->getWosxGpuDirectoryPath() + "/entrypoints/distance-queries.cs.slang");
    const std::vector<GPUModule>& libraryModules = handle->getLibraryModules();
    computeDistToBoundaryShader.loadProgram(context, mainModule, libraryModules, "computeDistToBoundary");
    intersectBoundaryShader.loadProgram(context, mainModule, libraryModules, "intersectBoundary");
    intersectAbsorbingBoundaryShader.loadProgram(context, mainModule, libraryModules, "intersectAbsorbingBoundary");
    intersectReflectingBoundaryShader.loadProgram(context, mainModule, libraryModules, "intersectReflectingBoundary");

    // bind shader resources
    bindShaderResources = [this](const ComputeShader& shader, const ShaderCursor& cursor) {
        ComPtr<IShaderObject> geometricQueriesShaderObject = shader.createShaderObject(
            context, geometricQueries->getReflectionType());
        ShaderCursor geometricQueriesShaderCursor(geometricQueriesShaderObject);
        geometricQueries->setResources(geometricQueriesShaderCursor, printLogs);
        cursor.getPath("gGeometricQueries").setObject(geometricQueriesShaderObject);
    };
}

template <size_t DIM>
class GPUComputeDistToBoundary: public GPUShaderEntryPoint {
public:
    // constructor
    GPUComputeDistToBoundary() {}

    // allocate resources
    void allocate(GPUContext& context, const std::vector<Vector<DIM>>& pointsData) {
        points.allocate<Vector<DIM>>(context, false, pointsData);
        nPoints = (uint32_t)pointsData.size();
        distToAbsorbingBoundary.allocate<float>(context, true, std::vector<float>(nPoints, 0.0f));
        distToReflectingBoundary.allocate<float>(context, true, std::vector<float>(nPoints, 0.0f));
    }

    // set resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const {
        cursor.getPath("points").setBinding(points.buffer);
        cursor.getPath("distToAbsorbingBoundary").setBinding(distToAbsorbingBoundary.buffer);
        cursor.getPath("distToReflectingBoundary").setBinding(distToReflectingBoundary.buffer);
        cursor.getPath("nPoints").setData(nPoints);
        if (printLogs) printReflectionInfo(cursor, 5, "computeDistToBoundary");
    }

    // read results
    void read(GPUContext& context,
              std::vector<float>& distToAbsorbingBoundaryData,
              std::vector<float>& distToReflectingBoundaryData) const {
        distToAbsorbingBoundary.read(context, distToAbsorbingBoundaryData);
        distToReflectingBoundary.read(context, distToReflectingBoundaryData);
    }

private:
    // members
    GPUBuffer points = {};
    GPUBuffer distToAbsorbingBoundary = {};
    GPUBuffer distToReflectingBoundary = {};
    uint32_t nPoints = 0;
};

template <typename T, size_t DIM>
void GPUDistanceQueries<T, DIM>::computeDistToBoundary(const std::vector<Vector<DIM>>& points,
                                                       std::vector<float>& distToAbsorbingBoundary,
                                                       std::vector<float>& distToReflectingBoundary)
{
    // allocate entry point data
    GPUComputeDistToBoundary<DIM> entryPoint;
    entryPoint.allocate(context, points);

    // run shader
    uint32_t nPoints = (uint32_t)points.size();
    uint32_t nThreadGroups = countThreadGroups(nPoints, nThreadsPerGroup, printLogs);
    runShader<GPUComputeDistToBoundary<DIM>>(context, computeDistToBoundaryShader,
                                             entryPoint, bindShaderResources,
                                             nThreadGroups, 1, printLogs);

    // read results from GPU
    entryPoint.read(context, distToAbsorbingBoundary, distToReflectingBoundary);
}

template <size_t DIM>
class GPUIntersectBoundary: public GPUShaderEntryPoint {
public:
    // constructor
    GPUIntersectBoundary() {}

    // allocate resources
    void allocate(GPUContext& context,
                  const std::vector<Vector<DIM>>& originsData,
                  const std::vector<Vector<DIM>>& directionsData) {
        origins.allocate<Vector<DIM>>(context, false, originsData);
        directions.allocate<Vector<DIM>>(context, false, directionsData);
        nRays = (uint32_t)originsData.size();
        distAlongRay.allocate<float>(context, true, std::vector<float>(nRays, 0.0f));
        intersectionPoints.allocate<Vector<DIM>>(context, true, std::vector<Vector<DIM>>(nRays, Vector<DIM>::Zero()));
        intersectionNormals.allocate<Vector<DIM>>(context, true, std::vector<Vector<DIM>>(nRays, Vector<DIM>::Zero()));
    }

    // set resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const {
        cursor.getPath("origins").setBinding(origins.buffer);
        cursor.getPath("directions").setBinding(directions.buffer);
        cursor.getPath("distAlongRay").setBinding(distAlongRay.buffer);
        cursor.getPath("intersectionPoints").setBinding(intersectionPoints.buffer);
        cursor.getPath("intersectionNormals").setBinding(intersectionNormals.buffer);
        cursor.getPath("nRays").setData(nRays);
        if (printLogs) printReflectionInfo(cursor, 7, "intersectBoundary");
    }

    // read results
    void read(GPUContext& context,
              std::vector<float>& distAlongRayData,
              std::vector<Vector<DIM>>& intersectionPointsData,
              std::vector<Vector<DIM>>& intersectionNormalsData) const {
        distAlongRay.read(context, distAlongRayData);
        intersectionPoints.read(context, intersectionPointsData);
        intersectionNormals.read(context, intersectionNormalsData);
    }

private:
    // members
    GPUBuffer origins = {};
    GPUBuffer directions = {};
    GPUBuffer distAlongRay = {};
    GPUBuffer intersectionPoints = {};
    GPUBuffer intersectionNormals = {};
    uint32_t nRays = 0;
};

template <typename T, size_t DIM>
void GPUDistanceQueries<T, DIM>::intersectBoundaryImpl(const ComputeShader& shader,
                                                       const std::vector<Vector<DIM>>& origins,
                                                       const std::vector<Vector<DIM>>& directions,
                                                       std::vector<float>& distAlongRay,
                                                       std::vector<Vector<DIM>>& intersectionPoints,
                                                       std::vector<Vector<DIM>>& intersectionNormals)
{
    // allocate entry point data
    GPUIntersectBoundary<DIM> entryPoint;
    entryPoint.allocate(context, origins, directions);

    // run shader
    uint32_t nRays = (uint32_t)origins.size();
    uint32_t nThreadGroups = countThreadGroups(nRays, nThreadsPerGroup, printLogs);
    runShader<GPUIntersectBoundary<DIM>>(context, shader, entryPoint, bindShaderResources,
                                         nThreadGroups, 1, printLogs);

    // read results from GPU
    entryPoint.read(context, distAlongRay, intersectionPoints, intersectionNormals);
}

template <typename T, size_t DIM>
void GPUDistanceQueries<T, DIM>::intersectBoundary(const std::vector<Vector<DIM>>& origins,
                                                   const std::vector<Vector<DIM>>& directions,
                                                   std::vector<float>& distAlongRay,
                                                   std::vector<Vector<DIM>>& intersectionPoints,
                                                   std::vector<Vector<DIM>>& intersectionNormals)
{
    intersectBoundaryImpl(intersectBoundaryShader, origins, directions,
                          distAlongRay, intersectionPoints, intersectionNormals);
}

template <typename T, size_t DIM>
void GPUDistanceQueries<T, DIM>::intersectAbsorbingBoundary(const std::vector<Vector<DIM>>& origins,
                                                            const std::vector<Vector<DIM>>& directions,
                                                            std::vector<float>& distAlongRay,
                                                            std::vector<Vector<DIM>>& intersectionPoints,
                                                            std::vector<Vector<DIM>>& intersectionNormals)
{
    intersectBoundaryImpl(intersectAbsorbingBoundaryShader, origins, directions,
                          distAlongRay, intersectionPoints, intersectionNormals);
}

template <typename T, size_t DIM>
void GPUDistanceQueries<T, DIM>::intersectReflectingBoundary(const std::vector<Vector<DIM>>& origins,
                                                             const std::vector<Vector<DIM>>& directions,
                                                             std::vector<float>& distAlongRay,
                                                             std::vector<Vector<DIM>>& intersectionPoints,
                                                             std::vector<Vector<DIM>>& intersectionNormals)
{
    intersectBoundaryImpl(intersectReflectingBoundaryShader, origins, directions,
                          distAlongRay, intersectionPoints, intersectionNormals);
}

} // wosx
