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

#include <wosx/utils/sdf_grid_geometric_queries.h>
#include <wosx/gpu/entrypoints/interop/reflectance_bvh.h>
#include <wosx/gpu/entrypoints/interop/dense_grid.h>

namespace wosx {

enum class GPUAbsorbingBoundaryType: int {
    Empty = 0,
    MeshBased = 1,
    SdfBased = 2
};

enum class GPUReflectingBoundaryType: int {
    Empty = 0,
    MeshBasedNeumann = 1,
    MeshBasedRobin = 2
};

class GPUBoundaryHandler: public GPUShaderObject {
public:
    // methods to allocate GPU resources, and return backend type
    virtual void allocate(GPUContext& context) = 0;
    virtual std::string getBackendType() const = 0;
    virtual std::string getName() const = 0;
};

class GPUAbsorbingBoundaryHandler: public GPUBoundaryHandler {
public:
    // returns boundary type
    virtual GPUAbsorbingBoundaryType getBoundaryType() const = 0;
};

class GPUEmptyAbsorbingBoundaryHandler: public GPUAbsorbingBoundaryHandler {
public:
    // methods to allocate and set GPU resources, and return type info
    void allocate(GPUContext& context);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    std::string getBackendType() const;
    std::string getName() const;
    GPUAbsorbingBoundaryType getBoundaryType() const;
};

template <size_t DIM>
class GPUFcpwDirichletBoundaryHandler: public GPUAbsorbingBoundaryHandler {
public:
    // constructor
    GPUFcpwDirichletBoundaryHandler(const std::vector<Vector<DIM>>& positions_,
                                    const std::vector<Vectori<DIM>>& indices_,
                                    bool printStats_=true);

    // methods to allocate and set GPU resources, and return type info
    void allocate(GPUContext& context);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    std::string getBackendType() const;
    std::string getName() const;
    GPUAbsorbingBoundaryType getBoundaryType() const;

private:
    // members
    FcpwDirichletBoundaryHandler<DIM> boundaryHandler;
    GPUBvhBuffers bvhBuffers;
};

template <size_t DIM>
class GPUSdfDirichletBoundaryHandler: public GPUAbsorbingBoundaryHandler {
public:
    // constructors
    GPUSdfDirichletBoundaryHandler(const Eigen::VectorXf& sdfData_,
                                   const Vectori<DIM>& gridShape_,
                                   const Vector<DIM>& gridMin_,
                                   const Vector<DIM>& gridMax_,
                                   bool unorderedAccess_=false);
    GPUSdfDirichletBoundaryHandler(std::function<Array<float, 1>(const Vector<DIM>&)> sdfDataCallback_,
                                   const Vectori<DIM>& gridShape_,
                                   const Vector<DIM>& gridMin_,
                                   const Vector<DIM>& gridMax_,
                                   bool unorderedAccess_=false);

    // methods to allocate and set GPU resources, and return type info
    void allocate(GPUContext& context);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    std::string getBackendType() const;
    std::string getName() const;
    GPUAbsorbingBoundaryType getBoundaryType() const;

private:
    // member
    GPUSdfGrid<DIM> sdfGrid;
};

class GPUReflectingBoundaryHandler: public GPUBoundaryHandler {
public:
    // returns boundary type
    virtual GPUReflectingBoundaryType getBoundaryType() const = 0;
};

class GPUEmptyReflectingBoundaryHandler: public GPUReflectingBoundaryHandler {
public:
    // methods to allocate and set GPU resources, and return type info
    void allocate(GPUContext& context);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    std::string getBackendType() const;
    std::string getName() const;
    GPUReflectingBoundaryType getBoundaryType() const;
};

template <size_t DIM>
class GPUFcpwNeumannBoundaryHandler: public GPUReflectingBoundaryHandler {
public:
    // constructor
    GPUFcpwNeumannBoundaryHandler(const std::vector<Vector<DIM>>& positions_,
                                  const std::vector<Vectori<DIM>>& indices_,
                                  std::function<bool(float, int)> ignoreCandidateSilhouette_,
                                  bool printStats_=true);

    // methods to allocate and set GPU resources, and return type info
    void allocate(GPUContext& context);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    std::string getBackendType() const;
    std::string getName() const;
    GPUReflectingBoundaryType getBoundaryType() const;

private:
    // members
    FcpwNeumannBoundaryHandler<DIM> boundaryHandler;
    GPUBvhBuffers bvhBuffers;
};

template <size_t DIM>
class GPUFcpwRobinBoundaryHandler: public GPUReflectingBoundaryHandler {
public:
    GPUFcpwRobinBoundaryHandler(const std::vector<Vector<DIM>>& positions_,
                                const std::vector<Vectori<DIM>>& indices_,
                                std::function<bool(float, int)> ignoreCandidateSilhouette_,
                                const std::vector<float>& minRobinCoeffValues_,
                                const std::vector<float>& maxRobinCoeffValues_,
                                bool printStats_=true);

    // methods to allocate, reallocate and set GPU resources, and return type info
    void allocate(GPUContext& context);
    void reallocate(GPUContext& context,
                    const std::vector<float>& minRobinCoeffValues,
                    const std::vector<float>& maxRobinCoeffValues);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    std::string getBackendType() const;
    std::string getName() const;
    GPUReflectingBoundaryType getBoundaryType() const;

private:
    // allocate GPU resources
    void allocate(GPUContext& context, bool allocatePrimitiveData,
                  bool allocateNodeData, bool allocateUpdateData);

    // members
    FcpwRobinBoundaryHandler<DIM> boundaryHandler;
    GPUReflectanceBvhBuffers bvhBuffers;
};

template <size_t DIM>
class GPUGeometricQueries: public GPUShaderObject {
public:
    // constructor
    GPUGeometricQueries(std::shared_ptr<GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler_,
                        std::shared_ptr<GPUReflectingBoundaryHandler> reflectingBoundaryHandler_,
                        const Vector<DIM>& domainMin_,
                        const Vector<DIM>& domainMax_,
                        bool domainIsWatertight_);

    // methods to allocate, reallocate and set GPU resources, and return type info
    void allocate(GPUContext& context);
    void reallocate(GPUContext& context,
                    std::shared_ptr<GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler_,
                    std::shared_ptr<GPUReflectingBoundaryHandler> reflectingBoundaryHandler_,
                    const Vector<DIM>& domainMin_,
                    const Vector<DIM>& domainMax_,
                    bool domainIsWatertight_);
    void reallocateAbsorbingBoundary(GPUContext& context,
                                     std::shared_ptr<GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler_,
                                     const Vector<DIM>& domainMin_,
                                     const Vector<DIM>& domainMax_,
                                     bool domainIsWatertight_);
    void reallocateReflectingBoundary(GPUContext& context,
                                      std::shared_ptr<GPUReflectingBoundaryHandler> reflectingBoundaryHandler_,
                                      const Vector<DIM>& domainMin_,
                                      const Vector<DIM>& domainMax_,
                                      bool domainIsWatertight_);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    GPUAbsorbingBoundaryType getAbsorbingBoundaryType() const;
    GPUReflectingBoundaryType getReflectingBoundaryType() const;

    // returns boundary handlers
    std::shared_ptr<GPUAbsorbingBoundaryHandler> getAbsorbingBoundaryHandler() const;
    std::shared_ptr<GPUReflectingBoundaryHandler> getReflectingBoundaryHandler() const;

private:
    // methods to return reflection types
    std::string getAbsorbingBoundaryReflectionType() const;
    std::string getReflectingBoundaryReflectionType() const;

    // members
    std::shared_ptr<GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler;
    std::shared_ptr<GPUReflectingBoundaryHandler> reflectingBoundaryHandler;
    Vector<DIM> domainMin;
    Vector<DIM> domainMax;
    uint32_t domainIsWatertight;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

inline void GPUEmptyAbsorbingBoundaryHandler::allocate(GPUContext& context)
{
    // do nothing
}

inline void GPUEmptyAbsorbingBoundaryHandler::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    // do nothing
}

inline std::string GPUEmptyAbsorbingBoundaryHandler::getReflectionType() const
{
    return "";
}

inline std::string GPUEmptyAbsorbingBoundaryHandler::getBackendType() const
{
    return "Empty";
}

inline std::string GPUEmptyAbsorbingBoundaryHandler::getName() const
{
    return "";
}

inline GPUAbsorbingBoundaryType GPUEmptyAbsorbingBoundaryHandler::getBoundaryType() const
{
    return GPUAbsorbingBoundaryType::Empty;
}

template <size_t DIM>
GPUFcpwDirichletBoundaryHandler<DIM>::GPUFcpwDirichletBoundaryHandler(const std::vector<Vector<DIM>>& positions_,
                                                                      const std::vector<Vectori<DIM>>& indices_,
                                                                      bool printStats_)
{
    boundaryHandler.buildAccelerationStructure(positions_, indices_, true, false, printStats_);
}

template <size_t DIM>
void GPUFcpwDirichletBoundaryHandler<DIM>::allocate(GPUContext& context)
{
    const bool allocatePrimitiveData = true;
    const bool allocateSilhouetteData = false;
    const bool allocateNodeData = true;
    const bool allocateRefitData = false;
    bvhBuffers.allocate<DIM>(context, boundaryHandler.scene.getSceneData(),
                             allocatePrimitiveData, allocateSilhouetteData,
                             allocateNodeData, allocateRefitData);
}

template <size_t DIM>
void GPUFcpwDirichletBoundaryHandler<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    bvhBuffers.setResources(cursor, printLogs);
}

template <size_t DIM>
std::string GPUFcpwDirichletBoundaryHandler<DIM>::getReflectionType() const
{
    return bvhBuffers.getReflectionType();
}

template <size_t DIM>
std::string GPUFcpwDirichletBoundaryHandler<DIM>::getBackendType() const
{
    return "Fcpw";
}

template <size_t DIM>
std::string GPUFcpwDirichletBoundaryHandler<DIM>::getName() const
{
    return "absorbingBoundaryAggregate";
}

template <size_t DIM>
GPUAbsorbingBoundaryType GPUFcpwDirichletBoundaryHandler<DIM>::getBoundaryType() const
{
    return GPUAbsorbingBoundaryType::MeshBased;
}

template <size_t DIM>
GPUSdfDirichletBoundaryHandler<DIM>::GPUSdfDirichletBoundaryHandler(const Eigen::VectorXf& sdfData_,
                                                                    const Vectori<DIM>& gridShape_,
                                                                    const Vector<DIM>& gridMin_,
                                                                    const Vector<DIM>& gridMax_,
                                                                    bool unorderedAccess_):
sdfGrid(sdfData_, gridShape_, gridMin_, gridMax_, unorderedAccess_)
{
    // do nothing
}

template <size_t DIM>
GPUSdfDirichletBoundaryHandler<DIM>::GPUSdfDirichletBoundaryHandler(std::function<Array<float, 1>(const Vector<DIM>&)> sdfDataCallback_,
                                                                    const Vectori<DIM>& gridShape_,
                                                                    const Vector<DIM>& gridMin_,
                                                                    const Vector<DIM>& gridMax_,
                                                                    bool unorderedAccess_):
sdfGrid(sdfDataCallback_, gridShape_, gridMin_, gridMax_, unorderedAccess_)
{
    // do nothing
}

template <size_t DIM>
void GPUSdfDirichletBoundaryHandler<DIM>::allocate(GPUContext& context)
{
    sdfGrid.allocate(context);
}

template <size_t DIM>
void GPUSdfDirichletBoundaryHandler<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    sdfGrid.setResources(cursor, printLogs);
}

template <size_t DIM>
std::string GPUSdfDirichletBoundaryHandler<DIM>::getReflectionType() const
{
    return sdfGrid.getReflectionType();
}

template <size_t DIM>
std::string GPUSdfDirichletBoundaryHandler<DIM>::getBackendType() const
{
    return "Sdf";
}

template <size_t DIM>
std::string GPUSdfDirichletBoundaryHandler<DIM>::getName() const
{
    return "sdfGrid";
}

template <size_t DIM>
GPUAbsorbingBoundaryType GPUSdfDirichletBoundaryHandler<DIM>::getBoundaryType() const
{
    return GPUAbsorbingBoundaryType::SdfBased;
}

inline void GPUEmptyReflectingBoundaryHandler::allocate(GPUContext& context)
{
    // do nothing
}

inline void GPUEmptyReflectingBoundaryHandler::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    // do nothing
}

inline std::string GPUEmptyReflectingBoundaryHandler::getReflectionType() const
{
    return "";
}

inline std::string GPUEmptyReflectingBoundaryHandler::getBackendType() const
{
    return "Empty";
}

inline std::string GPUEmptyReflectingBoundaryHandler::getName() const
{
    return "";
}

inline GPUReflectingBoundaryType GPUEmptyReflectingBoundaryHandler::getBoundaryType() const
{
    return GPUReflectingBoundaryType::Empty;
}

template <size_t DIM>
GPUFcpwNeumannBoundaryHandler<DIM>::GPUFcpwNeumannBoundaryHandler(const std::vector<Vector<DIM>>& positions_,
                                                                  const std::vector<Vectori<DIM>>& indices_,
                                                                  std::function<bool(float, int)> ignoreCandidateSilhouette_,
                                                                  bool printStats_)
{
    boundaryHandler.buildAccelerationStructure(positions_, indices_, ignoreCandidateSilhouette_,
                                               true, false, printStats_);
}

template <size_t DIM>
void GPUFcpwNeumannBoundaryHandler<DIM>::allocate(GPUContext& context)
{
    const bool allocatePrimitiveData = true;
    const bool allocateSilhouetteData = true;
    const bool allocateNodeData = true;
    const bool allocateRefitData = false;
    bvhBuffers.allocate<DIM>(context, boundaryHandler.scene.getSceneData(),
                             allocatePrimitiveData, allocateSilhouetteData,
                             allocateNodeData, allocateRefitData);
}

template <size_t DIM>
void GPUFcpwNeumannBoundaryHandler<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    bvhBuffers.setResources(cursor, printLogs);
}

template <size_t DIM>
std::string GPUFcpwNeumannBoundaryHandler<DIM>::getReflectionType() const
{
    return bvhBuffers.getReflectionType();
}

template <size_t DIM>
std::string GPUFcpwNeumannBoundaryHandler<DIM>::getBackendType() const
{
    return "Fcpw";
}

template <size_t DIM>
std::string GPUFcpwNeumannBoundaryHandler<DIM>::getName() const
{
    return "reflectingBoundaryAggregate";
}

template <size_t DIM>
GPUReflectingBoundaryType GPUFcpwNeumannBoundaryHandler<DIM>::getBoundaryType() const
{
    return GPUReflectingBoundaryType::MeshBasedNeumann;
}

template <size_t DIM>
GPUFcpwRobinBoundaryHandler<DIM>::GPUFcpwRobinBoundaryHandler(const std::vector<Vector<DIM>>& positions_,
                                                              const std::vector<Vectori<DIM>>& indices_,
                                                              std::function<bool(float, int)> ignoreCandidateSilhouette_,
                                                              const std::vector<float>& minRobinCoeffValues_,
                                                              const std::vector<float>& maxRobinCoeffValues_,
                                                              bool printStats_)
{
    boundaryHandler.buildAccelerationStructure(positions_, indices_, ignoreCandidateSilhouette_,
                                               minRobinCoeffValues_, maxRobinCoeffValues_,
                                               true, false, printStats_);
}

template <size_t DIM>
void GPUFcpwRobinBoundaryHandler<DIM>::allocate(GPUContext& context)
{
    const bool allocatePrimitiveData = true;
    const bool allocateNodeData = true;
    const bool allocateUpdateData = false;
    allocate(context, allocatePrimitiveData, allocateNodeData, allocateUpdateData);
}

template <size_t DIM>
void GPUFcpwRobinBoundaryHandler<DIM>::reallocate(GPUContext& context,
                                                  const std::vector<float>& minRobinCoeffValues,
                                                  const std::vector<float>& maxRobinCoeffValues)
{
    boundaryHandler.updateCoefficientValues(minRobinCoeffValues, maxRobinCoeffValues);
    allocate(context);
}

template <size_t DIM>
void GPUFcpwRobinBoundaryHandler<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    bvhBuffers.setResources(cursor, printLogs);
}

template <size_t DIM>
std::string GPUFcpwRobinBoundaryHandler<DIM>::getReflectionType() const
{
    return bvhBuffers.getReflectionType();
}

template <size_t DIM>
std::string GPUFcpwRobinBoundaryHandler<DIM>::getBackendType() const
{
    return "Fcpw";
}

template <size_t DIM>
std::string GPUFcpwRobinBoundaryHandler<DIM>::getName() const
{
    return "reflectingBoundaryAggregate";
}

template <size_t DIM>
GPUReflectingBoundaryType GPUFcpwRobinBoundaryHandler<DIM>::getBoundaryType() const
{
    return GPUReflectingBoundaryType::MeshBasedRobin;
}

template <size_t DIM>
void GPUFcpwRobinBoundaryHandler<DIM>::allocate(GPUContext& context, bool allocatePrimitiveData,
                                                bool allocateNodeData, bool allocateUpdateData)
{
    using PrimitiveBound = typename FcpwRobinBoundaryHandler<DIM>::PrimitiveBound;
    using NodeBound = typename FcpwRobinBoundaryHandler<DIM>::NodeBound;

    if constexpr (DIM == 2) {
        bvhBuffers.allocate<2, ReflectanceLineSegment<PrimitiveBound>,
                            GPUReflectanceLineSegment, NodeBound>(
            context, boundaryHandler.bvh.get(), allocatePrimitiveData,
            allocateNodeData, allocateUpdateData);

    } else if constexpr (DIM == 3) {
        bvhBuffers.allocate<3, ReflectanceTriangle<PrimitiveBound>,
                            GPUReflectanceTriangle, NodeBound>(
            context, boundaryHandler.bvh.get(), allocatePrimitiveData,
            allocateNodeData, allocateUpdateData);
    }
}

template <size_t DIM>
GPUGeometricQueries<DIM>::GPUGeometricQueries(std::shared_ptr<GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler_,
                                              std::shared_ptr<GPUReflectingBoundaryHandler> reflectingBoundaryHandler_,
                                              const Vector<DIM>& domainMin_,
                                              const Vector<DIM>& domainMax_,
                                              bool domainIsWatertight_):
absorbingBoundaryHandler(absorbingBoundaryHandler_),
reflectingBoundaryHandler(reflectingBoundaryHandler_),
domainMin(domainMin_),
domainMax(domainMax_),
domainIsWatertight(domainIsWatertight_ ? 1 : 0)
{
    // do nothing
}

template <size_t DIM>
void GPUGeometricQueries<DIM>::allocate(GPUContext& context)
{
    absorbingBoundaryHandler->allocate(context);
    reflectingBoundaryHandler->allocate(context);
}

template <size_t DIM>
void GPUGeometricQueries<DIM>::reallocate(GPUContext& context,
                                          std::shared_ptr<GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler_,
                                          std::shared_ptr<GPUReflectingBoundaryHandler> reflectingBoundaryHandler_,
                                          const Vector<DIM>& domainMin_,
                                          const Vector<DIM>& domainMax_,
                                          bool domainIsWatertight_)
{
    absorbingBoundaryHandler = absorbingBoundaryHandler_;
    reflectingBoundaryHandler = reflectingBoundaryHandler_;
    domainMin = domainMin_;
    domainMax = domainMax_;
    domainIsWatertight = domainIsWatertight_ ? 1 : 0;
    allocate(context);
}

template <size_t DIM>
void GPUGeometricQueries<DIM>::reallocateAbsorbingBoundary(GPUContext& context,
                                                           std::shared_ptr<GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler_,
                                                           const Vector<DIM>& domainMin_,
                                                           const Vector<DIM>& domainMax_,
                                                           bool domainIsWatertight_)
{
    absorbingBoundaryHandler = absorbingBoundaryHandler_;
    domainMin = domainMin_;
    domainMax = domainMax_;
    domainIsWatertight = domainIsWatertight_ ? 1 : 0;
    absorbingBoundaryHandler->allocate(context);
}

template <size_t DIM>
void GPUGeometricQueries<DIM>::reallocateReflectingBoundary(GPUContext& context,
                                                            std::shared_ptr<GPUReflectingBoundaryHandler> reflectingBoundaryHandler_,
                                                            const Vector<DIM>& domainMin_,
                                                            const Vector<DIM>& domainMax_,
                                                            bool domainIsWatertight_)
{
    reflectingBoundaryHandler = reflectingBoundaryHandler_;
    domainMin = domainMin_;
    domainMax = domainMax_;
    domainIsWatertight = domainIsWatertight_ ? 1 : 0;
    reflectingBoundaryHandler->allocate(context);
}

template <size_t DIM>
void GPUGeometricQueries<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    uint32_t hasNonEmptyAbsorbingBoundary = absorbingBoundaryHandler->getBackendType() != "Empty" ? 1 : 0;
    if (hasNonEmptyAbsorbingBoundary != 0) {
        ShaderCursor nestedCursor = cursor["absorbingBoundaryGeometricQueries"];
        std::string structureName = absorbingBoundaryHandler->getName();
        absorbingBoundaryHandler->setResources(nestedCursor[structureName.c_str()], printLogs);
        if (printLogs) {
            printReflectionInfo(nestedCursor, 1, getAbsorbingBoundaryReflectionType());
        }

    } else {
        if (printLogs) {
            std::cout << "Reflection: " << getAbsorbingBoundaryReflectionType() << std::endl;
        }
    }

    uint32_t hasNonEmptyReflectingBoundary = reflectingBoundaryHandler->getBackendType() != "Empty" ? 1 : 0;
    if (hasNonEmptyReflectingBoundary != 0) {
        ShaderCursor nestedCursor = cursor["reflectingBoundaryGeometricQueries"];
        std::string structureName = reflectingBoundaryHandler->getName();
        reflectingBoundaryHandler->setResources(nestedCursor[structureName.c_str()], printLogs);
        if (printLogs) {
            printReflectionInfo(nestedCursor, 1, getReflectingBoundaryReflectionType());
        }

    } else {
        if (printLogs) {
            std::cout << "Reflection: " << getReflectingBoundaryReflectionType() << std::endl;
        }
    }

    if (DIM == 2) {
        cursor["domainMin"].setData(float2{domainMin[0], domainMin[1]});
        cursor["domainMax"].setData(float2{domainMax[0], domainMax[1]});

    } else if (DIM == 3) {
        cursor["domainMin"].setData(float3{domainMin[0], domainMin[1], domainMin[2]});
        cursor["domainMax"].setData(float3{domainMax[0], domainMax[1], domainMax[2]});
    }
    cursor["domainIsWatertight"].setData(domainIsWatertight);
    cursor["hasNonEmptyAbsorbingBoundary"].setData(hasNonEmptyAbsorbingBoundary);
    cursor["hasNonEmptyReflectingBoundary"].setData(hasNonEmptyReflectingBoundary);
    if (printLogs) printReflectionInfo(cursor, 7, getReflectionType());
}

template <size_t DIM>
std::string GPUGeometricQueries<DIM>::getReflectionType() const
{
    std::string arguments = getAbsorbingBoundaryReflectionType() + ", " +
                            getReflectingBoundaryReflectionType() + ", " +
                            std::to_string(DIM);
    return "GeometricQueries<" + arguments + ">";
}

template <size_t DIM>
std::string GPUGeometricQueries<DIM>::getAbsorbingBoundaryReflectionType() const
{
    std::string arguments = std::to_string(DIM);
    std::string structureReflectionType = absorbingBoundaryHandler->getReflectionType();
    if (!structureReflectionType.empty()) arguments = structureReflectionType + ", " + arguments;
    return absorbingBoundaryHandler->getBackendType() +
           "AbsorbingBoundaryGeometricQueries<" + arguments + ">";
}

template <size_t DIM>
std::string GPUGeometricQueries<DIM>::getReflectingBoundaryReflectionType() const
{
    std::string arguments = std::to_string(DIM);
    std::string structureReflectionType = reflectingBoundaryHandler->getReflectionType();
    if (!structureReflectionType.empty()) arguments = structureReflectionType + ", " + arguments;
    return reflectingBoundaryHandler->getBackendType() +
           "ReflectingBoundaryGeometricQueries<" + arguments + ">";
}

template <size_t DIM>
GPUAbsorbingBoundaryType GPUGeometricQueries<DIM>::getAbsorbingBoundaryType() const
{
    return absorbingBoundaryHandler->getBoundaryType();
}

template <size_t DIM>
GPUReflectingBoundaryType GPUGeometricQueries<DIM>::getReflectingBoundaryType() const
{
    return reflectingBoundaryHandler->getBoundaryType();
}

template <size_t DIM>
std::shared_ptr<GPUAbsorbingBoundaryHandler> GPUGeometricQueries<DIM>::getAbsorbingBoundaryHandler() const
{
    return absorbingBoundaryHandler;
}

template <size_t DIM>
std::shared_ptr<GPUReflectingBoundaryHandler> GPUGeometricQueries<DIM>::getReflectingBoundaryHandler() const
{
    return reflectingBoundaryHandler;
}

} // wosx
