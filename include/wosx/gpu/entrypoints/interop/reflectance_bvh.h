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

#include <wosx/utils/fcpw_geometric_queries.h>
#include <fcpw/fcpw_gpu.h>

namespace wosx {

enum class GPUReflectanceBvhType: int {
    LineSegmentRobinBound = 1,
    TriangleRobinBound = 2
};

using fcpw::float2;
using fcpw::float3;
using fcpw::uint2;
using fcpw::uint3;
using namespace fcpw;

struct GPUReflectanceBvhNode {
    // constructors
    GPUReflectanceBvhNode();
    GPUReflectanceBvhNode(const GPUBoundingBox& box_,
                          const GPUBoundingCone& cone_,
                          uint32_t nPrimitives_,
                          uint32_t offset_,
                          float minCoefficientValue_,
                          float maxCoefficientValue_);

    // members
    GPUBoundingBox box;
    GPUBoundingCone cone;
    uint32_t nPrimitives;
    uint32_t offset;
    float minCoefficientValue;
    float maxCoefficientValue;
};

struct GPUReflectanceLineSegment {
    // constructors
    GPUReflectanceLineSegment();
    GPUReflectanceLineSegment(float3 p0_, float3 p1_,
                              float3 n0_, float3 n1_,
                              int hasAdjacentFace0_,
                              int hasAdjacentFace1_,
                              float minCoefficientValue_,
                              float maxCoefficientValue_,
                              uint32_t index_);

    // members
    float3 p[2];
    float3 n[2];
    int hasAdjacentFace[2];
    float minCoefficientValue;
    float maxCoefficientValue;
    uint32_t index;
};

struct GPUReflectanceTriangle {
    // constructors
    GPUReflectanceTriangle();
    GPUReflectanceTriangle(float3 p0_, float3 p1_, float3 p2_,
                           float3 n0_, float3 n1_, float3 n2_,
                           int hasAdjacentFace0_,
                           int hasAdjacentFace1_,
                           int hasAdjacentFace2_,
                           float minCoefficientValue_,
                           float maxCoefficientValue_,
                           uint32_t index_);

    // members
    float3 p[3];
    float3 n[3];
    int hasAdjacentFace[3];
    float minCoefficientValue;
    float maxCoefficientValue;
    uint32_t index;
};

class GPUReflectanceBvhBuffers: public GPUShaderObject {
public:
    // members
    GPUBuffer nodes = {};
    GPUBuffer primitives = {};
    GPUBuffer nodeIndices = {};
    std::vector<std::pair<uint32_t, uint32_t>> updateEntryData;
    uint32_t maxUpdateDepth = 0;
    std::string reflectionType = "";

    // allocates GPU resources
    template <size_t DIM, typename PrimitiveType, typename GPUPrimitiveType, typename NodeBound>
    void allocate(GPUContext& context,
                  const ReflectanceBvh<DIM, ReflectanceBvhNode<DIM>, PrimitiveType, NodeBound> *bvh,
                  bool allocatePrimitiveData, bool allocateNodeData, bool allocateUpdateData);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

    // returns reflection types
    std::string getReflectionType() const;

private:
    // allocates GPU geometry buffer
    template <size_t DIM, typename PrimitiveType, typename GPUPrimitiveType, typename NodeBound>
    void allocateGeometryBuffers(GPUContext& context,
                                 const ReflectanceBvh<DIM, ReflectanceBvhNode<DIM>, PrimitiveType, NodeBound> *bvh);

    // allocates GPU node buffer
    template <size_t DIM, typename PrimitiveType, typename GPUPrimitiveType, typename NodeBound>
    void allocateNodeBuffer(GPUContext& context,
                            const ReflectanceBvh<DIM, ReflectanceBvhNode<DIM>, PrimitiveType, NodeBound> *bvh);

    // allocates GPU update buffer
    template <size_t DIM, typename PrimitiveType, typename NodeBound>
    void allocateUpdateBuffer(GPUContext& context,
                              const ReflectanceBvh<DIM, ReflectanceBvhNode<DIM>, PrimitiveType, NodeBound> *bvh);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

GPUReflectanceBvhNode::GPUReflectanceBvhNode()
{
    box = GPUBoundingBox();
    cone = GPUBoundingCone();
    nPrimitives = 0;
    offset = 0;
    minCoefficientValue = 0.0f;
    maxCoefficientValue = maxFloat;
}

GPUReflectanceBvhNode::GPUReflectanceBvhNode(const GPUBoundingBox& box_,
                                             const GPUBoundingCone& cone_,
                                             uint32_t nPrimitives_,
                                             uint32_t offset_,
                                             float minCoefficientValue_,
                                             float maxCoefficientValue_):
                                             box(box_),
                                             cone(cone_),
                                             nPrimitives(nPrimitives_),
                                             offset(offset_),
                                             minCoefficientValue(minCoefficientValue_),
                                             maxCoefficientValue(maxCoefficientValue_)
{
    // do nothing
}

GPUReflectanceLineSegment::GPUReflectanceLineSegment()
{
    for (int i = 0; i < 2; i++) {
        p[i] = float3{0.0f, 0.0f, 0.0f};
        n[i] = float3{0.0f, 0.0f, 0.0f};
        hasAdjacentFace[i] = 1;
    }
    minCoefficientValue = 0.0f;
    maxCoefficientValue = maxFloat;
    index = FCPW_GPU_UINT_MAX;
}

GPUReflectanceLineSegment::GPUReflectanceLineSegment(float3 p0_, float3 p1_,
                                                     float3 n0_, float3 n1_,
                                                     int hasAdjacentFace0_,
                                                     int hasAdjacentFace1_,
                                                     float minCoefficientValue_,
                                                     float maxCoefficientValue_,
                                                     uint32_t index_)
{
    p[0] = p0_;
    p[1] = p1_;
    n[0] = n0_;
    n[1] = n1_;
    hasAdjacentFace[0] = hasAdjacentFace0_;
    hasAdjacentFace[1] = hasAdjacentFace1_;
    minCoefficientValue = minCoefficientValue_;
    maxCoefficientValue = maxCoefficientValue_;
    index = index_;
}

GPUReflectanceTriangle::GPUReflectanceTriangle()
{
    for (int i = 0; i < 3; i++) {
        p[i] = float3{0.0f, 0.0f, 0.0f};
        n[i] = float3{0.0f, 0.0f, 0.0f};
        hasAdjacentFace[i] = 1;
    }
    minCoefficientValue = 0.0f;
    maxCoefficientValue = maxFloat;
    index = FCPW_GPU_UINT_MAX;
}

GPUReflectanceTriangle::GPUReflectanceTriangle(float3 p0_, float3 p1_, float3 p2_,
                                               float3 n0_, float3 n1_, float3 n2_,
                                               int hasAdjacentFace0_,
                                               int hasAdjacentFace1_,
                                               int hasAdjacentFace2_,
                                               float minCoefficientValue_,
                                               float maxCoefficientValue_,
                                               uint32_t index_)
{
    p[0] = p0_;
    p[1] = p1_;
    p[2] = p2_;
    n[0] = n0_;
    n[1] = n1_;
    n[2] = n2_;
    hasAdjacentFace[0] = hasAdjacentFace0_;
    hasAdjacentFace[1] = hasAdjacentFace1_;
    hasAdjacentFace[2] = hasAdjacentFace2_;
    minCoefficientValue = minCoefficientValue_;
    maxCoefficientValue = maxCoefficientValue_;
    index = index_;
}

template <size_t DIM>
void extractReflectanceBvhNodes(const std::vector<ReflectanceBvhNode<DIM>>& flatTree,
                                std::vector<GPUReflectanceBvhNode>& reflectanceBvhNodes)
{
    int nNodes = (int)flatTree.size();
    reflectanceBvhNodes.resize(nNodes);

    for (int i = 0; i < nNodes; i++) {
        const ReflectanceBvhNode<DIM>& node = flatTree[i];
        const Vector<DIM>& pMin = node.box.pMin;
        const Vector<DIM>& pMax = node.box.pMax;
        const Vector<DIM>& axis = node.cone.axis;
        float halfAngle = node.cone.halfAngle;
        float radius = node.cone.radius;
        uint32_t nPrimitives = node.nReferences;
        uint32_t offset = nPrimitives > 0 ? node.referenceOffset : node.secondChildOffset;
        float minCoefficientValue = node.minCoefficientValue;
        float maxCoefficientValue = node.maxCoefficientValue;

        GPUBoundingBox boundingBox(float3{pMin[0], pMin[1], DIM == 2 ? 0.0f : pMin[2]},
                                   float3{pMax[0], pMax[1], DIM == 2 ? 0.0f : pMax[2]});
        GPUBoundingCone boundingCone(float3{axis[0], axis[1], DIM == 2 ? 0.0f : axis[2]},
                                     halfAngle, radius);
        reflectanceBvhNodes[i] = GPUReflectanceBvhNode(boundingBox, boundingCone, nPrimitives, offset,
                                                       minCoefficientValue, maxCoefficientValue);
    }
}

template <typename PrimitiveBound>
void extractReflectanceLineSegments(const std::vector<ReflectanceLineSegment<PrimitiveBound> *>& primitives,
                                    std::vector<GPUReflectanceLineSegment>& reflectanceLineSegments)
{
    int nPrimitives = (int)primitives.size();
    reflectanceLineSegments.resize(nPrimitives);

    for (int i = 0; i < nPrimitives; i++) {
        const ReflectanceLineSegment<PrimitiveBound> *lineSegment = primitives[i];
        const Vector2& pa = lineSegment->soup->positions[lineSegment->indices[0]];
        const Vector2& pb = lineSegment->soup->positions[lineSegment->indices[1]];
        const Vector2& na = lineSegment->n[0];
        const Vector2& nb = lineSegment->n[1];
        int hasAdjacentFace0 = lineSegment->hasAdjacentFace[0] ? 1 : 0;
        if (hasAdjacentFace0 == 1 && lineSegment->ignoreAdjacentFace[0]) hasAdjacentFace0 = -1;
        int hasAdjacentFace1 = lineSegment->hasAdjacentFace[1] ? 1 : 0;
        if (hasAdjacentFace1 == 1 && lineSegment->ignoreAdjacentFace[1]) hasAdjacentFace1 = -1;

        reflectanceLineSegments[i] = GPUReflectanceLineSegment(float3{pa[0], pa[1], 0.0f},
                                                               float3{pb[0], pb[1], 0.0f},
                                                               float3{na[0], na[1], 0.0f},
                                                               float3{nb[0], nb[1], 0.0f},
                                                               hasAdjacentFace0,
                                                               hasAdjacentFace1,
                                                               lineSegment->minCoefficientValue,
                                                               lineSegment->maxCoefficientValue,
                                                               lineSegment->pIndex);
    }
}

template <typename PrimitiveBound>
void extractReflectanceTriangles(const std::vector<ReflectanceTriangle<PrimitiveBound> *>& primitives,
                                 std::vector<GPUReflectanceTriangle>& reflectanceTriangles)
{
    int nPrimitives = (int)primitives.size();
    reflectanceTriangles.resize(nPrimitives);

    for (int i = 0; i < nPrimitives; i++) {
        const ReflectanceTriangle<PrimitiveBound> *triangle = primitives[i];
        const Vector3& pa = triangle->soup->positions[triangle->indices[0]];
        const Vector3& pb = triangle->soup->positions[triangle->indices[1]];
        const Vector3& pc = triangle->soup->positions[triangle->indices[2]];
        const Vector3& na = triangle->n[0];
        const Vector3& nb = triangle->n[1];
        const Vector3& nc = triangle->n[2];
        int hasAdjacentFace0 = triangle->hasAdjacentFace[0] ? 1 : 0;
        if (hasAdjacentFace0 == 1 && triangle->ignoreAdjacentFace[0]) hasAdjacentFace0 = -1;
        int hasAdjacentFace1 = triangle->hasAdjacentFace[1] ? 1 : 0;
        if (hasAdjacentFace1 == 1 && triangle->ignoreAdjacentFace[1]) hasAdjacentFace1 = -1;
        int hasAdjacentFace2 = triangle->hasAdjacentFace[2] ? 1 : 0;
        if (hasAdjacentFace2 == 1 && triangle->ignoreAdjacentFace[2]) hasAdjacentFace2 = -1;

        reflectanceTriangles[i] = GPUReflectanceTriangle(float3{pa[0], pa[1], pa[2]},
                                                         float3{pb[0], pb[1], pb[2]},
                                                         float3{pc[0], pc[1], pc[2]},
                                                         float3{na[0], na[1], na[2]},
                                                         float3{nb[0], nb[1], nb[2]},
                                                         float3{nc[0], nc[1], nc[2]},
                                                         hasAdjacentFace0,
                                                         hasAdjacentFace1,
                                                         hasAdjacentFace2,
                                                         triangle->minCoefficientValue,
                                                         triangle->maxCoefficientValue,
                                                         triangle->pIndex);
    }
}

template <typename PrimitiveBound, typename NodeBound, typename PrimitiveType, typename GPUPrimitiveType, size_t DIM>
class CPUReflectanceBvhDataExtractor {
public:
    // constructor
    CPUReflectanceBvhDataExtractor(const ReflectanceBvh<DIM, ReflectanceBvhNode<DIM>, PrimitiveType, NodeBound> *bvh_) {
        std::cerr << "CPUReflectanceBvhDataExtractor() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    // populates GPU bvh nodes array from CPU bvh
    void extractNodes(std::vector<GPUReflectanceBvhNode>& nodes) {
        std::cerr << "CPUReflectanceBvhDataExtractor::extractNodes() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    // populates GPU bvh primitives array from CPU bvh
    void extractPrimitives(std::vector<GPUPrimitiveType>& primitives) {
        std::cerr << "CPUReflectanceBvhDataExtractor::extractPrimitives() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    // returns reflection type
    std::string getReflectionType() const {
        std::cerr << "CPUReflectanceBvhDataExtractor::getReflectionType() not supported" << std::endl;
        exit(EXIT_FAILURE);

        return "";
    }
};

template <typename PrimitiveBound, typename NodeBound>
class CPUReflectanceBvhDataExtractor<PrimitiveBound, NodeBound, ReflectanceLineSegment<PrimitiveBound>, GPUReflectanceLineSegment, 2> {
public:
    // constructor
    CPUReflectanceBvhDataExtractor(const ReflectanceBvh<2, ReflectanceBvhNode<2>, ReflectanceLineSegment<PrimitiveBound>, NodeBound> *bvh_): bvh(bvh_) {}

    // populates GPU bvh nodes array from CPU bvh
    void extractNodes(std::vector<GPUReflectanceBvhNode>& nodes) {
        extractReflectanceBvhNodes<2>(bvh->flatTree, nodes);
    }

    // populates GPU bvh primitives array from CPU bvh
    void extractPrimitives(std::vector<GPUReflectanceLineSegment>& primitives) {
        extractReflectanceLineSegments<PrimitiveBound>(bvh->primitives, primitives);
    }

    // returns reflection type
    std::string getReflectionType() const {
        if (std::is_same<PrimitiveBound, RobinLineSegmentBound>::value &&
            std::is_same<NodeBound, RobinBvhNodeBound<2>>::value) {
            return "ReflectanceBvh<ReflectanceLineSegment<RobinLineSegmentBound>, RobinBvhNodeBound2D>";
        }

        std::cerr << "CPUReflectanceBvhDataExtractor::getReflectionType(): unsupported template parameters" << std::endl;
        exit(EXIT_FAILURE);
        return "";
    }

    // member
    const ReflectanceBvh<2, ReflectanceBvhNode<2>, ReflectanceLineSegment<PrimitiveBound>, NodeBound> *bvh;
};

template <typename PrimitiveBound, typename NodeBound>
class CPUReflectanceBvhDataExtractor<PrimitiveBound, NodeBound, ReflectanceTriangle<PrimitiveBound>, GPUReflectanceTriangle, 3> {
public:
    // constructor
    CPUReflectanceBvhDataExtractor(const ReflectanceBvh<3, ReflectanceBvhNode<3>, ReflectanceTriangle<PrimitiveBound>, NodeBound> *bvh_): bvh(bvh_) {}

    // populates GPU bvh nodes array from CPU bvh
    void extractNodes(std::vector<GPUReflectanceBvhNode>& nodes) {
        extractReflectanceBvhNodes<3>(bvh->flatTree, nodes);
    }

    // populates GPU bvh primitives array from CPU bvh
    void extractPrimitives(std::vector<GPUReflectanceTriangle>& primitives) {
        extractReflectanceTriangles<PrimitiveBound>(bvh->primitives, primitives);
    }

    // returns reflection type
    std::string getReflectionType() const {
        if (std::is_same<PrimitiveBound, RobinTriangleBound>::value &&
            std::is_same<NodeBound, RobinBvhNodeBound<3>>::value) {
            return "ReflectanceBvh<ReflectanceTriangle<RobinTriangleBound>, RobinBvhNodeBound3D>";
        }

        std::cerr << "CPUReflectanceBvhDataExtractor::getReflectionType(): unsupported template parameters" << std::endl;
        exit(EXIT_FAILURE);
        return "";
    }

    // member
    const ReflectanceBvh<3, ReflectanceBvhNode<3>, ReflectanceTriangle<PrimitiveBound>, NodeBound> *bvh;
};

template <size_t DIM, typename PrimitiveType, typename GPUPrimitiveType, typename NodeBound>
void GPUReflectanceBvhBuffers::allocateGeometryBuffers(
    GPUContext& context, const ReflectanceBvh<DIM, ReflectanceBvhNode<DIM>, PrimitiveType, NodeBound> *bvh)
{
    // extract primitives data from cpu bvh
    CPUReflectanceBvhDataExtractor<typename PrimitiveType::Bound, NodeBound,
                                   PrimitiveType, GPUPrimitiveType, DIM> cpuReflectanceBvhDataExtractor(bvh);

    std::vector<GPUPrimitiveType> primitivesData;
    cpuReflectanceBvhDataExtractor.extractPrimitives(primitivesData);

    // allocate buffer
    primitives.allocate<GPUPrimitiveType>(context, false, primitivesData);
}

template <size_t DIM, typename PrimitiveType, typename GPUPrimitiveType, typename NodeBound>
void GPUReflectanceBvhBuffers::allocateNodeBuffer(
    GPUContext& context, const ReflectanceBvh<DIM, ReflectanceBvhNode<DIM>, PrimitiveType, NodeBound> *bvh)
{
    // extract nodes data from cpu bvh
    CPUReflectanceBvhDataExtractor<typename PrimitiveType::Bound, NodeBound,
                                   PrimitiveType, GPUPrimitiveType, DIM> cpuReflectanceBvhDataExtractor(bvh);

    std::vector<GPUReflectanceBvhNode> nodesData;
    cpuReflectanceBvhDataExtractor.extractNodes(nodesData);
    reflectionType = cpuReflectanceBvhDataExtractor.getReflectionType();

    // allocate buffer
    nodes.allocate<GPUReflectanceBvhNode>(context, true, nodesData);
}

template <size_t DIM, typename PrimitiveType, typename NodeBound>
void GPUReflectanceBvhBuffers::allocateUpdateBuffer(
    GPUContext& context, const ReflectanceBvh<DIM, ReflectanceBvhNode<DIM>, PrimitiveType, NodeBound> *bvh)
{
    // extract update data from cpu bvh
    CPUBvhUpdateDataExtractor<DIM,
                              ReflectanceBvhNode<DIM>,
                              PrimitiveType,
                              SilhouettePrimitive<DIM>> cpuBvhUpdateDataExtractor(bvh);

    updateEntryData.clear();
    std::vector<uint32_t> nodeIndicesData;
    maxUpdateDepth = cpuBvhUpdateDataExtractor.extract(nodeIndicesData, updateEntryData);

    // allocate buffer
    nodeIndices.allocate<uint32_t>(context, false, nodeIndicesData);
}

template <size_t DIM, typename PrimitiveType, typename GPUPrimitiveType, typename NodeBound>
void GPUReflectanceBvhBuffers::allocate(GPUContext& context,
                                        const ReflectanceBvh<DIM, ReflectanceBvhNode<DIM>, PrimitiveType, NodeBound> *bvh,
                                        bool allocatePrimitiveData, bool allocateNodeData, bool allocateUpdateData)
{
    if (allocatePrimitiveData) {
        allocateGeometryBuffers<DIM, PrimitiveType, GPUPrimitiveType, NodeBound>(context, bvh);
    }

    if (allocateNodeData) {
        allocateNodeBuffer<DIM, PrimitiveType, GPUPrimitiveType, NodeBound>(context, bvh);
    }

    if (allocateUpdateData) {
        allocateUpdateBuffer<DIM, PrimitiveType, NodeBound>(context, bvh);
    }
}

void GPUReflectanceBvhBuffers::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor["nodes"].setBinding(nodes.buffer);
    cursor["primitives"].setBinding(primitives.buffer);
    if (printLogs) printReflectionInfo(cursor, 2, reflectionType);
}

std::string GPUReflectanceBvhBuffers::getReflectionType() const
{
    return reflectionType;
}

} // wosx
