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

#include <wosx/variance_reduction/boundary_samplers.h>
#include <wosx/gpu/entrypoints/interop/geometric_queries.h>
#include <wosx/gpu/entrypoints/interop/sample_and_evaluation_points.h>
#include <wosx/gpu/entrypoints/interop/supported_datatypes.h>

namespace wosx {

enum class GPUSolveRegionType: int {
    Undefined = 0,
    BoundingBox = 1,
    WatertightDomain = 2
};

enum class GPUDomainSamplerType: int {
    Empty = 0,
    Uniform = 1
};

enum class GPUBoundarySamplerType: int {
    Empty = 0,
    UniformLineSegment = 1,
    UniformTriangle = 2
};

class GPUCDFTable: public GPUShaderObject {
public:
    // methods to allocate and set GPU resources, and return reflection type
    void allocate(GPUContext& context, const CDFTable& cdfTable);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;

private:
    // members
    GPUBuffer table = {};
    uint32_t size;
};

class GPUSolveRegion: public GPUShaderObject {
public:
    // returns region type
    virtual GPUSolveRegionType getRegionType() const = 0;
};

template <size_t DIM>
class GPUBoundingBoxSolveRegion: public GPUSolveRegion {
public:
    // constructor
    GPUBoundingBoxSolveRegion(const Vector<DIM>& regionMin_,
                              const Vector<DIM>& regionMax_);

    // methods to set GPU resources, and return type info
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    GPUSolveRegionType getRegionType() const;

private:
    // members
    Vector<DIM> regionMin;
    Vector<DIM> regionMax;
};

template <size_t DIM>
class GPUWatertightDomainSolveRegion: public GPUSolveRegion {
public:
    // constructor
    GPUWatertightDomainSolveRegion(std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries_,
                                   const Vector<DIM>& regionMin_,
                                   const Vector<DIM>& regionMax_,
                                   float regionVolume_);

    // methods to set GPU resources, and return type info
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    GPUSolveRegionType getRegionType() const;

private:
    // members
    std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries;
    Vector<DIM> regionMin;
    Vector<DIM> regionMax;
    float regionVolume;
};

class GPUDomainSampler: public GPUShaderObject {
public:
    // methods to allocate GPU resources, and returns region and sampler type
    virtual void allocate(GPUContext& context) = 0;
    virtual GPUSolveRegionType getRegionType() const = 0;
    virtual GPUDomainSamplerType getSamplerType() const = 0;
};

template <size_t DIM>
class GPUEmptyDomainSampler: public GPUDomainSampler {
public:
    // methods to allocate and set GPU resources, and return type info
    void allocate(GPUContext& context);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    GPUSolveRegionType getRegionType() const;
    GPUDomainSamplerType getSamplerType() const;
};

template <size_t DIM>
class GPUUniformDomainSampler: public GPUDomainSampler {
public:
    // constructor
    GPUUniformDomainSampler(std::shared_ptr<GPUSolveRegion> solveRegion_,
                            std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries_);

    // methods to allocate and set GPU resources, and return type info
    void allocate(GPUContext& context);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    GPUSolveRegionType getRegionType() const;
    GPUDomainSamplerType getSamplerType() const;

private:
    // members
    std::shared_ptr<GPUSolveRegion> solveRegion;
    std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries;
};

class GPUBoundarySampler: public GPUShaderObject {
public:
    // methods to allocate and reallocate GPU resources, and return sampler type
    virtual void allocate(GPUContext& context) = 0;
    virtual void reallocate(GPUContext& context, float normalOffsetForBoundary) = 0;
    virtual GPUBoundarySamplerType getSamplerType() const = 0;

    // returns the normal offset for the boundary
    virtual float getNormalOffsetForBoundary() const = 0;

    // returns the number of sample points to be generated on the user-specified side of the boundary
    virtual int getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples=false) const = 0;
};

template <size_t DIM>
class GPUEmptyBoundarySampler: public GPUBoundarySampler {
public:
    // methods to allocate, reallocate and set GPU resources, and return type info
    void allocate(GPUContext& context);
    void reallocate(GPUContext& context, float normalOffsetForBoundary);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    GPUBoundarySamplerType getSamplerType() const;

    // returns sampler related parameters
    float getNormalOffsetForBoundary() const;
    int getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples=false) const;
};

class GPUUniformLineSegmentBoundarySampler: public GPUBoundarySampler {
public:
    // constructor
    GPUUniformLineSegmentBoundarySampler(std::shared_ptr<GPUGeometricQueries<2>> geometricQueries_,
                                         const std::vector<Vector2>& positions_,
                                         const std::vector<Vector2i>& indices_,
                                         const std::vector<float>& primitiveWeights_,
                                         const std::vector<float>& primitiveWeightsNormalAligned_,
                                         std::function<bool(const Vector2&)> insideSolveRegion_,
                                         float normalOffsetForBoundary_,
                                         bool solveDoubleSided_,
                                         bool computeWeightedNormals_=false);

    // methods to allocate, reallocate and set GPU resources, and return type info
    void allocate(GPUContext& context);
    void reallocate(GPUContext& context, float normalOffsetForBoundary);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    GPUBoundarySamplerType getSamplerType() const;

    // returns sampler related parameters
    float getNormalOffsetForBoundary() const;
    int getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples=false) const;

private:
    // members
    GPUBuffer positions = {};
    GPUBuffer normals = {};
    GPUBuffer indices = {};
    GPUBuffer primitiveWeights = {};
    GPUBuffer primitiveWeightsNormalAligned = {};
    std::shared_ptr<GPUGeometricQueries<2>> geometricQueries;
    float normalOffsetForBoundary;
    bool solveDoubleSided;
    GPUCDFTable cdfTable;
    GPUCDFTable cdfTableNormalAligned;
    float boundarySamplingMass;
    float boundarySamplingMassNormalAligned;
    UniformLineSegmentBoundarySampler<float> sampler;
};

class GPUUniformTriangleBoundarySampler: public GPUBoundarySampler {
public:
    // constructor
    GPUUniformTriangleBoundarySampler(std::shared_ptr<GPUGeometricQueries<3>> geometricQueries_,
                                      const std::vector<Vector3>& positions_,
                                      const std::vector<Vector3i>& indices_,
                                      const std::vector<float>& primitiveWeights_,
                                      const std::vector<float>& primitiveWeightsNormalAligned_,
                                      std::function<bool(const Vector3&)> insideSolveRegion_,
                                      float normalOffsetForBoundary_,
                                      bool solveDoubleSided_,
                                      bool computeWeightedNormals_=false);

    // methods to allocate, reallocate and set GPU resources, and return type info
    void allocate(GPUContext& context);
    void reallocate(GPUContext& context, float normalOffsetForBoundary);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    GPUBoundarySamplerType getSamplerType() const;

    // returns sampler related parameters
    float getNormalOffsetForBoundary() const;
    int getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples=false) const;

private:
    // members
    GPUBuffer positions = {};
    GPUBuffer normals = {};
    GPUBuffer indices = {};
    GPUBuffer primitiveWeights = {};
    GPUBuffer primitiveWeightsNormalAligned = {};
    std::shared_ptr<GPUGeometricQueries<3>> geometricQueries;
    float normalOffsetForBoundary;
    bool solveDoubleSided;
    GPUCDFTable cdfTable;
    GPUCDFTable cdfTableNormalAligned;
    float boundarySamplingMass;
    float boundarySamplingMassNormalAligned;
    UniformTriangleBoundarySampler<float> sampler;
};

std::shared_ptr<GPUBoundarySampler> createUniformBoundarySampler(std::shared_ptr<GPUGeometricQueries<2>> geometricQueries,
                                                                 const std::vector<Vector2>& positions,
                                                                 const std::vector<Vector2i>& indices,
                                                                 std::function<bool(const Vector2&)> insideSolveRegion,
                                                                 float normalOffsetForBoundary,
                                                                 bool solveDoubleSided,
                                                                 bool computeWeightedNormals=false);
std::shared_ptr<GPUBoundarySampler> createUniformBoundarySampler(std::shared_ptr<GPUGeometricQueries<2>> geometricQueries,
                                                                 const std::vector<Vector2>& positions,
                                                                 const std::vector<Vector2i>& indices,
                                                                 const std::vector<float>& primitiveWeights,
                                                                 std::function<bool(const Vector2&)> insideSolveRegion,
                                                                 float normalOffsetForBoundary,
                                                                 bool solveDoubleSided,
                                                                 bool computeWeightedNormals=false);
std::shared_ptr<GPUBoundarySampler> createUniformBoundarySampler(std::shared_ptr<GPUGeometricQueries<2>> geometricQueries,
                                                                 const std::vector<Vector2>& positions,
                                                                 const std::vector<Vector2i>& indices,
                                                                 const std::vector<float>& primitiveWeights,
                                                                 const std::vector<float>& primitiveWeightsNormalAligned,
                                                                 std::function<bool(const Vector2&)> insideSolveRegion,
                                                                 float normalOffsetForBoundary,
                                                                 bool solveDoubleSided,
                                                                 bool computeWeightedNormals=false);

std::shared_ptr<GPUBoundarySampler> createUniformBoundarySampler(std::shared_ptr<GPUGeometricQueries<3>> geometricQueries,
                                                                 const std::vector<Vector3>& positions,
                                                                 const std::vector<Vector3i>& indices,
                                                                 std::function<bool(const Vector3&)> insideSolveRegion,
                                                                 float normalOffsetForBoundary,
                                                                 bool solveDoubleSided,
                                                                 bool computeWeightedNormals=false);
std::shared_ptr<GPUBoundarySampler> createUniformBoundarySampler(std::shared_ptr<GPUGeometricQueries<3>> geometricQueries,
                                                                 const std::vector<Vector3>& positions,
                                                                 const std::vector<Vector3i>& indices,
                                                                 const std::vector<float>& primitiveWeights,
                                                                 std::function<bool(const Vector3&)> insideSolveRegion,
                                                                 float normalOffsetForBoundary,
                                                                 bool solveDoubleSided,
                                                                 bool computeWeightedNormals=false);
std::shared_ptr<GPUBoundarySampler> createUniformBoundarySampler(std::shared_ptr<GPUGeometricQueries<3>> geometricQueries,
                                                                 const std::vector<Vector3>& positions,
                                                                 const std::vector<Vector3i>& indices,
                                                                 const std::vector<float>& primitiveWeights,
                                                                 const std::vector<float>& primitiveWeightsNormalAligned,
                                                                 std::function<bool(const Vector3&)> insideSolveRegion,
                                                                 float normalOffsetForBoundary,
                                                                 bool solveDoubleSided,
                                                                 bool computeWeightedNormals=false);

template <size_t DIM>
class GPUGenerateDomainSamples: public GPUShaderEntryPoint {
public:
    // constructor
    GPUGenerateDomainSamples(GPUBuffer& rngs_);

    // allocates GPU resources
    void allocate(GPUContext& context, uint32_t nSamplePoints_);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

    // reads results from GPU
    void read(GPUContext& context,
              std::vector<GPUSamplePoint<DIM>>& samplePointsData) const;

    // returns sample points buffer size
    uint32_t getBufferSize() const;

    // returns sample points buffer
    GPUBuffer& getBuffer();

private:
    // members
    GPUBuffer& rngs;
    GPUBuffer samplePoints = {};
    uint32_t nSamplePoints = 0;
};

template <size_t DIM>
class GPUGenerateBoundarySamples: public GPUShaderEntryPoint {
public:
    // constructor
    GPUGenerateBoundarySamples(GPUBuffer& rngs_,
                               SampleType sampleType_,
                               bool generateBoundaryNormalAligned_);

    // allocates GPU resources
    void allocate(GPUContext& context, uint32_t nSamplePoints_);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

    // reads results from GPU
    void read(GPUContext& context,
              std::vector<GPUSamplePoint<DIM>>& samplePointsData) const;

    // sets normal offset for boundary
    void setNormalOffsetForBoundary(float offset);

    // returns sample points buffer size
    uint32_t getBufferSize() const;

    // returns sample points buffer
    GPUBuffer& getBuffer();

private:
    // members
    GPUBuffer& rngs;
    GPUBuffer samplePoints = {};
    SampleType sampleType;
    float normalOffsetForBoundary = 0.0f;
    uint32_t generateBoundaryNormalAligned = 0;
    uint32_t nSamplePoints = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

void GPUCDFTable::allocate(GPUContext& context, const CDFTable& cdfTable)
{
    const std::vector<float>& tableData = cdfTable.getTable();
    table.allocate<float>(context, false, tableData);
    size = (uint32_t)tableData.size();
}

void GPUCDFTable::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor["table"].setBinding(table.buffer);
    cursor["size"].setData(size);
    if (printLogs) printReflectionInfo(cursor, 2, getReflectionType());
}

std::string GPUCDFTable::getReflectionType() const
{
    return "CDFTable";
}

template <size_t DIM>
GPUBoundingBoxSolveRegion<DIM>::GPUBoundingBoxSolveRegion(const Vector<DIM>& regionMin_,
                                                          const Vector<DIM>& regionMax_):
regionMin(regionMin_),
regionMax(regionMax_)
{
    // do nothing
}

template <size_t DIM>
void GPUBoundingBoxSolveRegion<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    if (DIM == 2) {
        cursor["regionMin"].setData(float2{regionMin[0], regionMin[1]});
        cursor["regionMax"].setData(float2{regionMax[0], regionMax[1]});

    } else if (DIM == 3) {
        cursor["regionMin"].setData(float3{regionMin[0], regionMin[1], regionMin[2]});
        cursor["regionMax"].setData(float3{regionMax[0], regionMax[1], regionMax[2]});
    }

    if (printLogs) {
        printReflectionInfo(cursor, 2, getReflectionType());
    }
}

template <size_t DIM>
std::string GPUBoundingBoxSolveRegion<DIM>::getReflectionType() const
{
    return "BoundingBoxSolveRegion<" + std::to_string(DIM) + ">";
}

template <size_t DIM>
GPUSolveRegionType GPUBoundingBoxSolveRegion<DIM>::getRegionType() const
{
    return GPUSolveRegionType::BoundingBox;
}

template <size_t DIM>
GPUWatertightDomainSolveRegion<DIM>::GPUWatertightDomainSolveRegion(std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries_,
                                                                    const Vector<DIM>& regionMin_,
                                                                    const Vector<DIM>& regionMax_,
                                                                    float regionVolume_):
geometricQueries(geometricQueries_),
regionMin(regionMin_),
regionMax(regionMax_),
regionVolume(regionVolume_)
{
    // do nothing
}

template <size_t DIM>
void GPUWatertightDomainSolveRegion<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    geometricQueries->setResources(cursor["geometricQueries"], printLogs);
    if (DIM == 2) {
        cursor["regionMin"].setData(float2{regionMin[0], regionMin[1]});
        cursor["regionMax"].setData(float2{regionMax[0], regionMax[1]});

    } else if (DIM == 3) {
        cursor["regionMin"].setData(float3{regionMin[0], regionMin[1], regionMin[2]});
        cursor["regionMax"].setData(float3{regionMax[0], regionMax[1], regionMax[2]});
    }
    cursor["regionVolume"].setData(regionVolume);
    if (printLogs) printReflectionInfo(cursor, 4, getReflectionType());
}

template <size_t DIM>
std::string GPUWatertightDomainSolveRegion<DIM>::getReflectionType() const
{
    std::string arguments = geometricQueries->getReflectionType() + ", " +
                            std::to_string(DIM);
    return "WatertightDomainSolveRegion<" + arguments + ">";
}

template <size_t DIM>
GPUSolveRegionType GPUWatertightDomainSolveRegion<DIM>::getRegionType() const
{
    return GPUSolveRegionType::WatertightDomain;
}

template <size_t DIM>
void GPUEmptyDomainSampler<DIM>::allocate(GPUContext& context)
{
    // do nothing
}

template <size_t DIM>
void GPUEmptyDomainSampler<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    // do nothing
}

template <size_t DIM>
std::string GPUEmptyDomainSampler<DIM>::getReflectionType() const
{
    return "EmptyDomainSampler<" + std::to_string(DIM) + ">";
}

template <size_t DIM>
GPUSolveRegionType GPUEmptyDomainSampler<DIM>::getRegionType() const
{
    return GPUSolveRegionType::Undefined;
}

template <size_t DIM>
GPUDomainSamplerType GPUEmptyDomainSampler<DIM>::getSamplerType() const
{
    return GPUDomainSamplerType::Empty;
}

template <size_t DIM>
GPUUniformDomainSampler<DIM>::GPUUniformDomainSampler(std::shared_ptr<GPUSolveRegion> solveRegion_,
                                                      std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries_):
solveRegion(solveRegion_),
geometricQueries(geometricQueries_)
{
    // do nothing
}

template <size_t DIM>
void GPUUniformDomainSampler<DIM>::allocate(GPUContext& context)
{
    // do nothing
}

template <size_t DIM>
void GPUUniformDomainSampler<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    solveRegion->setResources(cursor["solveRegion"], printLogs);
    geometricQueries->setResources(cursor["geometricQueries"], printLogs);
    if (printLogs) printReflectionInfo(cursor, 2, getReflectionType());
}

template <size_t DIM>
std::string GPUUniformDomainSampler<DIM>::getReflectionType() const
{
    std::string arguments = solveRegion->getReflectionType() + ", " +
                            geometricQueries->getReflectionType() + ", " +
                            std::to_string(DIM);
    return "UniformDomainSampler<" + arguments + ">";
}

template <size_t DIM>
GPUSolveRegionType GPUUniformDomainSampler<DIM>::getRegionType() const
{
    return solveRegion->getRegionType();
}

template <size_t DIM>
GPUDomainSamplerType GPUUniformDomainSampler<DIM>::getSamplerType() const
{
    return GPUDomainSamplerType::Uniform;
}

template <typename T, typename S>
std::vector<T> toGPUVectorType(const std::vector<S>& u) {
    std::cerr << "toGPUVectorType() not implemented for this data type." << std::endl;
    exit(EXIT_FAILURE);

    return std::vector<T>{};
}

template <>
std::vector<float2> toGPUVectorType(const std::vector<Vector2>& u) {
    int size = (int)u.size();
    std::vector<float2> v(size);
    for (int i = 0; i < size; i++) {
        v[i].x = u[i][0];
        v[i].y = u[i][1];
    }

    return v;
}

template <>
std::vector<uint2> toGPUVectorType(const std::vector<Vector2i>& u) {
    int size = (int)u.size();
    std::vector<uint2> v(size);
    for (int i = 0; i < size; i++) {
        v[i].x = u[i][0];
        v[i].y = u[i][1];
    }

    return v;
}

template <>
std::vector<float3> toGPUVectorType(const std::vector<Vector3>& u) {
    int size = (int)u.size();
    std::vector<float3> v(size);
    for (int i = 0; i < size; i++) {
        v[i].x = u[i][0];
        v[i].y = u[i][1];
        v[i].z = u[i][2];
    }

    return v;
}

template <>
std::vector<uint3> toGPUVectorType(const std::vector<Vector3i>& u) {
    int size = (int)u.size();
    std::vector<uint3> v(size);
    for (int i = 0; i < size; i++) {
        v[i].x = u[i][0];
        v[i].y = u[i][1];
        v[i].z = u[i][2];
    }

    return v;
}

template <size_t DIM>
void GPUEmptyBoundarySampler<DIM>::allocate(GPUContext& context)
{
    // do nothing
}

template <size_t DIM>
void GPUEmptyBoundarySampler<DIM>::reallocate(GPUContext& context, float normalOffsetForBoundary)
{
    // do nothing
}

template <size_t DIM>
void GPUEmptyBoundarySampler<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    // do nothing
}

template <size_t DIM>
std::string GPUEmptyBoundarySampler<DIM>::getReflectionType() const
{
    return "EmptyBoundarySampler<" + std::to_string(DIM) + ">";
}

template <size_t DIM>
GPUBoundarySamplerType GPUEmptyBoundarySampler<DIM>::getSamplerType() const
{
    return GPUBoundarySamplerType::Empty;
}

template <size_t DIM>
float GPUEmptyBoundarySampler<DIM>::getNormalOffsetForBoundary() const
{
    return 0.0f;
}

template <size_t DIM>
int GPUEmptyBoundarySampler<DIM>::getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples) const
{
    return 0;
}

GPUUniformLineSegmentBoundarySampler::GPUUniformLineSegmentBoundarySampler(std::shared_ptr<GPUGeometricQueries<2>> geometricQueries_,
                                                                           const std::vector<Vector2>& positions_,
                                                                           const std::vector<Vector2i>& indices_,
                                                                           const std::vector<float>& primitiveWeights_,
                                                                           const std::vector<float>& primitiveWeightsNormalAligned_,
                                                                           std::function<bool(const Vector2&)> insideSolveRegion_,
                                                                           float normalOffsetForBoundary_,
                                                                           bool solveDoubleSided_,
                                                                           bool computeWeightedNormals_):
geometricQueries(geometricQueries_),
normalOffsetForBoundary(normalOffsetForBoundary_),
solveDoubleSided(solveDoubleSided_),
boundarySamplingMass(0.0f),
boundarySamplingMassNormalAligned(0.0f),
sampler(positions_, indices_, primitiveWeights_, primitiveWeightsNormalAligned_,
        insideSolveRegion_, computeWeightedNormals_)
{
    sampler.initialize(normalOffsetForBoundary_, solveDoubleSided);
}

void GPUUniformLineSegmentBoundarySampler::allocate(GPUContext& context)
{
    std::vector<float2> positionsData = toGPUVectorType<float2, Vector2>(sampler.getPositions());
    positions.allocate<float2>(context, false, positionsData);
    std::vector<float2> normalsData = toGPUVectorType<float2, Vector2>(sampler.getNormals());
    normals.allocate<float2>(context, false, normalsData);
    std::vector<uint2> indicesData = toGPUVectorType<uint2, Vector2i>(sampler.getIndices());
    indices.allocate<uint2>(context, false, indicesData);
    primitiveWeights.allocate<float>(context, false, sampler.getPrimitiveWeights(false));
    primitiveWeightsNormalAligned.allocate<float>(context, false, sampler.getPrimitiveWeights(true));
    cdfTable.allocate(context, sampler.getCDFTable(false));
    cdfTableNormalAligned.allocate(context, sampler.getCDFTable(true));
    boundarySamplingMass = sampler.getBoundarySamplingMass(false);
    boundarySamplingMassNormalAligned = sampler.getBoundarySamplingMass(true);
}

void GPUUniformLineSegmentBoundarySampler::reallocate(GPUContext& context, float normalOffsetForBoundary)
{
    this->normalOffsetForBoundary = normalOffsetForBoundary;
    sampler.initialize(normalOffsetForBoundary, solveDoubleSided);
    cdfTable.allocate(context, sampler.getCDFTable(false));
    cdfTableNormalAligned.allocate(context, sampler.getCDFTable(true));
    boundarySamplingMass = sampler.getBoundarySamplingMass(false);
    boundarySamplingMassNormalAligned = sampler.getBoundarySamplingMass(true);
}

void GPUUniformLineSegmentBoundarySampler::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor["positions"].setBinding(positions.buffer);
    cursor["normals"].setBinding(normals.buffer);
    cursor["indices"].setBinding(indices.buffer);
    cursor["primitiveWeights"].setBinding(primitiveWeights.buffer);
    cursor["primitiveWeightsNormalAligned"].setBinding(primitiveWeightsNormalAligned.buffer);
    geometricQueries->setResources(cursor["geometricQueries"], printLogs);
    cdfTable.setResources(cursor["cdfTable"], printLogs);
    cdfTableNormalAligned.setResources(cursor["cdfTableNormalAligned"], printLogs);
    cursor["boundarySamplingMass"].setData(boundarySamplingMass);
    cursor["boundarySamplingMassNormalAligned"].setData(boundarySamplingMassNormalAligned);
    if (printLogs) printReflectionInfo(cursor, 10, getReflectionType());
}

std::string GPUUniformLineSegmentBoundarySampler::getReflectionType() const
{
    std::string arguments = geometricQueries->getReflectionType();
    return "UniformLineSegmentBoundarySampler<" + arguments + ">";
}

GPUBoundarySamplerType GPUUniformLineSegmentBoundarySampler::getSamplerType() const
{
    return GPUBoundarySamplerType::UniformLineSegment;
}

float GPUUniformLineSegmentBoundarySampler::getNormalOffsetForBoundary() const
{
    return normalOffsetForBoundary;
}

int GPUUniformLineSegmentBoundarySampler::getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples) const
{
    return sampler.getSampleCount(nTotalSamples, boundaryNormalAlignedSamples);
}

GPUUniformTriangleBoundarySampler::GPUUniformTriangleBoundarySampler(std::shared_ptr<GPUGeometricQueries<3>> geometricQueries_,
                                                                     const std::vector<Vector3>& positions_,
                                                                     const std::vector<Vector3i>& indices_,
                                                                     const std::vector<float>& primitiveWeights_,
                                                                     const std::vector<float>& primitiveWeightsNormalAligned_,
                                                                     std::function<bool(const Vector3&)> insideSolveRegion_,
                                                                     float normalOffsetForBoundary_,
                                                                     bool solveDoubleSided_,
                                                                     bool computeWeightedNormals_):
geometricQueries(geometricQueries_),
normalOffsetForBoundary(normalOffsetForBoundary_),
solveDoubleSided(solveDoubleSided_),
boundarySamplingMass(0.0f),
boundarySamplingMassNormalAligned(0.0f),
sampler(positions_, indices_, primitiveWeights_, primitiveWeightsNormalAligned_,
        insideSolveRegion_, computeWeightedNormals_)
{
    sampler.initialize(normalOffsetForBoundary_, solveDoubleSided);
}

void GPUUniformTriangleBoundarySampler::allocate(GPUContext& context)
{
    std::vector<float3> positionsData = toGPUVectorType<float3, Vector3>(sampler.getPositions());
    positions.allocate<float3>(context, false, positionsData);
    std::vector<float3> normalsData = toGPUVectorType<float3, Vector3>(sampler.getNormals());
    normals.allocate<float3>(context, false, normalsData);
    std::vector<uint3> indicesData = toGPUVectorType<uint3, Vector3i>(sampler.getIndices());
    indices.allocate<uint3>(context, false, indicesData);
    primitiveWeights.allocate<float>(context, false, sampler.getPrimitiveWeights(false));
    primitiveWeightsNormalAligned.allocate<float>(context, false, sampler.getPrimitiveWeights(true));
    cdfTable.allocate(context, sampler.getCDFTable(false));
    cdfTableNormalAligned.allocate(context, sampler.getCDFTable(true));
    boundarySamplingMass = sampler.getBoundarySamplingMass(false);
    boundarySamplingMassNormalAligned = sampler.getBoundarySamplingMass(true);
}

void GPUUniformTriangleBoundarySampler::reallocate(GPUContext& context, float normalOffsetForBoundary)
{
    this->normalOffsetForBoundary = normalOffsetForBoundary;
    sampler.initialize(normalOffsetForBoundary, solveDoubleSided);
    cdfTable.allocate(context, sampler.getCDFTable(false));
    cdfTableNormalAligned.allocate(context, sampler.getCDFTable(true));
    boundarySamplingMass = sampler.getBoundarySamplingMass(false);
    boundarySamplingMassNormalAligned = sampler.getBoundarySamplingMass(true);
}

void GPUUniformTriangleBoundarySampler::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor["positions"].setBinding(positions.buffer);
    cursor["normals"].setBinding(normals.buffer);
    cursor["indices"].setBinding(indices.buffer);
    cursor["primitiveWeights"].setBinding(primitiveWeights.buffer);
    cursor["primitiveWeightsNormalAligned"].setBinding(primitiveWeightsNormalAligned.buffer);
    geometricQueries->setResources(cursor["geometricQueries"], printLogs);
    cdfTable.setResources(cursor["cdfTable"], printLogs);
    cdfTableNormalAligned.setResources(cursor["cdfTableNormalAligned"], printLogs);
    cursor["boundarySamplingMass"].setData(boundarySamplingMass);
    cursor["boundarySamplingMassNormalAligned"].setData(boundarySamplingMassNormalAligned);
    if (printLogs) printReflectionInfo(cursor, 10, getReflectionType());
}

std::string GPUUniformTriangleBoundarySampler::getReflectionType() const
{
    std::string arguments = geometricQueries->getReflectionType();
    return "UniformTriangleBoundarySampler<" + arguments + ">";
}

GPUBoundarySamplerType GPUUniformTriangleBoundarySampler::getSamplerType() const
{
    return GPUBoundarySamplerType::UniformTriangle;
}

float GPUUniformTriangleBoundarySampler::getNormalOffsetForBoundary() const
{
    return normalOffsetForBoundary;
}

int GPUUniformTriangleBoundarySampler::getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples) const
{
    return sampler.getSampleCount(nTotalSamples, boundaryNormalAlignedSamples);
}

std::shared_ptr<GPUBoundarySampler> createUniformBoundarySampler(std::shared_ptr<GPUGeometricQueries<2>> geometricQueries,
                                                                 const std::vector<Vector2>& positions,
                                                                 const std::vector<Vector2i>& indices,
                                                                 std::function<bool(const Vector2&)> insideSolveRegion,
                                                                 float normalOffsetForBoundary, bool solveDoubleSided,
                                                                 bool computeWeightedNormals)
{
    std::vector<float> constantPrimitiveWeights(indices.size(), 1.0f);
    return std::make_shared<GPUUniformLineSegmentBoundarySampler>(
        geometricQueries, positions, indices, constantPrimitiveWeights,
        constantPrimitiveWeights, insideSolveRegion, normalOffsetForBoundary,
        solveDoubleSided, computeWeightedNormals);
}

std::shared_ptr<GPUBoundarySampler> createUniformBoundarySampler(std::shared_ptr<GPUGeometricQueries<2>> geometricQueries,
                                                                 const std::vector<Vector2>& positions,
                                                                 const std::vector<Vector2i>& indices,
                                                                 const std::vector<float>& primitiveWeights,
                                                                 std::function<bool(const Vector2&)> insideSolveRegion,
                                                                 float normalOffsetForBoundary, bool solveDoubleSided,
                                                                 bool computeWeightedNormals)
{
    return std::make_shared<GPUUniformLineSegmentBoundarySampler>(
        geometricQueries, positions, indices, primitiveWeights,
        primitiveWeights, insideSolveRegion, normalOffsetForBoundary,
        solveDoubleSided, computeWeightedNormals);
}

std::shared_ptr<GPUBoundarySampler> createUniformBoundarySampler(std::shared_ptr<GPUGeometricQueries<2>> geometricQueries,
                                                                 const std::vector<Vector2>& positions,
                                                                 const std::vector<Vector2i>& indices,
                                                                 const std::vector<float>& primitiveWeights,
                                                                 const std::vector<float>& primitiveWeightsNormalAligned,
                                                                 std::function<bool(const Vector2&)> insideSolveRegion,
                                                                 float normalOffsetForBoundary, bool solveDoubleSided,
                                                                 bool computeWeightedNormals)
{
    return std::make_shared<GPUUniformLineSegmentBoundarySampler>(
        geometricQueries, positions, indices, primitiveWeights,
        primitiveWeightsNormalAligned, insideSolveRegion,
        normalOffsetForBoundary, solveDoubleSided, computeWeightedNormals);
}

std::shared_ptr<GPUBoundarySampler> createUniformBoundarySampler(std::shared_ptr<GPUGeometricQueries<3>> geometricQueries,
                                                                 const std::vector<Vector3>& positions,
                                                                 const std::vector<Vector3i>& indices,
                                                                 std::function<bool(const Vector3&)> insideSolveRegion,
                                                                 float normalOffsetForBoundary, bool solveDoubleSided,
                                                                 bool computeWeightedNormals)
{
    std::vector<float> constantPrimitiveWeights(indices.size(), 1.0f);
    return std::make_shared<GPUUniformTriangleBoundarySampler>(
        geometricQueries, positions, indices, constantPrimitiveWeights,
        constantPrimitiveWeights, insideSolveRegion, normalOffsetForBoundary,
        solveDoubleSided, computeWeightedNormals);
}

std::shared_ptr<GPUBoundarySampler> createUniformBoundarySampler(std::shared_ptr<GPUGeometricQueries<3>> geometricQueries,
                                                                 const std::vector<Vector3>& positions,
                                                                 const std::vector<Vector3i>& indices,
                                                                 const std::vector<float>& primitiveWeights,
                                                                 std::function<bool(const Vector3&)> insideSolveRegion,
                                                                 float normalOffsetForBoundary, bool solveDoubleSided,
                                                                 bool computeWeightedNormals)
{
    return std::make_shared<GPUUniformTriangleBoundarySampler>(
        geometricQueries, positions, indices, primitiveWeights,
        primitiveWeights, insideSolveRegion, normalOffsetForBoundary,
        solveDoubleSided, computeWeightedNormals);
}

std::shared_ptr<GPUBoundarySampler> createUniformBoundarySampler(std::shared_ptr<GPUGeometricQueries<3>> geometricQueries,
                                                                 const std::vector<Vector3>& positions,
                                                                 const std::vector<Vector3i>& indices,
                                                                 const std::vector<float>& primitiveWeights,
                                                                 const std::vector<float>& primitiveWeightsNormalAligned,
                                                                 std::function<bool(const Vector3&)> insideSolveRegion,
                                                                 float normalOffsetForBoundary, bool solveDoubleSided,
                                                                 bool computeWeightedNormals)
{
    return std::make_shared<GPUUniformTriangleBoundarySampler>(
        geometricQueries, positions, indices, primitiveWeights,
        primitiveWeightsNormalAligned, insideSolveRegion,
        normalOffsetForBoundary, solveDoubleSided, computeWeightedNormals);
}

template <size_t DIM>
GPUGenerateDomainSamples<DIM>::GPUGenerateDomainSamples(GPUBuffer& rngs_):
rngs(rngs_)
{
    // do nothing
}

template <size_t DIM>
void GPUGenerateDomainSamples<DIM>::allocate(GPUContext& context, uint32_t nSamplePoints_)
{
    nSamplePoints = nSamplePoints_;
    std::vector<GPUSamplePoint<DIM>> samplePointsData(nSamplePoints);
    samplePoints.allocate<GPUSamplePoint<DIM>>(context, true, samplePointsData);
}

template <size_t DIM>
void GPUGenerateDomainSamples<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("rngs").setBinding(rngs.buffer);
    cursor.getPath("samplePoints").setBinding(samplePoints.buffer);
    cursor.getPath("nSamplePoints").setData(nSamplePoints);
    if (printLogs) printReflectionInfo(cursor, 4, "generateDomainSamples");
}

template <size_t DIM>
void GPUGenerateDomainSamples<DIM>::read(GPUContext& context,
                                         std::vector<GPUSamplePoint<DIM>>& samplePointsData) const
{
    samplePoints.read<GPUSamplePoint<DIM>>(context, samplePointsData);
}

template <size_t DIM>
uint32_t GPUGenerateDomainSamples<DIM>::getBufferSize() const
{
    return nSamplePoints;
}

template <size_t DIM>
GPUBuffer& GPUGenerateDomainSamples<DIM>::getBuffer()
{
    return samplePoints;
}

template <size_t DIM>
GPUGenerateBoundarySamples<DIM>::GPUGenerateBoundarySamples(GPUBuffer& rngs_,
                                                            SampleType sampleType_,
                                                            bool generateBoundaryNormalAligned_):
rngs(rngs_),
sampleType(sampleType_),
generateBoundaryNormalAligned(generateBoundaryNormalAligned_ ? 1 : 0)
{
    // do nothing
}

template <size_t DIM>
void GPUGenerateBoundarySamples<DIM>::allocate(GPUContext& context, uint32_t nSamplePoints_)
{
    nSamplePoints = nSamplePoints_;
    std::vector<GPUSamplePoint<DIM>> samplePointsData(nSamplePoints);
    samplePoints.allocate<GPUSamplePoint<DIM>>(context, true, samplePointsData);
}

template <size_t DIM>
void GPUGenerateBoundarySamples<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("rngs").setBinding(rngs.buffer);
    cursor.getPath("samplePoints").setBinding(samplePoints.buffer);
    cursor.getPath("normalOffsetForBoundary").setData(normalOffsetForBoundary);
    cursor.getPath("generateBoundaryNormalAligned").setData(generateBoundaryNormalAligned);
    cursor.getPath("nSamplePoints").setData(nSamplePoints);
    std::string name = sampleType == SampleType::OnAbsorbingBoundary ?
                       "generateAbsorbingBoundarySamples" :
                       "generateReflectingBoundarySamples";
    if (printLogs) printReflectionInfo(cursor, 6, name);
}

template <size_t DIM>
void GPUGenerateBoundarySamples<DIM>::read(GPUContext& context,
                                           std::vector<GPUSamplePoint<DIM>>& samplePointsData) const
{
    samplePoints.read<GPUSamplePoint<DIM>>(context, samplePointsData);
}

template <size_t DIM>
void GPUGenerateBoundarySamples<DIM>::setNormalOffsetForBoundary(float offset)
{
    normalOffsetForBoundary = offset;
}

template <size_t DIM>
uint32_t GPUGenerateBoundarySamples<DIM>::getBufferSize() const
{
    return nSamplePoints;
}

template <size_t DIM>
GPUBuffer& GPUGenerateBoundarySamples<DIM>::getBuffer()
{
    return samplePoints;
}

} // wosx
