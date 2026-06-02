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

// This file defines a boundary sampler for generating uniformly distributed sample points
// on a 2D or 3D boundary mesh defined by a set of vertices and indices. These sample points
// are required by the Boundary Value Caching (BVC) algorithm for reducing variance of the
// walk-on-spheres and walk-on-stars estimators. BVC currently requires sample points on the
// absorbing boundary to be displaced slightly along the boundary normal.

implementing wosx;
__include point_estimation.common;

public interface IBoundarySampler<let DIM : int>
{
    // generates a sample point on the boundary
    void generateSample(SampleType sampleType,
                        float normalOffsetForBoundary,
                        uint generateBoundaryNormalAligned,
                        uint sampleIndex,
                        uint nTotalSamples,
                        inout pcg32 rng,
                        inout SamplePoint<DIM> samplePt);
};

public struct EmptyBoundarySampler<let DIM : int> : IBoundarySampler<DIM>
{
    // generates a sample point on the boundary
    public void generateSample(SampleType sampleType,
                               float normalOffsetForBoundary,
                               uint generateBoundaryNormalAligned,
                               uint sampleIndex,
                               uint nTotalSamples,
                               inout pcg32 rng,
                               inout SamplePoint<DIM> samplePt)
    {
        // do nothing
    }
};

public interface ILineSegmentSampler
{
    // samples point on line segment
    static float samplePoint(float2 pa, float2 pb, inout pcg32 rng,
                             out float2 pt, out float2 n);
};

public struct UniformLineSegmentSampler : ILineSegmentSampler
{
    // samples point on line segment
    public static float samplePoint(float2 pa, float2 pb, inout pcg32 rng,
                                    out float2 pt, out float2 n)
    {
        float u = rng.nextFloat();
        float2 s = pb - pa;
        pt = pa + u * s;
        n = float2(s.y, -s.x);
        float norm = length(n);
        n /= norm;

        return 1.0 / norm;
    }
};

public struct UniformLineSegmentBoundarySampler<GeometricQueries> : IBoundarySampler<2>
    where GeometricQueries : IGeometricQueries<2>
{
    private StructuredBuffer<float2> positions;
    private StructuredBuffer<float2> normals;
    private StructuredBuffer<uint2> indices;
    private StructuredBuffer<float> primitiveWeights;
    private StructuredBuffer<float> primitiveWeightsNormalAligned;
    private GeometricQueries geometricQueries;
    private CDFTable cdfTable;
    private CDFTable cdfTableNormalAligned;
    private float boundarySamplingMass;
    private float boundarySamplingMassNormalAligned;

    // generates a uniformly distributed sample point on the boundary
    public void generateSample(SampleType sampleType,
                               float normalOffsetForBoundary,
                               float pdf,
                               uint primitiveIndex,
                               inout pcg32 rng,
                               inout SamplePoint<2> samplePt)
    {
        uint2 index = indices[primitiveIndex];
        float2 p0 = positions[index[0]] + normalOffsetForBoundary * normals[index[0]];
        float2 p1 = positions[index[1]] + normalOffsetForBoundary * normals[index[1]];
        UniformLineSegmentSampler.samplePoint(p0, p1, rng, samplePt.pt, samplePt.normal);
        samplePt.type = sampleType;
        samplePt.estimationQuantity = EstimationQuantity::Solution;
        samplePt.pdf = pdf;
        samplePt.distToAbsorbingBoundary = geometricQueries.computeDistToAbsorbingBoundary(samplePt.pt, false);
        samplePt.distToReflectingBoundary = geometricQueries.computeDistToReflectingBoundary(samplePt.pt, false);
    }

    // generates a uniformly distributed sample point on the boundary
    public void generateSample(SampleType sampleType,
                               float normalOffsetForBoundary,
                               uint generateBoundaryNormalAligned,
                               uint sampleIndex,
                               uint nTotalSamples,
                               inout pcg32 rng,
                               inout SamplePoint<2> samplePt)
    {
        float pdf = 1.0f;
        uint primitiveIndex = UINT_MAX;
        float u = (sampleIndex + rng.nextFloat()) / nTotalSamples;
        if (generateBoundaryNormalAligned != 0)
        {
            primitiveIndex = cdfTableNormalAligned.sample(u);
            pdf = primitiveWeightsNormalAligned[primitiveIndex] / boundarySamplingMassNormalAligned;
            samplePt.estimateBoundaryNormalAligned = 1;
        }
        else
        {
            primitiveIndex = cdfTable.sample(u);
            pdf = primitiveWeights[primitiveIndex] / boundarySamplingMass;
            normalOffsetForBoundary = -1.0 * normalOffsetForBoundary;
            samplePt.estimateBoundaryNormalAligned = 0;
        }

        generateSample(sampleType, normalOffsetForBoundary, pdf, primitiveIndex, rng, samplePt);
    }
};

public interface ITriangleSampler
{
    // samples point on triangle
    static float samplePoint(float3 pa, float3 pb, float3 pc,
                             inout pcg32 rng, out float3 pt,
                             out float3 n);
}

public struct UniformTriangleSampler : ITriangleSampler
{
    // samples point on triangle
    public static float samplePoint(float3 pa, float3 pb, float3 pc,
                                    inout pcg32 rng, out float3 pt,
                                    out float3 n)
    {
        float u1 = sqrt(rng.nextFloat());
        float u2 = rng.nextFloat();
        float a = 1.0 - u1;
        float b = u2 * u1;
        float c = 1.0 - a - b;
        pt = pa * a + pb * b + pc * c;
        n = cross(pb - pa, pc - pa);
        float norm = length(n);
        n /= norm;

        return 2.0 / norm;
    }
};

public struct UniformTriangleBoundarySampler<GeometricQueries> : IBoundarySampler<3>
    where GeometricQueries : IGeometricQueries<3>
{
    private StructuredBuffer<float3> positions;
    private StructuredBuffer<float3> normals;
    private StructuredBuffer<uint3> indices;
    private StructuredBuffer<float> primitiveWeights;
    private StructuredBuffer<float> primitiveWeightsNormalAligned;
    private GeometricQueries geometricQueries;
    private CDFTable cdfTable;
    private CDFTable cdfTableNormalAligned;
    private float boundarySamplingMass;
    private float boundarySamplingMassNormalAligned;

    // generates a uniformly distributed sample point on the boundary
    public void generateSample(SampleType sampleType,
                               float normalOffsetForBoundary,
                               float pdf,
                               uint primitiveIndex,
                               inout pcg32 rng,
                               inout SamplePoint<3> samplePt)
    {
        uint3 index = indices[primitiveIndex];
        float3 p0 = positions[index[0]] + normalOffsetForBoundary * normals[index[0]];
        float3 p1 = positions[index[1]] + normalOffsetForBoundary * normals[index[1]];
        float3 p2 = positions[index[2]] + normalOffsetForBoundary * normals[index[2]];
        UniformTriangleSampler.samplePoint(p0, p1, p2, rng, samplePt.pt, samplePt.normal);
        samplePt.type = sampleType;
        samplePt.estimationQuantity = EstimationQuantity::Solution;
        samplePt.pdf = pdf;
        samplePt.distToAbsorbingBoundary = geometricQueries.computeDistToAbsorbingBoundary(samplePt.pt, false);
        samplePt.distToReflectingBoundary = geometricQueries.computeDistToReflectingBoundary(samplePt.pt, false);
    }

    // generates a uniformly distributed sample point on the boundary
    public void generateSample(SampleType sampleType,
                               float normalOffsetForBoundary,
                               uint generateBoundaryNormalAligned,
                               uint sampleIndex,
                               uint nTotalSamples,
                               inout pcg32 rng,
                               inout SamplePoint<3> samplePt)
    {
        float pdf = 1.0f;
        uint primitiveIndex = UINT_MAX;
        float u = (sampleIndex + rng.nextFloat()) / nTotalSamples;
        if (generateBoundaryNormalAligned != 0)
        {
            primitiveIndex = cdfTableNormalAligned.sample(u);
            pdf = primitiveWeightsNormalAligned[primitiveIndex] / boundarySamplingMassNormalAligned;
            samplePt.estimateBoundaryNormalAligned = 1;
        }
        else
        {
            primitiveIndex = cdfTable.sample(u);
            pdf = primitiveWeights[primitiveIndex] / boundarySamplingMass;
            normalOffsetForBoundary = -1.0 * normalOffsetForBoundary;
            samplePt.estimateBoundaryNormalAligned = 0;
        }

        generateSample(sampleType, normalOffsetForBoundary, pdf, primitiveIndex, rng, samplePt);
    }
};
