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

// This file provides various utility functions, used internally by the algorithms
// implemented in WoSX, for generating random samples.

implementing wosx;
__include deps.pcg32;
__include core.math_helpers;

public interface ISphereSampler<let DIM : int>
{
    // samples a direction on the unit sphere
    static vector<float, DIM> sampleUnitSphereUniform(inout pcg32 rng);

    // samples a point inside the unit ball
    static vector<float, DIM> sampleUnitBallUniform(inout pcg32 rng);

    // samples a direction on the unit hemisphere using a cosine-weighted distribution
    static vector<float, DIM> sampleUnitHemisphereCosine(inout pcg32 rng);

    // computes the PDF of sampling a direction on a sphere
    static float pdfSampleSphereUniform(float r);

    // computes the PDF of sampling a point inside a ball
    static float pdfSampleBallUniform(float r);

    // computes the PDF of sampling a direction on a hemisphere using a cosine-weighted distribution
    static float pdfSampleUnitHemisphereCosine(vector<float, DIM> d);

    // transforms a vector d to a coordinate system aligned with the direction n
    static void transformCoordinates(vector<float, DIM> n, inout vector<float, DIM> d);
};

public struct SphereSampler2D : ISphereSampler<2>
{
    // samples a direction on the unit sphere
    public static float2 sampleUnitSphereUniform(inout pcg32 rng)
    {
        float u = rng.nextFloat();
        float phi = 2.0 * M_PI * u;

        return float2(cos(phi), sin(phi));
    }

    // samples a point inside the unit ball
    public static float2 sampleUnitBallUniform(inout pcg32 rng)
    {
        float u = rng.nextFloat();
        float r = sqrt(u);

        return r * sampleUnitSphereUniform(rng);
    }

    // samples a direction on the unit hemisphere using a cosine-weighted distribution
    public static float2 sampleUnitHemisphereCosine(inout pcg32 rng)
    {
        float u = 2.0 * rng.nextFloat() - 1.0;
        float z = sqrt(max(0.0, 1.0 - u * u));

        return float2(u, z);
    }

    // computes the PDF of sampling a direction on a sphere
    public static float pdfSampleSphereUniform(float r)
    {
        return 1.0 / (2.0 * M_PI * r);
    }

    // computes the PDF of sampling a point inside a ball
    public static float pdfSampleBallUniform(float r)
    {
        return 1.0 / (M_PI * r * r);
    }

    // computes the PDF of sampling a direction on a hemisphere using a cosine-weighted distribution
    public static float pdfSampleUnitHemisphereCosine(float2 d)
    {
        return abs(d.y) / 2.0;
    }

    // transforms a vector d to a coordinate system aligned with the direction n
    public static void transformCoordinates(float2 n, inout float2 d)
    {
        // compute orthonormal basis
        float2 s = float2(n.y, -n.x);

        // transform
        d = d.x * s + d.y * n;
    }
};

public struct SphereSampler3D : ISphereSampler<3>
{
    // samples a direction on the unit sphere
    public static float3 sampleUnitSphereUniform(inout pcg32 rng)
    {
        float u1 = rng.nextFloat();
        float u2 = rng.nextFloat();
        float z = 1.0 - 2.0 * u1;
        float r = sqrt(max(0.0, 1.0 - z * z));
        float phi = 2.0 * M_PI * u2;

        return float3(r * cos(phi), r * sin(phi), z);
    }

    // samples a point inside the unit ball
    public static float3 sampleUnitBallUniform(inout pcg32 rng)
    {
        float u = rng.nextFloat();
        float r = pow(u, 1.0 / 3.0);

        return r * sampleUnitSphereUniform(rng);
    }

    private static float2 sampleUnitDiskConcentric(inout pcg32 rng)
    {
        // map uniform random numbers to [-1,1]^2
        float u1 = 2.0 * rng.nextFloat() - 1.0;
        float u2 = 2.0 * rng.nextFloat() - 1.0;

        // handle degeneracy at the origin
        if (u1 == 0.0 && u2 == 0.0)
        {
            return float2(0.0, 0.0);
        }

        // apply concentric mapping to point
        float theta, r;
        if (abs(u1) > abs(u2))
        {
            r = u1;
            theta = 0.25 * M_PI * (u2 / u1);
        }
        else
        {
            r = u2;
            theta = 0.5 * M_PI * (1.0 - 0.5 * (u1 / u2));
        }

        return r * float2(cos(theta), sin(theta));
    }

    // samples a direction on the unit hemisphere using a cosine-weighted distribution
    public static float3 sampleUnitHemisphereCosine(inout pcg32 rng)
    {
        float2 d = sampleUnitDiskConcentric(rng);
        float z = sqrt(max(0.0, 1.0 - dot(d, d)));

        return float3(d.x, d.y, z);
    }

    // computes the PDF of sampling a direction on a sphere
    public static float pdfSampleSphereUniform(float r)
    {
        return 1.0 / (4.0 * M_PI * r * r);
    }

    // computes the PDF of sampling a point inside a ball
    public static float pdfSampleBallUniform(float r)
    {
        return 3.0 / (4.0 * M_PI * r * r * r);
    }

    // computes the PDF of sampling a direction on a hemisphere using a cosine-weighted distribution
    public static float pdfSampleUnitHemisphereCosine(float3 d)
    {
        return abs(d.z) / M_PI;
    }

    // transforms a vector d to a coordinate system aligned with the direction n
    public static void transformCoordinates(float3 n, inout float3 d)
    {
        // compute orthonormal basis; source: https://graphics.pixar.com/library/OrthonormalB/paper.pdf
        int sign = n.z >= 0.0 ? 1 : -1;
        const float a = -1.0 / (sign + n.z);
        const float b = n.x * n.y * a;
        float3 b1 = float3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        float3 b2 = float3(b, sign + n.y * n.y * a, -n.y);

        // transform
        d = d.x * b1 + d.y * b2 + d.z * n;
    }
};

// source: https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Sampling_Random_Variables
public struct CDFTable
{
    private StructuredBuffer<float> table;
    private uint size;

    // generates sample from table using one uniform sample in the range [0, 1)
    public uint sample(float u)
    {
        uint first = 0;
        uint len = size;

        while (len > 0)
        {
            uint halfLen = len >> 1;
            uint middle = first + halfLen;

            // bisect range based on table value at the middle index
            if (table[middle] <= u)
            {
                first = middle + 1;
                len -= halfLen + 1;
            }
            else
            {
                len = halfLen;
            }
        }

        return clamp(first - 1, 0, size - 2);
    }

    // generates sample from table using one uniform sample in the range [0, 1)
    public uint sample(inout pcg32 rng)
    {
        return sample(rng.nextFloat());
    }
};
