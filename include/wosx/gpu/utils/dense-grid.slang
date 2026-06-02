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

// This file implements a dense grid data structure for storing and querying data.

implementing wosx;

public interface IDenseGrid<T, let DIM : int>
    where T : IFloat
{
    // returns data at the given point
    T operator()(vector<float, DIM> x);
};

public struct FloatDenseGrid2D : IDenseGrid<float, 2>
{
    private Texture2D<float> data;
    private SamplerState sampler;
    private float2 origin;
    private float2 extent;

    // returns data at the given point
    public float operator()(float2 x)
    {
        float2 localCoords = (x - origin) / extent;
        return data.SampleLevel(sampler, localCoords.yx, 0);
    }
};

public struct FloatDenseGrid3D : IDenseGrid<float, 3>
{
    private Texture3D<float> data;
    private SamplerState sampler;
    private float3 origin;
    private float3 extent;

    // returns data at the given point
    public float operator()(float3 x)
    {
        float3 localCoords = (x - origin) / extent;
        return data.SampleLevel(sampler, localCoords.zyx, 0);
    }
};

public struct Float4DenseGrid2D : IDenseGrid<float4, 2>
{
    private Texture2D<float4> data;
    private SamplerState sampler;
    private float2 origin;
    private float2 extent;

    // returns data at the given point
    public float4 operator()(float2 x)
    {
        float2 localCoords = (x - origin) / extent;
        return data.SampleLevel(sampler, localCoords.yx, 0);
    }
};

public struct Float4DenseGrid3D : IDenseGrid<float4, 3>
{
    private Texture3D<float4> data;
    private SamplerState sampler;
    private float3 origin;
    private float3 extent;

    // returns data at the given point
    public float4 operator()(float3 x)
    {
        float3 localCoords = (x - origin) / extent;
        return data.SampleLevel(sampler, localCoords.zyx, 0);
    }
};
