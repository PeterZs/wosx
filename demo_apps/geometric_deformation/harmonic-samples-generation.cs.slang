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

// For each query point and walk index, the generateHarmonicSamples entrypoint
// runs a walk-on-spheres trajectory from the query point and projects it to the
// cage boundary once the walk is sufficiently close to the boundary. The closest
// cage hit point and triangle id are recorded for each query point and walk index.

import wosx;
import fcpw;
import wosx_pde_definition;

static const float FLT_MAX = 3.402823466e+38F; // max float value
static const uint UINT_MAX = 4294967295;       // max unsigned int value

uniform ParameterBlock<GeometricQueriesType> gGeometricQueries;
uniform ParameterBlock<WalkOnSpheres<float, PDEType, GeometricQueriesType, 3>> gWalkOnSpheres;
uniform ParameterBlock<Bvh<BvhNode, Triangle, NoSilhouette>> gAccelerationStructure;

#include "entrypoints/common-shaders/rng-seeding.cs.slang"
#include "entrypoints/common-shaders/sample-points-utils.cs.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void generateHarmonicSamples(uint3 threadId: SV_DispatchThreadID,
                             uniform StructuredBuffer<SamplePoint<3>> queryPoints,
                             uniform RWStructuredBuffer<pcg32> rngs,
                             uniform RWStructuredBuffer<float3> hitPoints,
                             uniform RWStructuredBuffer<uint> hitTriangleIds,
                             uniform uint nQueryPoints,
                             uniform uint nWalksPerQueryPoint)
{
    uint index = threadId.x;
    if (index >= nQueryPoints * nWalksPerQueryPoint)
    {
        return;
    }

    // run a random walk from the query point
    uint queryIndex = index / nWalksPerQueryPoint;
    SamplePoint<3> queryPoint = queryPoints[queryIndex];
    PDEType::GreensFnBall greensFn;
    WalkState<float, 3> state;
    state.currentPt = queryPoint.pt;
    gWalkOnSpheres.walk(queryPoint.distToAbsorbingBoundary, rngs[index], greensFn, state);

    // project the terminal walk location to the bounding geometry and
    // record the hit point and triangle id
    BoundingSphere sphere;
    sphere.c = state.currentPt;
    sphere.r2 = FLT_MAX;
    Interaction interaction;
    bool found = gAccelerationStructure.findClosestPoint(sphere, interaction, false);
    if (found)
    {
        hitPoints[index] = interaction.p;
        hitTriangleIds[index] = interaction.index;
    }
    else
    {
        hitPoints[index] = float3(0.0);
        hitTriangleIds[index] = UINT_MAX;
    }
}
