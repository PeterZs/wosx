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

import wosx;

static const float FLT_MAX = 3.402823466e+38F; // max float value
uniform ParameterBlock<GeometricQueriesType> gGeometricQueries;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeDistToBoundary(uint3 threadId: SV_DispatchThreadID,
                           uniform StructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> points,
                           uniform RWStructuredBuffer<float> distToAbsorbingBoundary,
                           uniform RWStructuredBuffer<float> distToReflectingBoundary,
                           uniform uint nPoints)
{
    uint index = threadId.x;
    if (index >= nPoints)
    {
        return;
    }

    vector<float, WOSX_PROBLEM_DIMENSION> pt = points[index];
    distToAbsorbingBoundary[index] = gGeometricQueries.computeDistToAbsorbingBoundary(pt, true);
    distToReflectingBoundary[index] = gGeometricQueries.computeDistToReflectingBoundary(pt, true);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void intersectBoundary(uint3 threadId: SV_DispatchThreadID,
                       uniform StructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> origins,
                       uniform StructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> directions,
                       uniform RWStructuredBuffer<float> distAlongRay,
                       uniform RWStructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> intersectionPoints,
                       uniform RWStructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> intersectionNormals,
                       uniform uint nRays)
{
    uint index = threadId.x;
    if (index >= nRays)
    {
        return;
    }

    vector<float, WOSX_PROBLEM_DIMENSION> normal = vector<float, WOSX_PROBLEM_DIMENSION>(0.0);
    IntersectionPoint<WOSX_PROBLEM_DIMENSION> intersectionPt;
    bool success = gGeometricQueries.intersectBoundary(origins[index], normal,
                                                       directions[index], FLT_MAX,
                                                       false, false, intersectionPt);
    if (success)
    {
        distAlongRay[index] = intersectionPt.dist;
        intersectionPoints[index] = intersectionPt.pt;
        intersectionNormals[index] = intersectionPt.normal;
    }
    else
    {
        distAlongRay[index] = -1.0;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void intersectAbsorbingBoundary(uint3 threadId: SV_DispatchThreadID,
                                uniform StructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> origins,
                                uniform StructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> directions,
                                uniform RWStructuredBuffer<float> distAlongRay,
                                uniform RWStructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> intersectionPoints,
                                uniform RWStructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> intersectionNormals,
                                uniform uint nRays)
{
    uint index = threadId.x;
    if (index >= nRays)
    {
        return;
    }

    vector<float, WOSX_PROBLEM_DIMENSION> normal = vector<float, WOSX_PROBLEM_DIMENSION>(0.0);
    IntersectionPoint<WOSX_PROBLEM_DIMENSION> intersectionPt;
    bool success = gGeometricQueries.intersectAbsorbingBoundary(origins[index], normal,
                                                                directions[index], FLT_MAX,
                                                                false, intersectionPt);
    if (success)
    {
        distAlongRay[index] = intersectionPt.dist;
        intersectionPoints[index] = intersectionPt.pt;
        intersectionNormals[index] = intersectionPt.normal;
    }
    else
    {
        distAlongRay[index] = -1.0;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void intersectReflectingBoundary(uint3 threadId: SV_DispatchThreadID,
                                 uniform StructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> origins,
                                 uniform StructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> directions,
                                 uniform RWStructuredBuffer<float> distAlongRay,
                                 uniform RWStructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> intersectionPoints,
                                 uniform RWStructuredBuffer<vector<float, WOSX_PROBLEM_DIMENSION>> intersectionNormals,
                                 uniform uint nRays)
{
    uint index = threadId.x;
    if (index >= nRays)
    {
        return;
    }

    vector<float, WOSX_PROBLEM_DIMENSION> normal = vector<float, WOSX_PROBLEM_DIMENSION>(0.0);
    IntersectionPoint<WOSX_PROBLEM_DIMENSION> intersectionPt;
    bool success = gGeometricQueries.intersectReflectingBoundary(origins[index], normal,
                                                                 directions[index], FLT_MAX,
                                                                 false, intersectionPt);
    if (success)
    {
        distAlongRay[index] = intersectionPt.dist;
        intersectionPoints[index] = intersectionPt.pt;
        intersectionNormals[index] = intersectionPt.normal;
    }
    else
    {
        distAlongRay[index] = -1.0;
    }
}
