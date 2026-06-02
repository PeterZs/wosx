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

[shader("compute")]
[numthreads(256, 1, 1)]
void runPointEstimator(uint3 threadId: SV_DispatchThreadID,
                       uniform StructuredBuffer<SamplePoint<WOSX_PROBLEM_DIMENSION>> samplePoints,
                       uniform RWStructuredBuffer<pcg32> rngs,
                       uniform RWStructuredBuffer<SampleStatistics<DataType, WOSX_PROBLEM_DIMENSION>> sampleStatistics,
                       uniform uint nInputSamplePoints,
                       uniform uint nSamplePoints,
                       uniform uint nWalks)
{
    uint index = threadId.x;
    if (index >= nSamplePoints)
    {
        return;
    }

    uint nSampleCopies = nSamplePoints / nInputSamplePoints;
    SamplePoint<WOSX_PROBLEM_DIMENSION> samplePt = samplePoints[index / nSampleCopies];
    gPointEstimator.solve(samplePt, nWalks, rngs[index], sampleStatistics[index]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void runPersistentPointEstimator(uint3 threadId: SV_DispatchThreadID,
                                 uniform StructuredBuffer<SamplePoint<WOSX_PROBLEM_DIMENSION>> samplePoints,
                                 uniform RWStructuredBuffer<pcg32> rngs,
                                 uniform RWStructuredBuffer<SampleStatistics<DataType, WOSX_PROBLEM_DIMENSION>> sampleStatistics,
                                 uniform RWStructuredBuffer<uint64_t> globalCounter,
                                 uniform RWStructuredBuffer<uint> locks,
                                 uniform uint64_t nTotalWorkItems,
                                 uniform uint nInputSamplePoints,
                                 uniform uint nPersistentThreads)
{
    uint threadIndex = threadId.x;
    if (threadIndex >= nPersistentThreads)
    {
        return;
    }

    while (true)
    {
        // atomically claim a work item
        uint64_t workIndex;
        InterlockedAdd(globalCounter[0], 1ULL, workIndex);
        if (workIndex >= nTotalWorkItems)
        {
            break;
        }

        // interleaved indexing: consecutive threads map to different samples
        uint sampleIndex = (uint)(workIndex % nInputSamplePoints);
        SamplePoint<WOSX_PROBLEM_DIMENSION> samplePt = samplePoints[sampleIndex];

        // perform a single walk into thread-local statistics
        SampleStatistics<DataType, WOSX_PROBLEM_DIMENSION> localStatistics;
        gPointEstimator.solve(samplePt, 1, rngs[threadIndex], localStatistics);

        // acquire per-sample spinlock
        uint locked;
        do {
            InterlockedCompareExchange(locks[sampleIndex], 0, 1, locked);
        } while (locked != 0);

        // ensure we read the latest sampleStatistics from global memory
        DeviceMemoryBarrier();

        // merge under lock
        sampleStatistics[sampleIndex].merge(localStatistics);

        // ensure merge writes are visible before releasing the lock
        DeviceMemoryBarrier();

        // release spinlock
        locks[sampleIndex] = 0;
    }
}
