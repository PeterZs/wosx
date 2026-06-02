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
void resetSampleStatistics(uint3 threadId: SV_DispatchThreadID,
                           uniform RWStructuredBuffer<SampleStatistics<DataType, WOSX_PROBLEM_DIMENSION>> sampleStatistics,
                           uniform uint nSamplePoints)
{
    uint index = threadId.x;
    if (index >= nSamplePoints)
    {
        return;
    }

    sampleStatistics[index].reset();
}

[shader("compute")]
[numthreads(256, 1, 1)]
void getSampleStatistics(uint3 threadId: SV_DispatchThreadID,
                         uniform StructuredBuffer<SampleStatistics<DataType, WOSX_PROBLEM_DIMENSION>> sampleStatistics,
                         uniform RWStructuredBuffer<SampleStatistics<DataType, WOSX_PROBLEM_DIMENSION>> outputSampleStatistics,
                         uniform uint nSamplePoints,
                         uniform uint nOutputSamplePoints)
{
    uint index = threadId.x;
    if (index >= nOutputSamplePoints)
    {
        return;
    }

    uint nSampleCopies = nSamplePoints / nOutputSamplePoints;
    outputSampleStatistics[index] = sampleStatistics[nSampleCopies * index];
    for (uint i = 1; i < nSampleCopies; i++)
    {
        outputSampleStatistics[index].merge(sampleStatistics[nSampleCopies * index + i]);
    }
}
