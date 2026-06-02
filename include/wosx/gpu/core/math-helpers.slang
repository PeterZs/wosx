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

// This file defines various mathematical constants and functions.

implementing wosx;

internal static const float M_PI = 3.14159265358979323846;               // pi
internal static const float M_PI_2 = 1.57079632679489661923;             // pi/2
internal static const float FLT_EPSILON = 1.192092896e-07F;              // smallest float value such that 1.0+FLT_EPSILON != 1.0
internal static const float FLT_MAX = 3.402823466e+38F;                  // max float value
internal static const uint UINT_MAX = 4294967295;                        // max unsigned int value
internal static const float SQRT_2 = 1.4142135623730950488016887242097F; // sqrt(2)
internal static const float SQRT_3 = 1.7320508075688772935274463415059F; // sqrt(3)

internal float erf(float x)
{
    // constants used in the approximation
    const float a1 = 0.254829592;
    const float a2 = -0.284496736;
    const float a3 = 1.421413741;
    const float a4 = -1.453152027;
    const float a5 = 1.061405429;
    const float p = 0.3275911;

    // save the sign of x
    int sign = (x >= 0.0) ? 1 : -1;
    x = abs(x);

    // Abramowitz and Stegun approximation
    float t = 1.0 / (1.0 + p * x);
    float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return sign * y;
}
