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

// This file defines a scalar-valued PDE for a 3D electrostatics problem that
// conforms to the IPDE interface defined in include/wosx/gpu/core/pde.slang.

module wosx_pde_definition;
import fcpw;
import wosx;

static const float FLT_MAX = 3.402823466e+38F; // max float value
static const uint UINT_MAX = 4294967295;       // max unsigned int value
static const uint MOVABLE_COMB = 1;

public struct ElectrostaticsPDE : IPDE<float, 3>
{
    public typedef HarmonicGreensFnFreeSpace3D GreensFnFreeSpace;
    public typedef HarmonicGreensFnBall3D GreensFnBall;
    private Bvh<BvhNode, Triangle, NoSilhouette> mAccelerationStructure;
    private StructuredBuffer<uint> mIsMovableFace;
    private float mMovableCombVoltage;

    // returns source term
    public float source(float3 x)
    {
        return 0.0;
    }

    // returns Dirichlet boundary conditions
    public float dirichlet(float3 x, bool returnBoundaryNormalAlignedValue = false)
    {
        BoundingSphere sphere;
        sphere.c = x;
        sphere.r2 = FLT_MAX;
        Interaction interaction;
        bool found = mAccelerationStructure.findClosestPoint(sphere, interaction, false);
        if (!found)
        {
            return 0.0;
        }

        return mIsMovableFace[interaction.index] == MOVABLE_COMB ? mMovableCombVoltage : 0.0;
    }

    // returns Robin boundary conditions (dual purposes for Neumann conditions when Robin coeff is zero)
    public float robin(float3 x, float3 n, bool returnBoundaryNormalAlignedValue = false)
    {
        return 0.0;
    }

    // returns Robin coefficient
    public float robinCoeff(float3 x, float3 n, bool returnBoundaryNormalAlignedValue = false)
    {
        return 0.0;
    }

    // returns absorption coefficient (must be positive or equal to zero)
    public float absorptionCoeff()
    {
        return 0.0;
    }

    // checks if Robin coefficients are non-zero anywhere on the reflecting boundary
    public bool areRobinConditionsPureNeumann()
    {
        return true;
    }

    // checks if Robin coefficients are non-negative everywhere on the reflecting boundary
    public bool areRobinCoeffsNonnegative()
    {
        return true;
    }
};

public typedef ElectrostaticsPDE PDEType;
