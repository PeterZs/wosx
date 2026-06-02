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

// This file defines a PDE with either 1 or 4 channels for a 2D model problem that
// conforms to the IPDE interface defined in include/wosx/gpu/core/pde.slang.

module wosx_pde_definition;
import wosx;

#define WOSX_UNDEFINED_PDE 0
#define WOSX_POISSON_EQUATION 1
#define WOSX_SCREENED_POISSON_EQUATION 2

#ifndef WOSX_PDE_TYPE
#define WOSX_PDE_TYPE WOSX_UNDEFINED_PDE
#endif

#ifndef WOSX_USE_KELVIN_TRANSFORM
#define WOSX_USE_KELVIN_TRANSFORM 0
#endif

public struct ModelProblemPDE : IPDE<DataType, 2>
{
#if WOSX_PDE_TYPE == WOSX_POISSON_EQUATION || WOSX_USE_KELVIN_TRANSFORM == 1
    public typedef HarmonicGreensFnFreeSpace2D GreensFnFreeSpace;
    public typedef HarmonicGreensFnBall2D GreensFnBall;

#elif WOSX_PDE_TYPE == WOSX_SCREENED_POISSON_EQUATION
    public typedef YukawaGreensFnFreeSpace2D GreensFnFreeSpace;
    public typedef YukawaGreensFnBall2D GreensFnBall;
    private float mAbsorptionCoeff;
#endif
    private float mRobinCoeff;
#if WOSX_DATA_CHANNELS == 1
    private FloatDenseGrid2D mSourceValue;
    private FloatDenseGrid2D mAbsorbingBoundaryValue;
    private FloatDenseGrid2D mReflectingBoundaryValue;
    private FloatDenseGrid2D mAbsorbingBoundaryNormalAlignedValue;
    private FloatDenseGrid2D mReflectingBoundaryNormalAlignedValue;
#elif WOSX_DATA_CHANNELS == 4
    private Float4DenseGrid2D mSourceValue;
    private Float4DenseGrid2D mAbsorbingBoundaryValue;
    private Float4DenseGrid2D mReflectingBoundaryValue;
    private Float4DenseGrid2D mAbsorbingBoundaryNormalAlignedValue;
    private Float4DenseGrid2D mReflectingBoundaryNormalAlignedValue;
#endif

    // returns source term
    public DataType source(float2 x)
    {
        return mSourceValue(x);
    }

    // returns Dirichlet boundary conditions
    public DataType dirichlet(float2 x, bool returnBoundaryNormalAlignedValue = false)
    {
        if (returnBoundaryNormalAlignedValue)
        {
            return mAbsorbingBoundaryNormalAlignedValue(x);
        }

        return mAbsorbingBoundaryValue(x);
    }

    // returns Robin boundary conditions (dual purposes for Neumann conditions when Robin coeff is zero)
    public DataType robin(float2 x, float2 n, bool returnBoundaryNormalAlignedValue = false)
    {
        if (returnBoundaryNormalAlignedValue)
        {
            return mReflectingBoundaryNormalAlignedValue(x);
        }

        return mReflectingBoundaryValue(x);
    }

    // returns Robin coefficient
    public float robinCoeff(float2 x, float2 n, bool returnBoundaryNormalAlignedValue = false)
    {
        return mRobinCoeff;
    }

    // returns absorption coefficient (must be positive or equal to zero)
    public float absorptionCoeff()
    {
#if WOSX_PDE_TYPE == WOSX_SCREENED_POISSON_EQUATION && WOSX_USE_KELVIN_TRANSFORM == 0
        return mAbsorptionCoeff;
#else
        return 0.0;
#endif
    }

    // checks if Robin coefficients are non-zero anywhere on the reflecting boundary
    public bool areRobinConditionsPureNeumann()
    {
        return mRobinCoeff == 0.0;
    }

    // checks if Robin coefficients are non-negative everywhere on the reflecting boundary
    public bool areRobinCoeffsNonnegative()
    {
        return mRobinCoeff >= 0.0;
    }
};

// PDEs defined in wosx-pde-definition.slang must be typedefed to "PDEType" to run WoSX's GPU solvers
#if WOSX_USE_KELVIN_TRANSFORM == 1
public typedef KelvinPDE<DataType, ModelProblemPDE, 2> PDEType;

#else
public typedef ModelProblemPDE PDEType;

#endif
