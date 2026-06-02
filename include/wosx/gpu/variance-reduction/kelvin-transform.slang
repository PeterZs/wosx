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

// This file enables solving Poisson equations with Dirichlet, Neumann, and/or
// Robin boundary conditions in an exterior domain using the Kelvin transform.
// This transformation maps the exterior of a watertight domain to the interior
// of an inverted domain, ensuring that random walks terminate on the boundary
// of the inverted domain, rather than escaping to infinity in the exterior domain.
//
// The KelvinTransform class provides methods for performing the inversion, and
// setting up the PDE on the inverted domain. The user can then run any solver
// on the inverted domain, and map the solution and gradient estimates back
// to the exterior domain via the transform*Estimate methods.
//
// Resources:
// - Kelvin Transformations for Simulations on Infinite Domains [2021]

implementing wosx;
__include core.pde;

public struct KelvinTransform<T, let DIM : int>
    where T : IFloat
{
    private vector<float, DIM> origin;

    // constructor
    public __init()
    {
        origin = vector<float, DIM>(0.0);
    }

    // set and get origin
    [mutating]
    public void setOrigin(vector<float, DIM> origin_)
    {
        origin = origin_;
    }
    public vector<float, DIM> getOrigin()
    {
        return origin;
    }

    // applies the Kelvin transform to a point in the exterior domain
    public vector<float, DIM> transformPoint(vector<float, DIM> x)
    {
        vector<float, DIM> xShifted = x - origin;
        return xShifted / dot(xShifted, xShifted);
    }

    // applies the Kelvin transform to a normal vector at a point in the exterior domain
    public vector<float, DIM> transformNormal(vector<float, DIM> x,
                                              vector<float, DIM> n)
    {
        vector<float, DIM> xShifted = x - origin;
        return n - 2.0 * dot(n, xShifted) * xShifted / dot(xShifted, xShifted);
    }

    // returns the estimated solution in the exterior domain, given the
    // solution estimate at a transformed point
    public T transformSolutionEstimate(T V, vector<float, DIM> y)
    {
        float rPow = DIM == 3 ? length(y) : 1.0;
        return V.scale(rPow);
    }

    // returns the estimated gradient in the exterior domain, given the
    // solution and gradient estimates at a transformed point
    public void transformGradientEstimate(T V, T dV[DIM],
                                          vector<float, DIM> y,
                                          out T gradient[DIM])
    {
        T dU[DIM];
        float r2 = dot(y, y);
        float rPow = 1.0;
        if (DIM == 3)
        {
            rPow *= sqrt(r2);
        }

        for (int i = 0; i < DIM; i++)
        {
            dU[i] = dV[i].scale(rPow * r2);
            if (DIM == 3)
            {
                dU[i] = dU[i] + V.scale(rPow * y[i]);
            }
        }

        for (int i = 0; i < DIM; i++)
        {
            gradient[i] = T(0.0);
            for (int j = 0; j < DIM; j++)
            {
                float R = -2.0 * y[i] * y[j] / r2;
                if (i == j)
                {
                    R += 1.0;
                }

                gradient[i] = gradient[i] + dU[j].scale(R);
            }
        }
    }
};

public struct KelvinPDE<T, PDE, let DIM : int> : IPDE<T, DIM>
    where T : IFloat
    where PDE : IPDE<T, DIM>
{
    private PDE pdeExteriorDomain;
    private KelvinTransform<T, DIM> kelvinTransform;
    public typedef PDE::GreensFnFreeSpace GreensFnFreeSpace;
    public typedef PDE::GreensFnBall GreensFnBall;

    // returns source term
    public T source(vector<float, DIM> y)
    {
        float r2 = dot(y, y);
        vector<float, DIM> x = y / r2 + kelvinTransform.getOrigin();
        float rPow = r2 * r2;
        if (DIM == 3)
        {
            rPow *= sqrt(r2);
        }

        return pdeExteriorDomain.source(x).scale(1.0 / rPow);
    }

    // returns Dirichlet boundary conditions
    public T dirichlet(vector<float, DIM> y, bool returnBoundaryNormalAlignedValue = false)
    {
        float r2 = dot(y, y);
        vector<float, DIM> x = y / r2 + kelvinTransform.getOrigin();
        float rPow = 1.0;
        if (DIM == 3)
        {
            rPow *= sqrt(r2);
        }

        return pdeExteriorDomain.dirichlet(x, false).scale(1.0 / rPow);
    }

    // returns Robin boundary conditions (dual purposes for Neumann conditions when Robin coeff is zero)
    public T robin(vector<float, DIM> y, vector<float, DIM> N,
                   bool returnBoundaryNormalAlignedValue = false)
    {
        float r2 = dot(y, y);
        vector<float, DIM> x = y / r2 + kelvinTransform.getOrigin();
        vector<float, DIM> n = N - 2.0 * dot(N, y) * y / r2;
        float rPow = r2;
        if (DIM == 3)
        {
            rPow *= sqrt(r2);
        }

        return pdeExteriorDomain.robin(x, n, false).scale(1.0 / rPow);
    }

    // returns Robin coefficient
    public float robinCoeff(vector<float, DIM> y, vector<float, DIM> N,
                            bool returnBoundaryNormalAlignedValue = false)
    {
        float r2 = dot(y, y);
        vector<float, DIM> x = y / r2 + kelvinTransform.getOrigin();
        float P = dot(N, y) / r2;
        float shift = DIM == 3 ? P : 0.0;
        vector<float, DIM> n = N - 2.0 * P * y;

        return pdeExteriorDomain.robinCoeff(x, n, false) / r2 + shift;
    }

    // returns absorption coefficient (must be positive or equal to zero)
    public float absorptionCoeff()
    {
        return 0.0;
    }

    // checks if Robin coefficients are non-zero anywhere on the reflecting boundary
    public bool areRobinConditionsPureNeumann()
    {
        return DIM == 2 ? pdeExteriorDomain.areRobinConditionsPureNeumann() : false;
    }

    // checks if Robin coefficients are non-negative everywhere on the reflecting boundary
    public bool areRobinCoeffsNonnegative()
    {
        return DIM == 2 ? pdeExteriorDomain.areRobinCoeffsNonnegative() : false;
    }
};
