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

// This file defines an interface for Partial Differential Equations (PDEs),
// specifically Poisson and screened Poisson equations, with Dirichlet, Neumann,
// and Robin boundary conditions. As part of the problem setup, users of WoSX
// should populate the callback functions defined by the PDE interface.

implementing wosx;
__include core.distributions;

public interface IPDE<T, let DIM : int>
    where T : IFloat
{
    associatedtype GreensFnFreeSpace : IGreensFnFreeSpace<DIM>;
    associatedtype GreensFnBall : IGreensFnBall<DIM>;

    // returns source term
    T source(vector<float, DIM> x);

    // returns Dirichlet boundary conditions
    T dirichlet(vector<float, DIM> x, bool returnBoundaryNormalAlignedValue = false);

    // returns Robin boundary conditions (dual purposes for Neumann conditions when Robin coeff is zero)
    T robin(vector<float, DIM> x, vector<float, DIM> n, bool returnBoundaryNormalAlignedValue = false);

    // returns Robin coefficient
    float robinCoeff(vector<float, DIM> x, vector<float, DIM> n, bool returnBoundaryNormalAlignedValue = false);

    // returns absorption coefficient (must be positive or equal to zero)
    float absorptionCoeff();

    // checks if Robin coefficients are non-zero anywhere on the reflecting boundary
    bool areRobinConditionsPureNeumann();

    // checks if Robin coefficients are non-negative everywhere on the reflecting boundary
    bool areRobinCoeffsNonnegative();
};
