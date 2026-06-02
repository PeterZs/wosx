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

// This file extends the 'Baseline' structure from the FCPW library to support reflectance-based
// boundary conditions. Users of WoSX need not interact with this file directly.

#pragma once

#include <wosx/utils/reflectance_boundary_bvh/geometry.h>

namespace wosx {

using namespace fcpw;

template <size_t DIM, typename PrimitiveType=Primitive<DIM>>
class ReflectanceBaseline: public Baseline<DIM, PrimitiveType> {
public:
    // constructor
    ReflectanceBaseline(std::vector<PrimitiveType *>& primitives_,
                        std::vector<SilhouettePrimitive<DIM> *>& silhouettes_);

    // updates coefficient values for each primitive
    void updateCoefficientValues(const std::vector<float>& minCoefficientValues,
                                 const std::vector<float>& maxCoefficientValues);

    // computes the squared reflectance star radius
    int computeSquaredStarRadius(BoundingSphere<DIM>& s,
                                 bool flipNormalOrientation,
                                 float silhouettePrecision) const;
};

template <size_t DIM, typename PrimitiveType>
std::unique_ptr<ReflectanceBaseline<DIM, PrimitiveType>> createReflectanceBaseline(
                                            std::vector<PrimitiveType *>& primitives,
                                            std::vector<SilhouettePrimitive<DIM> *>& silhouettes);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <size_t DIM, typename PrimitiveType>
inline ReflectanceBaseline<DIM, PrimitiveType>::ReflectanceBaseline(std::vector<PrimitiveType *>& primitives_,
                                                                    std::vector<SilhouettePrimitive<DIM> *>& silhouettes_):
Baseline<DIM, PrimitiveType>(primitives_, silhouettes_)
{
    // do nothing
}

template <size_t DIM, typename PrimitiveType>
inline void ReflectanceBaseline<DIM, PrimitiveType>::updateCoefficientValues(const std::vector<float>& minCoefficientValues,
                                                                             const std::vector<float>& maxCoefficientValues)
{
    for (int p = 0; p < (int)Baseline<DIM, PrimitiveType>::primitives.size(); p++) {
        PrimitiveType *prim = Baseline<DIM, PrimitiveType>::primitives[p];

        prim->minCoefficientValue = minCoefficientValues[prim->getIndex()];
        prim->maxCoefficientValue = maxCoefficientValues[prim->getIndex()];
    }
}

template <size_t DIM, typename PrimitiveType>
inline int ReflectanceBaseline<DIM, PrimitiveType>::computeSquaredStarRadius(BoundingSphere<DIM>& s,
                                                                             bool flipNormalOrientation,
                                                                             float silhouettePrecision) const
{
    int nPrimitives = (int)Baseline<DIM, PrimitiveType>::primitives.size();
    for (int p = 0; p < nPrimitives; p++) {
        Baseline<DIM, PrimitiveType>::primitives[p]->computeSquaredStarRadius(s, flipNormalOrientation,
                                                                              silhouettePrecision);
    }

    return nPrimitives;
}

template <size_t DIM, typename PrimitiveType>
std::unique_ptr<ReflectanceBaseline<DIM, PrimitiveType>> createReflectanceBaseline(
                                            std::vector<PrimitiveType *>& primitives,
                                            std::vector<SilhouettePrimitive<DIM> *>& silhouettes)
{
    if (primitives.size() > 0) {
        return std::unique_ptr<ReflectanceBaseline<DIM, PrimitiveType>>(
                new ReflectanceBaseline<DIM, PrimitiveType>(primitives, silhouettes));
    }

    return nullptr;
}

} // wosx
