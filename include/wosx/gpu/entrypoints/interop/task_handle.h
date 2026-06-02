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

// GPUTaskHandle manages the GPU state needed to run one WoSX GPU task,
// such as solving a PDE, generating samples, or performing distance queries.
//
// It is more than a handle to a physical GPU device: it also stores the
// shader settings and GPU resources set for that task. Create a separate
// handle for each task that needs different geometry, PDE, samplers,
// dimensions, channels, or shader macros.

#pragma once

#include <wosx/gpu/entrypoints/interop/solvers.h>
#include <wosx/gpu/entrypoints/interop/library_paths.h>

namespace wosx {

template <typename T, size_t DIM>
class GPUTaskHandle {
public:
    // constructor
    GPUTaskHandle(const std::string& wosxDirectoryPath_,
                  const std::string& wosxPdeDirectoryPath_="",
                  const std::string& deviceBackend_="default");

    // set primary WoSX structures
    void setGeometricQueries(std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries_);
    void setAbsorbingBoundarySampler(std::shared_ptr<GPUBoundarySampler> sampler_);
    void setReflectingBoundarySampler(std::shared_ptr<GPUBoundarySampler> sampler_);
    void setDomainSampler(std::shared_ptr<GPUDomainSampler> sampler_);
    void setPDE(std::shared_ptr<GPUPDE> pde_);

    // initialize the handle (call after setting WoSX structures)
    void init(const std::vector<std::pair<std::string, std::string>>& userMacros_={});

    // return the GPU context
    GPUContext& getContext();

    // return the library modules
    const std::vector<GPUModule>& getLibraryModules() const;

    // returns WoSX directory paths
    std::string getWosxDirectoryPath() const;
    std::string getWosxGpuDirectoryPath() const;
    std::string getWosxPdeDirectoryPath() const;

    // return the device backend
    std::string getDeviceBackend() const;

    // get primary WoSX structures
    std::shared_ptr<GPUGeometricQueries<DIM>> getGeometricQueries() const;
    std::shared_ptr<GPUBoundarySampler> getAbsorbingBoundarySampler() const;
    std::shared_ptr<GPUBoundarySampler> getReflectingBoundarySampler() const;
    std::shared_ptr<GPUDomainSampler> getDomainSampler() const;
    std::shared_ptr<GPUPDE> getPDE() const;

private:
    // members
    GPUContext context;
    GPULibraryPaths libraryPaths;
    std::vector<GPUModule> libraryModules;
    std::string wosxDirectoryPath;
    std::string wosxPdeDirectoryPath;
    std::string deviceBackend;
    std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries;
    std::shared_ptr<GPUBoundarySampler> absorbingBoundarySampler;
    std::shared_ptr<GPUBoundarySampler> reflectingBoundarySampler;
    std::shared_ptr<GPUDomainSampler> domainSampler;
    std::shared_ptr<GPUPDE> pde;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
GPUTaskHandle<T, DIM>::GPUTaskHandle(const std::string& wosxDirectoryPath_,
                                     const std::string& wosxPdeDirectoryPath_,
                                     const std::string& deviceBackend_):
wosxDirectoryPath(wosxDirectoryPath_),
wosxPdeDirectoryPath(wosxPdeDirectoryPath_),
deviceBackend(deviceBackend_),
libraryPaths(wosxDirectoryPath_, wosxPdeDirectoryPath_),
geometricQueries(nullptr),
absorbingBoundarySampler(std::make_shared<GPUEmptyBoundarySampler<DIM>>()),
reflectingBoundarySampler(std::make_shared<GPUEmptyBoundarySampler<DIM>>()),
domainSampler(std::make_shared<GPUEmptyDomainSampler<DIM>>()),
pde(nullptr)
{
    // do nothing
}

template <typename T, size_t DIM>
void GPUTaskHandle<T, DIM>::setGeometricQueries(std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries_)
{
    geometricQueries = geometricQueries_;
}

template <typename T, size_t DIM>
void GPUTaskHandle<T, DIM>::setAbsorbingBoundarySampler(std::shared_ptr<GPUBoundarySampler> sampler_)
{
    absorbingBoundarySampler = sampler_;
}

template <typename T, size_t DIM>
void GPUTaskHandle<T, DIM>::setReflectingBoundarySampler(std::shared_ptr<GPUBoundarySampler> sampler_)
{
    reflectingBoundarySampler = sampler_;
}

template <typename T, size_t DIM>
void GPUTaskHandle<T, DIM>::setDomainSampler(std::shared_ptr<GPUDomainSampler> sampler_)
{
    domainSampler = sampler_;
}

template <typename T, size_t DIM>
void GPUTaskHandle<T, DIM>::setPDE(std::shared_ptr<GPUPDE> pde_)
{
    pde = pde_;
}

template <typename T, size_t DIM>
void GPUTaskHandle<T, DIM>::init(const std::vector<std::pair<std::string, std::string>>& userMacros_)
{
    // initialize the GPU context
    context.searchPaths = libraryPaths.getSearchPaths();
    context.macros = {
        {"WOSX_PROBLEM_DIMENSION", std::to_string(DIM)},
        {"WOSX_DATA_CHANNELS", std::to_string(getDataTypeChannels<T>())},
        {"WOSX_SOLVE_REGION_TYPE", std::to_string(static_cast<int>(domainSampler->getRegionType()))},
        {"WOSX_DOMAIN_SAMPLER_TYPE", std::to_string(static_cast<int>(domainSampler->getSamplerType()))},
        {"WOSX_ABSORBING_BOUNDARY_SAMPLER_TYPE", std::to_string(static_cast<int>(absorbingBoundarySampler->getSamplerType()))},
        {"WOSX_REFLECTING_BOUNDARY_SAMPLER_TYPE", std::to_string(static_cast<int>(reflectingBoundarySampler->getSamplerType()))}
    };
    if (geometricQueries) {
        context.macros.emplace_back(std::make_pair("WOSX_ABSORBING_BOUNDARY_TYPE",
                                                   std::to_string(static_cast<int>(geometricQueries->getAbsorbingBoundaryType()))));
        context.macros.emplace_back(std::make_pair("WOSX_REFLECTING_BOUNDARY_TYPE",
                                                   std::to_string(static_cast<int>(geometricQueries->getReflectingBoundaryType()))));
    }
    if (pde) {
        context.macros.emplace_back(std::make_pair("WOSX_PDE_TYPE",
                                                   std::to_string(static_cast<int>(pde->getType()))));
        context.macros.emplace_back(std::make_pair("WOSX_USE_KELVIN_TRANSFORM",
                                                   pde->usesKelvinTransform() ? "1" : "0"));
    }
    for (const auto& macro: userMacros_) {
        context.macros.emplace_back(macro);
    }
    context.initDevice(parseDeviceBackend(deviceBackend));

    // load the library modules
    std::vector<std::string> libraryModuleNames = libraryPaths.getModules();
    libraryModules.resize(libraryModuleNames.size());
    for (size_t i = 0; i < libraryModuleNames.size(); i++) {
        libraryModules[i].load(context, libraryModuleNames[i]);
    }

    // allocate the program resources
    if (geometricQueries) geometricQueries->allocate(context);
    absorbingBoundarySampler->allocate(context);
    reflectingBoundarySampler->allocate(context);
    domainSampler->allocate(context);
    if (pde) pde->allocate(context);
}

template <typename T, size_t DIM>
GPUContext& GPUTaskHandle<T, DIM>::getContext()
{
    return context;
}

template <typename T, size_t DIM>
const std::vector<GPUModule>& GPUTaskHandle<T, DIM>::getLibraryModules() const
{
    return libraryModules;
}

template <typename T, size_t DIM>
std::string GPUTaskHandle<T, DIM>::getWosxDirectoryPath() const
{
    return wosxDirectoryPath;
}

template <typename T, size_t DIM>
std::string GPUTaskHandle<T, DIM>::getWosxGpuDirectoryPath() const
{
    return libraryPaths.getWosxGpuDirectoryPath();
}

template <typename T, size_t DIM>
std::string GPUTaskHandle<T, DIM>::getWosxPdeDirectoryPath() const
{
    return wosxPdeDirectoryPath;
}

template <typename T, size_t DIM>
std::string GPUTaskHandle<T, DIM>::getDeviceBackend() const
{
    return deviceBackend;
}

template <typename T, size_t DIM>
std::shared_ptr<GPUGeometricQueries<DIM>> GPUTaskHandle<T, DIM>::getGeometricQueries() const
{
    return geometricQueries;
}

template <typename T, size_t DIM>
std::shared_ptr<GPUBoundarySampler> GPUTaskHandle<T, DIM>::getAbsorbingBoundarySampler() const
{
    return absorbingBoundarySampler;
}

template <typename T, size_t DIM>
std::shared_ptr<GPUBoundarySampler> GPUTaskHandle<T, DIM>::getReflectingBoundarySampler() const
{
    return reflectingBoundarySampler;
}

template <typename T, size_t DIM>
std::shared_ptr<GPUDomainSampler> GPUTaskHandle<T, DIM>::getDomainSampler() const
{
    return domainSampler;
}

template <typename T, size_t DIM>
std::shared_ptr<GPUPDE> GPUTaskHandle<T, DIM>::getPDE() const
{
    return pde;
}

} // wosx
