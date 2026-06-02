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

// This file contains helper functions for reading settings from a JSON file.

#pragma once

#include "json.hpp"
using json = nlohmann::json;

template <typename T>
T getRequired(const json& j, const std::string& key)
{
    if (j.contains(key)) return j.at(key);
    std::cerr << "Missing required setting: " << key << std::endl;
    exit(EXIT_FAILURE);
}

template <typename T>
T getOptional(const json& j, const std::string& key, T defaultValue)
{
    if (j.contains(key)) return j.at(key);
    return defaultValue;
}
