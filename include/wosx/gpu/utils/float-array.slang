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

// This file defines a FloatArray struct that can be used to store
// and manipulate arrays of float values.

implementing wosx;

public struct FloatArray<let CHANNELS : int> : IFloat
{
    private float data[CHANNELS];

    [Differentiable]
    public __init(float value)
    {
        [ForceUnroll]
        for (int i = 0; i < CHANNELS; ++i)
        {
            data[i] = value;
        }
    }

    [Differentiable]
    public __init(FloatArray<CHANNELS> other)
    {
        [ForceUnroll]
        for (int i = 0; i < CHANNELS; ++i)
        {
            data[i] = other.data[i];
        }
    }

    public __subscript(int index) -> float
    {
        get
        {
            return data[index];
        }
        set(float value)
        {
            data[index] = value;
        }
    }

    [Differentiable]
    public float toFloat()
    {
        return data[0];
    }

    public bool equals(FloatArray<CHANNELS> other)
    {
        [ForceUnroll]
        for (int i = 0; i < CHANNELS; ++i)
        {
            if (data[i] != other.data[i])
            {
                return false;
            }
        }

        return true;
    }

    public bool lessThan(FloatArray<CHANNELS> other)
    {
        [ForceUnroll]
        for (int i = 0; i < CHANNELS; ++i)
        {
            if (data[i] < other.data[i])
            {
                return true;
            }
        }

        return false;
    }

    public bool lessThanOrEquals(FloatArray<CHANNELS> other)
    {
        [ForceUnroll]
        for (int i = 0; i < CHANNELS; ++i)
        {
            if (data[i] <= other.data[i])
            {
                return true;
            }
        }

        return false;
    }

    [Differentiable]
    public FloatArray<CHANNELS> add(FloatArray<CHANNELS> other)
    {
        FloatArray<CHANNELS> result;
        [ForceUnroll]
        for (int i = 0; i < CHANNELS; ++i)
        {
            result.data[i] = data[i] + other.data[i];
        }

        return result;
    }

    [Differentiable]
    public FloatArray<CHANNELS> sub(FloatArray<CHANNELS> other)
    {
        FloatArray<CHANNELS> result;
        [ForceUnroll]
        for (int i = 0; i < CHANNELS; ++i)
        {
            result.data[i] = data[i] - other.data[i];
        }

        return result;
    }

    [Differentiable]
    public FloatArray<CHANNELS> mul(FloatArray<CHANNELS> other)
    {
        FloatArray<CHANNELS> result;
        [ForceUnroll]
        for (int i = 0; i < CHANNELS; ++i)
        {
            result.data[i] = data[i] * other.data[i];
        }

        return result;
    }

    [Differentiable]
    public FloatArray<CHANNELS> div(FloatArray<CHANNELS> other)
    {
        FloatArray<CHANNELS> result;
        [ForceUnroll]
        for (int i = 0; i < CHANNELS; ++i)
        {
            result.data[i] = data[i] / other.data[i];
        }

        return result;
    }

    [Differentiable]
    public FloatArray<CHANNELS> mod(FloatArray<CHANNELS> other)
    {
        FloatArray<CHANNELS> result;
        [ForceUnroll]
        for (int i = 0; i < CHANNELS; ++i)
        {
            result.data[i] = data[i] % other.data[i];
        }

        return result;
    }

    [Differentiable]
    public FloatArray<CHANNELS> neg()
    {
        FloatArray<CHANNELS> result;
        [ForceUnroll]
        for (int i = 0; i < CHANNELS; ++i)
        {
            result.data[i] = -data[i];
        }

        return result;
    }

    [Differentiable]
    public FloatArray<CHANNELS> scale<T : __BuiltinFloatingPointType>(T other)
    {
        FloatArray<CHANNELS> result;
        float otherFloat = other.toFloat();
        [ForceUnroll]
        for (int i = 0; i < CHANNELS; ++i)
        {
            result.data[i] = data[i] * otherFloat;
        }

        return result;
    }

    [Differentiable]
    public FloatArray<CHANNELS> operator+(FloatArray<CHANNELS> other)
    {
        return add(other);
    }

    [Differentiable]
    public FloatArray<CHANNELS> operator-(FloatArray<CHANNELS> other)
    {
        return sub(other);
    }

    [Differentiable]
    public FloatArray<CHANNELS> operator*(FloatArray<CHANNELS> other)
    {
        return mul(other);
    }

    [Differentiable]
    public FloatArray<CHANNELS> operator/(FloatArray<CHANNELS> other)
    {
        return div(other);
    }

    [Differentiable]
    public FloatArray<CHANNELS> operator%(FloatArray<CHANNELS> other)
    {
        return mod(other);
    }

    [Differentiable]
    public FloatArray<CHANNELS> operator-()
    {
        return neg();
    }
};
