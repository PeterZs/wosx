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

// This file implements the stochastic barycentric coordinates method of De Goes
// and Desbrun [2024], specialized to harmonic coordinates generated from walk-on-spheres
// samples on a closed triangular cage. It accumulates boundary hit contributions,
// optionally applies RKPM correction and cotangent-Laplacian denoising, and stores
// the resulting coordinates as a sparse matrix for deforming an embedded mesh.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <Eigen/SparseCholesky>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <array>
#include <vector>
#include <set>
#include <chrono>

struct MeshData {
    // members
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3i> indices;

    // helper function to convert positions to an Eigen matrix
    Eigen::MatrixXf convertPositionsToEigen() const;

    // helper function to extract unique undirected mesh edges
    std::vector<std::array<size_t, 2>> getEdges() const;
};

class HarmonicCoordinatesSolver {
public:
    // constructor
    HarmonicCoordinatesSolver(const MeshData& embeddedMeshData_,
                              const MeshData& cageMeshData_,
                              bool enableRKPM_,
                              bool enableDenoising_,
                              int nWalksPerBatch_);

    // accumulates coordinates data from generated harmonic samples
    void accumulateSamples(const std::vector<Eigen::Vector3f>& hitPoints,
                           const std::vector<uint32_t>& hitTriangleIds);

    // finalizes coordinates from the accumulated data
    void finalizeCoordinates();

    // prints sparsity and partition-of-unity diagnostics for coordinates
    void analyzeCoordinates() const;

    // checks whether harmonic coordinates have been finalized
    bool hasFinalizedCoordinates() const;

    // returns the number of accumulated samples per query point
    int getSamplesAccumulatedPerQueryPoint() const;

    // tests linear precision by reconstructing the embedded mesh geometry
    // from affinely transformed cage positions
    Eigen::MatrixXf testLinearPrecision(const Eigen::MatrixXf& cagePositions) const;

    // applies cage displacements to the embedded mesh geometry
    Eigen::MatrixXf deformEmbeddedGeometry(const Eigen::MatrixXf& deformedCagePositions) const;

    // resets accumulated coordinates data
    void reset();

    // toggles RKPM correction (resets accumulated coordinates data)
    void toggleRKPM();
    bool isRKPMEnabled() const;

    // toggles Laplacian denoising (invalidates finalized coordinates)
    void toggleDenoising();
    bool isDenoisingEnabled() const;

private:
    // members
    const MeshData& embeddedMeshData;
    const MeshData& cageMeshData;
    bool enableRKPM;
    bool enableDenoising;
    int nWalksPerBatch;
    int nAccumulatedSamples;
    float meanSquaredEdgeLength;
    struct CoordinateEntry {
        int vIndex;
        std::array<float, 4> value;
    };
    struct CoefficientEntry {
        int qIndex;
        std::array<float, 4> value;
    };
    Eigen::MatrixXf restEmbeddedPositions;
    Eigen::MatrixXf restCagePositions;
    Eigen::MatrixXf momentData;
    std::vector<std::vector<CoordinateEntry>> mCoordinateEntries;
    Eigen::SparseMatrix<float> laplacianMatrix, massMatrix;
    Eigen::SparseMatrix<float, Eigen::RowMajor> alpha;

    // assembles laplacian and mass matrices for denoising the coordinates
    void assembleDenoisingMatrices();

    // computes denoised coordinates from the RKPM coefficient values
    void denoiseCoordinates(const std::vector<std::vector<CoefficientEntry>>& uCoefficientEntries);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

inline Eigen::MatrixXf MeshData::convertPositionsToEigen() const
{
    Eigen::MatrixXf positionMatrix((int)positions.size(), 3);
    for (int i = 0; i < (int)positions.size(); i++) {
        positionMatrix.row(i) = positions[i].transpose();
    }

    return positionMatrix;
}

inline std::vector<std::array<size_t, 2>> MeshData::getEdges() const
{
    std::set<std::pair<int, int>> edgeSet;
    std::vector<std::array<size_t, 2>> edges;

    auto addEdge = [&](int i, int j) {
        if (i > j) std::swap(i, j);
        if (edgeSet.emplace(i, j).second) {
            edges.push_back({(size_t)i, (size_t)j});
        }
    };

    for (const Eigen::Vector3i& face: indices) {
        addEdge(face[0], face[1]);
        addEdge(face[1], face[2]);
        addEdge(face[2], face[0]);
    }

    return edges;
}

inline HarmonicCoordinatesSolver::HarmonicCoordinatesSolver(const MeshData& embeddedMeshData_,
                                                            const MeshData& cageMeshData_,
                                                            bool enableRKPM_,
                                                            bool enableDenoising_,
                                                            int nWalksPerBatch_):
embeddedMeshData(embeddedMeshData_),
cageMeshData(cageMeshData_),
enableRKPM(enableRKPM_),
enableDenoising(enableDenoising_),
nWalksPerBatch(nWalksPerBatch_),
nAccumulatedSamples(0),
meanSquaredEdgeLength(0.0),
restEmbeddedPositions(embeddedMeshData_.convertPositionsToEigen()),
restCagePositions(cageMeshData_.convertPositionsToEigen()),
momentData(Eigen::MatrixXf::Zero((int)embeddedMeshData_.positions.size(), 16)),
mCoordinateEntries(embeddedMeshData_.positions.size())
{
    // assemble laplacian and mass matrices for denoising the coordinates
    assembleDenoisingMatrices();
}

inline void HarmonicCoordinatesSolver::accumulateSamples(const std::vector<Eigen::Vector3f>& hitPoints,
                                                         const std::vector<uint32_t>& hitTriangleIds)
{
    // reset previously finalized coordinates
    alpha.resize(0, 0);

    // define a lambda function to compute barycentric coordinates
    auto computeBarycentricCoordinates = [](const Eigen::Vector3f& p,
                                            const Eigen::Vector3f& pa,
                                            const Eigen::Vector3f& pb,
                                            const Eigen::Vector3f& pc) -> Eigen::Vector2f {
        Eigen::Vector3f v1 = pb - pa;
        Eigen::Vector3f v2 = pc - pa;
        Eigen::Vector3f v3 = p - pa;

        float d11 = v1.dot(v1);
        float d12 = v1.dot(v2);
        float d22 = v2.dot(v2);
        float d31 = v3.dot(v1);
        float d32 = v3.dot(v2);
        float denom = d11*d22 - d12*d12;
        float v = (d22*d31 - d12*d32)/denom;
        float w = (d11*d32 - d12*d31)/denom;

        return Eigen::Vector2f(1.0f - v - w, v);
    };

    // define a lambda function to accumulate coordinate entries
    auto accumulateCoordinateEntry = [](std::vector<CoordinateEntry>& coordinateEntries,
                                        int vIndex, const Eigen::Vector4f& value) {
        for (CoordinateEntry& entry: coordinateEntries) {
            if (entry.vIndex == vIndex) {
                for (int a = 0; a < 4; a++) {
                    entry.value[a] += value[a];
                }
                return;
            }
        }

        CoordinateEntry entry = {vIndex, {value[0], value[1], value[2], value[3]}};
        coordinateEntries.emplace_back(entry);
    };

    // accumulate the samples
    auto start = std::chrono::high_resolution_clock::now();
    int nHitPoints = (int)hitPoints.size();
    int nValidHitPoints = 0;
    for (int sampleIndex = 0; sampleIndex < nHitPoints; sampleIndex++) {
        if (hitTriangleIds[sampleIndex] == std::numeric_limits<uint32_t>::max()) {
            continue;
        }

        // compute boundary barycentric coordinates for the hit point
        const Eigen::Vector3f& p = hitPoints[sampleIndex];
        const Eigen::Vector3i& vIndices = cageMeshData.indices[hitTriangleIds[sampleIndex]];
        Eigen::Vector2f coordinates = computeBarycentricCoordinates(p, cageMeshData.positions[vIndices[0]],
                                                                    cageMeshData.positions[vIndices[1]],
                                                                    cageMeshData.positions[vIndices[2]]);
        float phi[3] = {coordinates[0], coordinates[1], 1.0f - coordinates[0] - coordinates[1]};

        int queryIndex = sampleIndex/nWalksPerBatch;
        if (enableRKPM) {
            // center the hit point at the query point
            const Eigen::Vector3f& x = embeddedMeshData.positions[queryIndex];
            Eigen::Vector4f z;
            z << p[0] - x[0],
                 p[1] - x[1],
                 p[2] - x[2],
                 1.0;

            // update the RKPM moment matrix M
            for (int a = 0; a < 4; a++) {
                for (int b = 0; b < 4; b++) {
                    momentData(queryIndex, 4*a + b) += z[a]*z[b];
                }
            }

            // update the RKPM right-hand side m
            for (int j = 0; j < 3; j++) {
                accumulateCoordinateEntry(mCoordinateEntries[queryIndex], vIndices[j], phi[j]*z);
            }

        } else {
            // update the MC coordinate estimate
            for (int j = 0; j < 3; j++) {
                Eigen::Vector4f value = Eigen::Vector4f::Zero();
                value[0] = phi[j];
                accumulateCoordinateEntry(mCoordinateEntries[queryIndex], vIndices[j], value);
            }
        }

        nValidHitPoints++;
    }

    nAccumulatedSamples += nValidHitPoints;
    if (nValidHitPoints < nHitPoints) {
        std::cout << "Warning: " << nHitPoints - nValidHitPoints << " hit points were ignored" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "(CPU) Accumulated harmonic samples in " << elapsed.count() << " ms" << std::endl;
}

inline void HarmonicCoordinatesSolver::finalizeCoordinates()
{
    if (nAccumulatedSamples == 0) {
        std::cerr << "Error: Generate harmonic samples before finalizing coordinates" << std::endl;
        exit(EXIT_FAILURE);
    }

    auto start = std::chrono::high_resolution_clock::now();
    int nQueryPoints = (int)embeddedMeshData.positions.size();
    int nCageVertices = (int)cageMeshData.positions.size();
    std::vector<Eigen::Triplet<float>> alphaTriplets;
    std::vector<std::vector<CoefficientEntry>> uCoefficientEntries;
    if (enableRKPM && enableDenoising) {
        uCoefficientEntries.resize(nCageVertices);
    }

    if (enableRKPM) {
        // solve the RKPM linear system Mu = m for each query point
        for (int q = 0; q < nQueryPoints; q++) {
            // factorize the 4x4 moment matrix M for query point
            Eigen::Matrix4f M;
            for (int a = 0; a < 4; a++) {
                for (int b = 0; b < 4; b++) {
                    M(a, b) = momentData(q, 4*a + b);
                }
            }

            Eigen::FullPivLU<Eigen::Matrix4f> solver(M);
            if (solver.info() != Eigen::Success) {
                std::cerr << "Error: Failed to factorize the RKPM system" << std::endl;
                exit(EXIT_FAILURE);
            }

            // solve the linear system for every active cage vertex
            for (const CoordinateEntry& coordinateEntry: mCoordinateEntries[q]) {
                Eigen::Vector4f m;
                for (int a = 0; a < 4; a++) {
                    m[a] = coordinateEntry.value[a];
                }

                Eigen::Vector4f uLocal = solver.solve(m);
                if (solver.info() != Eigen::Success) {
                    std::cerr << "Error: Failed to solve the RKPM system" << std::endl;
                    exit(EXIT_FAILURE);
                }

                if (enableDenoising) {
                    // convert from the query-centered basis [p - x, 1] to global homogeneous coordinates [p, 1].
                    Eigen::Vector4f uGlobal;
                    uGlobal.head<3>() = uLocal.head<3>();
                    uGlobal[3] = uLocal[3] - uLocal.head<3>().dot(embeddedMeshData.positions[q]);
                    CoefficientEntry coefficientEntry = {q, {uGlobal[0], uGlobal[1],
                                                             uGlobal[2], uGlobal[3]}};
                    uCoefficientEntries[coordinateEntry.vIndex].emplace_back(coefficientEntry);

                } else {
                    // since z = [p - x, 1], the coordinate value at x is uLocal[3].
                    alphaTriplets.emplace_back(q, coordinateEntry.vIndex, uLocal[3]);
                }
            }
        }

    } else {
        // build the sparse MC coordinate matrix
        float scale = 1.0f/(float)getSamplesAccumulatedPerQueryPoint();
        for (int q = 0; q < nQueryPoints; q++) {
            for (const CoordinateEntry& entry: mCoordinateEntries[q]) {
                alphaTriplets.emplace_back(q, entry.vIndex, scale*entry.value[0]);
            }
        }
    }

    if (enableRKPM && enableDenoising) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "(CPU) Computed RKPM coefficient values in " << elapsed.count() << " ms" << std::endl;

        // compute denoised coordinates from the RKPM coefficient values
        denoiseCoordinates(uCoefficientEntries);

    } else {
        // build the sparse coordinate matrix from the triplets
        alpha.resize(nQueryPoints, nCageVertices);
        alpha.setFromTriplets(alphaTriplets.begin(), alphaTriplets.end());
        alpha.makeCompressed();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "(CPU) Computed " << (enableRKPM ? "RKPM-corrected" : "MC-based")
                  << " harmonic coordinates in " << elapsed.count() << " ms" << std::endl;

        if (enableDenoising && !enableRKPM) {
            std::cout << "Warning: Laplacian denoising is only applied when RKPM is enabled" << std::endl;
        }
    }
}

inline void HarmonicCoordinatesSolver::assembleDenoisingMatrices()
{
    // assemble the mass and Laplacian matrices
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::Triplet<float>> MTriplets;
    std::vector<Eigen::Triplet<float>> LTriplets;

    auto cotangent = [](const Eigen::Vector3f& p,
                        const Eigen::Vector3f& pa,
                        const Eigen::Vector3f& pb) -> float {
        Eigen::Vector3f u = pa - p;
        Eigen::Vector3f v = pb - p;
        float area2 = u.cross(v).norm();
        if (area2 <= 1e-10f) {
            return 0.0f;
        }

        return u.dot(v)/area2;
    };

    auto addLaplacianEntry = [&](int i, int j, float w) {
        LTriplets.emplace_back(i, i, w);
        LTriplets.emplace_back(j, j, w);
        LTriplets.emplace_back(i, j, -w);
        LTriplets.emplace_back(j, i, -w);
    };

    for (const Eigen::Vector3i& face: embeddedMeshData.indices) {
        int i = face[0]; const Eigen::Vector3f& pi = embeddedMeshData.positions[i];
        int j = face[1]; const Eigen::Vector3f& pj = embeddedMeshData.positions[j];
        int k = face[2]; const Eigen::Vector3f& pk = embeddedMeshData.positions[k];

        float area = 0.5f*(pj - pi).cross(pk - pi).norm();
        float mass = area/3.0f;
        MTriplets.emplace_back(i, i, mass);
        MTriplets.emplace_back(j, j, mass);
        MTriplets.emplace_back(k, k, mass);

        addLaplacianEntry(j, k, 0.5f*cotangent(pi, pj, pk));
        addLaplacianEntry(k, i, 0.5f*cotangent(pj, pk, pi));
        addLaplacianEntry(i, j, 0.5f*cotangent(pk, pi, pj));
    }

    int nQueryPoints = (int)embeddedMeshData.positions.size();
    massMatrix.resize(nQueryPoints, nQueryPoints);
    massMatrix.setFromTriplets(MTriplets.begin(), MTriplets.end());
    massMatrix.makeCompressed();

    laplacianMatrix.resize(nQueryPoints, nQueryPoints);
    laplacianMatrix.setFromTriplets(LTriplets.begin(), LTriplets.end());
    laplacianMatrix.makeCompressed();

    // compute the mean squared edge length
    meanSquaredEdgeLength = 0.0f;
    std::vector<std::array<size_t, 2>> meshEdges = embeddedMeshData.getEdges();
    for (const std::array<size_t, 2>& edge: meshEdges) {
        meanSquaredEdgeLength += (embeddedMeshData.positions[edge[0]] -
                                  embeddedMeshData.positions[edge[1]]).squaredNorm();
    }
    meanSquaredEdgeLength /= (float)meshEdges.size();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "(CPU) Assembled matrices needed for denoising in " << elapsed.count() << " ms" << std::endl;
}

inline void HarmonicCoordinatesSolver::denoiseCoordinates(const std::vector<std::vector<CoefficientEntry>>& uCoefficientEntries)
{
    // assemble and factorize the denoising matrix
    auto start = std::chrono::high_resolution_clock::now();
    int nQueryPoints = (int)embeddedMeshData.positions.size();
    float totalSamplesPerQueryPoint = getSamplesAccumulatedPerQueryPoint();
    float timeStep = meanSquaredEdgeLength*std::max(1.0f, 5.0f - std::log10(totalSamplesPerQueryPoint));
    Eigen::SparseMatrix<float> A = massMatrix + timeStep*laplacianMatrix;
    A.makeCompressed();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> denoisingSolver;
    denoisingSolver.compute(A);
    if (denoisingSolver.info() != Eigen::Success) {
        std::cerr << "Error: Failed to factorize the denoising matrix" << std::endl;
        exit(EXIT_FAILURE);
    }

    // denoise the RKPM coefficients and evaluate alpha_v(q) = u_v(q)^T [x, 1]
    int nCageVertices = (int)cageMeshData.positions.size();
    const float dropTolerance = 1e-10f;
    struct DenoisingWorkspace {
        // members
        Eigen::MatrixXf u;
        Eigen::MatrixXf rhs;
        Eigen::MatrixXf uDenoised;

        // constructor
        explicit DenoisingWorkspace(int nQueryPoints_):
            u(Eigen::MatrixXf::Zero(nQueryPoints_, 4)),
            rhs(nQueryPoints_, 4),
            uDenoised(nQueryPoints_, 4) {}
    };

    tbb::enumerable_thread_specific<DenoisingWorkspace> workspaces(
        [nQueryPoints]() { return DenoisingWorkspace(nQueryPoints); });
    std::vector<std::vector<Eigen::Triplet<float>>> tripletsByVertex(nCageVertices);

    auto run = [&](const tbb::blocked_range<int>& range) {
        DenoisingWorkspace& workspace = workspaces.local();
        for (int v = range.begin(); v < range.end(); v++) {
            // assemble the RKPM coefficient matrix
            workspace.u.setZero();
            for (const CoefficientEntry& entry: uCoefficientEntries[v]) {
                for (int a = 0; a < 4; a++) {
                    workspace.u(entry.qIndex, a) = entry.value[a];
                }
            }

            // solve the denoising system
            workspace.rhs.noalias() = massMatrix*workspace.u;
            workspace.uDenoised = denoisingSolver.solve(workspace.rhs);
            if (denoisingSolver.info() != Eigen::Success) {
                std::cerr << "Error: Failed to solve the denoising system" << std::endl;
                exit(EXIT_FAILURE);
            }

            // evaluate the denoised coordinates
            std::vector<Eigen::Triplet<float>>& vertexTriplets = tripletsByVertex[v];
            for (int q = 0; q < nQueryPoints; q++) {
                const Eigen::Vector3f& x = embeddedMeshData.positions[q];
                float alphaValue = workspace.uDenoised(q, 0)*x[0] +
                                   workspace.uDenoised(q, 1)*x[1] +
                                   workspace.uDenoised(q, 2)*x[2] +
                                   workspace.uDenoised(q, 3);
                if (std::fabs(alphaValue) > dropTolerance) {
                    vertexTriplets.emplace_back(q, v, alphaValue);
                }
            }
        }
    };

    tbb::blocked_range<int> range(0, nCageVertices);
    tbb::parallel_for(range, run);

    // build the coordinate matrix directly in column-major order; entries for
    // each cage vertex are already sorted by query index
    Eigen::SparseMatrix<float> alphaColumnMajor(nQueryPoints, nCageVertices);
    for (int v = 0; v < nCageVertices; v++) {
        alphaColumnMajor.startVec(v);
        std::vector<Eigen::Triplet<float>>& vertexTriplets = tripletsByVertex[v];
        for (const Eigen::Triplet<float>& triplet: vertexTriplets) {
            alphaColumnMajor.insertBackByOuterInner(v, triplet.row()) = triplet.value();
        }

        // release the completed column's triplet storage immediately
        std::vector<Eigen::Triplet<float>>().swap(vertexTriplets);
    }

    alphaColumnMajor.finalize();
    alpha = alphaColumnMajor; // convert to row-major storage for efficient per-query access

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "(CPU) Denoised harmonic coordinates in " << elapsed.count() << " ms" << std::endl;
}

inline void HarmonicCoordinatesSolver::analyzeCoordinates() const
{
    // print sparsity
    int nQueryPoints = (int)embeddedMeshData.positions.size();
    int nCageVertices = (int)cageMeshData.positions.size();
    float nEntries = (float)(nQueryPoints*nCageVertices);
    float nNonZeros = (float)alpha.nonZeros();
    std::cout << (enableRKPM ? "RKPM" : "MC")
              << " harmonic coordinates matrix fill rate = "
              << 100.0f*nNonZeros/nEntries << "%" << std::endl;

    // print partition-of-unity diagnostics
    float minRowSum = std::numeric_limits<float>::infinity();
    float maxRowSum = -std::numeric_limits<float>::infinity();
    float maxRowSumError = 0.0f;
    for (int q = 0; q < nQueryPoints; q++) {
        float rowSum = 0.0f;
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(alpha, q); it; ++it) {
            rowSum += it.value();
        }

        minRowSum = std::min(minRowSum, rowSum);
        maxRowSum = std::max(maxRowSum, rowSum);
        maxRowSumError = std::max(maxRowSumError, std::fabs(rowSum - 1.0f));
    }

    std::cout << "Partition-of-unity deviation: max |row sum - 1| = " << maxRowSumError
              << ", row sum range = [" << minRowSum << ", " << maxRowSum << "]" << std::endl;
}

inline bool HarmonicCoordinatesSolver::hasFinalizedCoordinates() const
{
    return alpha.rows() == (int)embeddedMeshData.positions.size() &&
           alpha.cols() == (int)cageMeshData.positions.size();
}

inline int HarmonicCoordinatesSolver::getSamplesAccumulatedPerQueryPoint() const
{
    return nAccumulatedSamples/(int)embeddedMeshData.positions.size();
}

inline Eigen::MatrixXf HarmonicCoordinatesSolver::testLinearPrecision(const Eigen::MatrixXf& cagePositions) const
{
    if (!hasFinalizedCoordinates()) {
        std::cerr << "Error: Harmonic coordinates must be finalized before they can be applied" << std::endl;
        exit(EXIT_FAILURE);
    }

    return alpha*cagePositions;
}

inline Eigen::MatrixXf HarmonicCoordinatesSolver::deformEmbeddedGeometry(const Eigen::MatrixXf& deformedCagePositions) const
{
    if (!hasFinalizedCoordinates()) {
        std::cerr << "Error: Harmonic coordinates must be finalized before they can be applied" << std::endl;
        exit(EXIT_FAILURE);
    }

    return restEmbeddedPositions + alpha*(deformedCagePositions - restCagePositions);
}

inline void HarmonicCoordinatesSolver::reset()
{
    std::cout << "Resetting harmonic coordinates" << std::endl;
    nAccumulatedSamples = 0;
    momentData.setZero();
    alpha.resize(0, 0);
    for (std::vector<CoordinateEntry>& entries: mCoordinateEntries) {
        entries.clear();
    }
}

inline void HarmonicCoordinatesSolver::toggleRKPM()
{
    enableRKPM = !enableRKPM;
    reset();
}

inline bool HarmonicCoordinatesSolver::isRKPMEnabled() const
{
    return enableRKPM;
}

inline void HarmonicCoordinatesSolver::toggleDenoising()
{
    enableDenoising = !enableDenoising;
    alpha.resize(0, 0);
}

inline bool HarmonicCoordinatesSolver::isDenoisingEnabled() const
{
    return enableDenoising;
}
