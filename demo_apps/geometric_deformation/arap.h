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

// This file implements as-rigid-as-possible surface deformation for triangle meshes
// [Sorkine and Alexa 2007]. The Laplace matrix is assembled in the rest pose ---
// vertices can be free, fixed, or act as handles for deformation.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>

class ARAPSolver {
public:
    // constructor
    ARAPSolver(const Eigen::MatrixXf& meshPositions_,
               const std::vector<Eigen::Vector3i>& meshIndices_,
               const std::vector<int>& fixedVertexIndices_,
               const std::vector<int>& handleVertexIndices_);

    // updates the fixed vertex set
    void updateFixedVertexSet(const std::vector<int>& fixedVertexIndices_);

    // updates the handle vertex set
    void updateHandleVertexSet(const std::vector<int>& handleVertexIndices_);

    // updates the position of a handle vertex
    void updateHandlePosition(int index, const Eigen::Vector3f& position);

    // runs local-global iterations initialized from the current deformation,
    // and returns the deformed mesh positions
    Eigen::MatrixXf deform(int nIterations=5);

    // restores the rest pose
    void reset();

    // checks whether a vertex is fixed
    bool isFixedVertex(int index) const;

    // returns the fixed vertex set
    std::vector<int> getFixedVertexSet() const;

    // checks whether a vertex is a handle
    bool isHandleVertex(int index) const;

    // returns the handle vertex set
    std::vector<int> getHandleVertexSet() const;

private:
    // members
    struct AdjacentVertex {
        int index;
        float weight;
    };
    Eigen::MatrixXf meshPositions;
    Eigen::MatrixXf deformedPositions;
    std::vector<int> fixedVertexIndices;
    std::vector<int> handleVertexIndices;
    std::vector<bool> isFixedVertexList;
    std::vector<bool> isHandleVertexList;
    std::vector<std::vector<AdjacentVertex>> vertexAdjacencyList;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> reducedLaplacianSolver;

    // computes cotangent weights from the rest pose
    void computeCotangentWeights(const std::vector<Eigen::Vector3i>& meshIndices);

    // builds and factorizes the reduced Laplace matrix for the free vertices
    void buildReducedLaplacianSystem();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

inline ARAPSolver::ARAPSolver(const Eigen::MatrixXf& meshPositions_,
                              const std::vector<Eigen::Vector3i>& meshIndices_,
                              const std::vector<int>& fixedVertexIndices_,
                              const std::vector<int>& handleVertexIndices_):
meshPositions(meshPositions_),
deformedPositions(meshPositions_),
fixedVertexIndices(fixedVertexIndices_),
handleVertexIndices(handleVertexIndices_),
isFixedVertexList(meshPositions_.rows(), false),
isHandleVertexList(meshPositions_.rows(), false)
{
    // initialize the fixed and handle vertex sets
    for (int index: fixedVertexIndices) {
        isFixedVertexList[index] = true;
    }
    for (int index: handleVertexIndices) {
        isHandleVertexList[index] = true;
    }

    // compute cotangent weights from the rest pose
    computeCotangentWeights(meshIndices_);

    // build and factorize the reduced Laplace matrix
    buildReducedLaplacianSystem();
}

inline void ARAPSolver::computeCotangentWeights(const std::vector<Eigen::Vector3i>& meshIndices)
{
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

    auto addCotangentWeight = [&](int i, int j, float w) {
        auto updateVertexAdjacencyList = [&](int source, int target) {
            for (AdjacentVertex& neighbor: vertexAdjacencyList[source]) {
                if (neighbor.index == target) {
                    neighbor.weight += w;
                    return;
                }
            }

            vertexAdjacencyList[source].emplace_back(AdjacentVertex{target, w});
        };

        updateVertexAdjacencyList(i, j);
        updateVertexAdjacencyList(j, i);
    };

    // populate the vertex adjacency list with cotangent weights
    vertexAdjacencyList.resize(meshPositions.rows());
    for (const Eigen::Vector3i& face: meshIndices) {
        int i = face[0]; Eigen::Vector3f pi = meshPositions.row(i).transpose();
        int j = face[1]; Eigen::Vector3f pj = meshPositions.row(j).transpose();
        int k = face[2]; Eigen::Vector3f pk = meshPositions.row(k).transpose();

        addCotangentWeight(j, k, 0.5f*cotangent(pi, pj, pk));
        addCotangentWeight(k, i, 0.5f*cotangent(pj, pk, pi));
        addCotangentWeight(i, j, 0.5f*cotangent(pk, pi, pj));
    }

    // clamp negative cotangent weights to zero, as they make the local ARAP
    // objective indefinite and can cause vertices far from a handle to
    // overshoot its displacement
    for (std::vector<AdjacentVertex>& adjacentVertices: vertexAdjacencyList) {
        for (AdjacentVertex& neighbor: adjacentVertices) {
            neighbor.weight = std::max(neighbor.weight, 0.0f);
        }
    }
}

inline void ARAPSolver::buildReducedLaplacianSystem()
{
    auto start = std::chrono::high_resolution_clock::now();

    // map global vertex indices to rows in the reduced system
    int nFreeVertices = 0;
    std::vector<int> freeVertexIndices(meshPositions.rows(), -1);
    for (int i = 0; i < meshPositions.rows(); i++) {
        if (!isFixedVertex(i) && !isHandleVertex(i)) {
            freeVertexIndices[i] = nFreeVertices++;
        }
    }

    // assemble the reduced Laplace matrix
    std::vector<Eigen::Triplet<float>> triplets;
    for (int i = 0; i < meshPositions.rows(); i++) {
        int row = freeVertexIndices[i];
        if (row >= 0) {
            float diagonal = 0.0f;
            for (const AdjacentVertex& neighbor: vertexAdjacencyList[i]) {
                diagonal += neighbor.weight;
                int column = freeVertexIndices[neighbor.index];
                if (column >= 0) {
                    triplets.emplace_back(row, column, -neighbor.weight);
                }
            }

            triplets.emplace_back(row, row, diagonal);
        }
    }

    Eigen::SparseMatrix<float> reducedLaplaceMatrix(nFreeVertices, nFreeVertices);
    reducedLaplaceMatrix.setFromTriplets(triplets.begin(), triplets.end());

    // factorize the reduced Laplace matrix
    reducedLaplacianSolver.compute(reducedLaplaceMatrix);
    if (reducedLaplacianSolver.info() != Eigen::Success) {
        std::cerr << "Error: Failed to factorize the reduced Laplace matrix for ARAP" << std::endl;
        exit(EXIT_FAILURE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "(CPU) Built and factorized the Laplace matrix for ARAP in " << elapsed.count() << " ms" << std::endl;
}

inline void ARAPSolver::updateFixedVertexSet(const std::vector<int>& fixedVertexIndices_)
{
    // update the fixed vertex set and reset the deformed positions
    fixedVertexIndices = fixedVertexIndices_;
    std::fill(isFixedVertexList.begin(), isFixedVertexList.end(), false);
    for (int index: fixedVertexIndices) {
        isFixedVertexList[index] = true;
        deformedPositions.row(index) = meshPositions.row(index);
    }

    // rebuild and factorize the reduced Laplace matrix
    buildReducedLaplacianSystem();
}

inline void ARAPSolver::updateHandleVertexSet(const std::vector<int>& handleVertexIndices_)
{
    // update the handle vertex set
    handleVertexIndices = handleVertexIndices_;
    std::fill(isHandleVertexList.begin(), isHandleVertexList.end(), false);
    for (int index: handleVertexIndices) {
        isHandleVertexList[index] = true;
    }

    // rebuild and factorize the reduced Laplace matrix
    buildReducedLaplacianSystem();
}

inline void ARAPSolver::updateHandlePosition(int index, const Eigen::Vector3f& position)
{
    deformedPositions.row(index) = position.transpose();
}

inline Eigen::MatrixXf ARAPSolver::deform(int nIterations)
{
    // perform the local-global iterations
    Eigen::Matrix3f sigma = Eigen::Matrix3f::Identity();
    std::vector<Eigen::Matrix3f> rotations(meshPositions.rows(), Eigen::Matrix3f::Identity());
    for (int iteration = 0; iteration < nIterations; iteration++) {
        // local step: estimate the best-fit rotation at each mesh vertex
        int nFreeVertices = 0;
        for (int i = 0; i < meshPositions.rows(); i++) {
            // increment the free vertex counter
            if (!isFixedVertex(i) && !isHandleVertex(i)) {
                nFreeVertices++;
            }

            // build a covariance matrix from the rest and deformed positions
            Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
            Eigen::Vector3f pi = meshPositions.row(i).transpose();
            Eigen::Vector3f qi = deformedPositions.row(i).transpose();
            for (const AdjacentVertex& neighbor: vertexAdjacencyList[i]) {
                int j = neighbor.index;
                Eigen::Vector3f restEdge = pi - meshPositions.row(j).transpose();
                Eigen::Vector3f deformedEdge = qi - deformedPositions.row(j).transpose();
                covariance += neighbor.weight*restEdge*deformedEdge.transpose();
            }

            // perform an SVD to compute the best-fit rotation matrix
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(
                covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f U = svd.matrixU();
            Eigen::Matrix3f V = svd.matrixV();
            sigma(2, 2) = (V*U.transpose()).determinant();
            rotations[i] = V*sigma*U.transpose();
        }

        // global step: solve for the free mesh vertices with hard constraints
        Eigen::MatrixXf rhs = Eigen::MatrixXf::Zero(nFreeVertices, 3);
        nFreeVertices = 0;
        for (int i = 0; i < meshPositions.rows(); i++) {
            if (!isFixedVertex(i) && !isHandleVertex(i)) {
                // assemble the right-hand side of the reduced matrix system
                Eigen::Vector3f pi = meshPositions.row(i).transpose();
                Eigen::Vector3f value = Eigen::Vector3f::Zero();
                for (const AdjacentVertex& neighbor: vertexAdjacencyList[i]) {
                    int j = neighbor.index;
                    value += 0.5f*neighbor.weight*(rotations[i] + rotations[j])*
                             (pi - meshPositions.row(j).transpose());

                    if (isFixedVertex(j)) {
                        value += neighbor.weight*meshPositions.row(j).transpose();

                    } else if (isHandleVertex(j)) {
                        value += neighbor.weight*deformedPositions.row(j).transpose();
                    }
                }

                rhs.row(nFreeVertices) = value.transpose();
                nFreeVertices++;
            }
        }

        // solve for the free vertex positions
        Eigen::MatrixXf freePositions = reducedLaplacianSolver.solve(rhs);
        if (reducedLaplacianSolver.info() != Eigen::Success) {
            std::cerr << "Error: Failed to solve the reduced Laplace system for ARAP" << std::endl;
            exit(EXIT_FAILURE);
        }

        // update the deformed positions
        nFreeVertices = 0;
        for (int i = 0; i < meshPositions.rows(); i++) {
            if (!isFixedVertex(i) && !isHandleVertex(i)) {
                deformedPositions.row(i) = freePositions.row(nFreeVertices);
                nFreeVertices++;
            }
        }
    }

    return deformedPositions;
}

inline void ARAPSolver::reset()
{
    deformedPositions = meshPositions;
}

inline bool ARAPSolver::isFixedVertex(int index) const
{
    return isFixedVertexList[index];
}

inline std::vector<int> ARAPSolver::getFixedVertexSet() const
{
    return fixedVertexIndices;
}

inline bool ARAPSolver::isHandleVertex(int index) const
{
    return isHandleVertexList[index];
}

inline std::vector<int> ARAPSolver::getHandleVertexSet() const
{
    return handleVertexIndices;
}
