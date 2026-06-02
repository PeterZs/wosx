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

// This file extends the 'Bvh' structure from the FCPW library to support reflectance-based
// boundary conditions. Users of WoSX need not interact with this file directly.

implementing wosx;
__include utils.reflectance_boundary_bvh.geometry;

public interface IReflectanceAggregate
{
    // updates coefficient values for each node
    void updateCoefficientValues(uint nodeIndex);

    // computes the squared reflectance star radius
    void computeSquaredStarRadius(inout BoundingSphere s,
                                  bool flipNormalOrientation,
                                  float silhouettePrecision);
};

public struct ReflectanceBvhNode : IBvhNode
{
    private BoundingBox box;
    private BoundingCone cone;
    private uint nPrimitives;
    private uint offset;
    private float minCoefficientValue;
    private float maxCoefficientValue;

    // returns the bounding box of the node
    public BoundingBox getBoundingBox()
    {
        return box;
    }

    // sets the bounding box of the node
    [mutating]
    public void setBoundingBox(BoundingBox box_)
    {
        box = box_;
    }

    // checks if the node has a bounding cone
    public bool hasBoundingCone()
    {
        return true;
    }

    // returns the bounding cone of the node
    public BoundingCone getBoundingCone()
    {
        return cone;
    }

    // sets the bounding cone of the node
    [mutating]
    public void setBoundingCone(BoundingCone cone_)
    {
        cone = cone_;
    }

    // returns the minimum coefficient value of the node
    public float getMinCoefficientValue()
    {
        return minCoefficientValue;
    }

    // returns the maximum coefficient value of the node
    [mutating]
    public void setMinCoefficientValue(float minCoefficientValue_)
    {
        minCoefficientValue = minCoefficientValue_;
    }

    // returns the maximum coefficient value of the node
    public float getMaxCoefficientValue()
    {
        return maxCoefficientValue;
    }

    // sets the maximum coefficient value of the node
    [mutating]
    public void setMaxCoefficientValue(float maxCoefficientValue_)
    {
        maxCoefficientValue = maxCoefficientValue_;
    }

    // checks if the node is a leaf node
    public bool isLeaf()
    {
        return nPrimitives > 0;
    }

    // returns the offset to the right child of the interior node
    public uint getRightChildOffset()
    {
        return offset;
    }

    // returns the number of primitives in the node
    public uint getNumPrimitives()
    {
        return nPrimitives;
    }

    // returns the offset to the first primitive of the leaf node
    public uint getPrimitiveOffset()
    {
        return offset;
    }

    // returns the number of silhouettes in the node
    public uint getNumSilhouettes()
    {
        return 0;
    }

    // returns the offset to the first silhouette of the leaf node
    public uint getSilhouetteOffset()
    {
        return 0;
    }
};

public interface IReflectanceBvhNodeBound
{
    // computes the minimum squared star radius bound
    static float computeMinSquaredStarRadiusBound(float rMin, float rMax,
                                                  float minCoefficientValue, float maxCoefficientValue,
                                                  float minCosTheta, float maxCosTheta);

    // computes the maximum squared star radius bound
    static float computeMaxSquaredStarRadiusBound(float rMin, float rMax,
                                                  float minCoefficientValue, float maxCoefficientValue,
                                                  float minCosTheta, float maxCosTheta);
};

public struct ReflectanceBvh<Primitive, NodeBound> : IAggregate, IReflectanceAggregate
    where Primitive : IPrimitive
    where Primitive : IReflectancePrimitive
    where NodeBound : IReflectanceBvhNodeBound
{
    private RWStructuredBuffer<ReflectanceBvhNode> nodes;
    private StructuredBuffer<Primitive> primitives;

    // updates the bounding volume of an aggregate leaf node
    internal void refitLeafNode(uint nodeIndex)
    {
        // update leaf node's bounding box
        float3 pMin = float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 pMax = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        ReflectanceBvhNode node = nodes[nodeIndex];
        uint nPrimitives = node.getNumPrimitives();

        for (uint p = 0; p < nPrimitives; p++)
        {
            uint primitiveIndex = node.getPrimitiveOffset() + p;
            Primitive primitive = primitives[primitiveIndex];
            BoundingBox primitiveBox = primitive.getBoundingBox();
            pMin = min(pMin, primitiveBox.pMin);
            pMax = max(pMax, primitiveBox.pMax);
        }

        nodes[nodeIndex].setBoundingBox(BoundingBox(pMin, pMax));

        // update leaf node's bounding cone
        float3 centroid = 0.5 * (pMin + pMax);
        float3 axis = float3(0.0, 0.0, 0.0);
        float halfAngle = 0.0;
        float radius = 0.0;
        bool allPrimitivesHaveAdjacentFaces = true;

        for (uint p = 0; p < nPrimitives; p++)
        {
            uint primitiveIndex = node.getPrimitiveOffset() + p;
            Primitive primitive = primitives[primitiveIndex];
            axis += primitive.getNormal();

            for (uint k = 0; k < primitive.getAdjacentFaceCount(); k++)
            {
                float3 c = primitive.getAdjacentEdgeCenter(k);
                radius = max(radius, length(c - centroid));
                allPrimitivesHaveAdjacentFaces = allPrimitivesHaveAdjacentFaces &&
                                                 primitive.checkHasAdjacentFace(k);
            }
        }

        if (!allPrimitivesHaveAdjacentFaces)
        {
            halfAngle = M_PI;
        }
        else
        {
            float axisNorm = length(axis);
            if (axisNorm > FLT_EPSILON)
            {
                axis /= axisNorm;

                for (uint p = 0; p < nPrimitives; p++)
                {
                    uint primitiveIndex = node.getPrimitiveOffset() + p;
                    Primitive primitive = primitives[primitiveIndex];

                    float3 nj = primitive.getNormal();
                    float angle = acos(max(-1.0, min(1.0, dot(axis, nj))));
                    halfAngle = max(halfAngle, angle);

                    for (uint k = 0; k < primitive.getAdjacentFaceCount(); k++)
                    {
                        float3 nk = primitive.getAdjacentFaceNormal(k);
                        float angle = acos(max(-1.0, min(1.0, dot(axis, nk))));
                        halfAngle = max(halfAngle, angle);
                    }
                }
            }
        }

        nodes[nodeIndex].setBoundingCone(BoundingCone(axis, halfAngle, radius));
    }

    // updates the bounding volume of an aggregate internal node
    internal void refitInternalNode(uint nodeIndex)
    {
        ReflectanceBvhNode node = nodes[nodeIndex];
        uint leftNodeIndex = nodeIndex + 1;
        uint rightNodeIndex = nodeIndex + node.getRightChildOffset();
        ReflectanceBvhNode leftNode = nodes[leftNodeIndex];
        ReflectanceBvhNode rightNode = nodes[rightNodeIndex];

        // update internal node's bounding box
        BoundingBox leftBox = leftNode.getBoundingBox();
        BoundingBox rightBox = rightNode.getBoundingBox();
        BoundingBox mergedBox = mergeBoundingBoxes(leftBox, rightBox);
        nodes[nodeIndex].setBoundingBox(mergedBox);

        // update internal node's bounding cone
        BoundingCone leftCone = leftNode.getBoundingCone();
        BoundingCone rightCone = rightNode.getBoundingCone();
        BoundingCone mergedCone = mergeBoundingCones(leftCone, rightCone,
                                                     leftBox.getCentroid(),
                                                     rightBox.getCentroid(),
                                                     mergedBox.getCentroid());
        nodes[nodeIndex].setBoundingCone(mergedCone);
    }

    // updates the bounding volume of an aggregate node
    // NOTE: assumes node indices are provided in bottom-up order
    public void refit(uint nodeIndex)
    {
        if (nodes[nodeIndex].isLeaf())
        {
            refitLeafNode(nodeIndex);
        }
        else
        {
            refitInternalNode(nodeIndex);
        }
    }

    // updates coefficient values for each node
    // NOTE: assumes node indices are provided in bottom-up order
    public void updateCoefficientValues(uint nodeIndex)
    {
        ReflectanceBvhNode node = nodes[nodeIndex];
        if (node.isLeaf())
        {
            // update leaf node's coefficient values
            float minCoefficientValue = FLT_MAX;
            float maxCoefficientValue = -FLT_MAX;
            uint nPrimitives = node.getNumPrimitives();

            for (uint p = 0; p < nPrimitives; p++)
            {
                uint primitiveIndex = node.getPrimitiveOffset() + p;
                Primitive primitive = primitives[primitiveIndex];
                minCoefficientValue = min(minCoefficientValue, primitive.getMinCoefficientValue());
                maxCoefficientValue = max(maxCoefficientValue, primitive.getMaxCoefficientValue());
            }

            nodes[nodeIndex].setMinCoefficientValue(minCoefficientValue);
            nodes[nodeIndex].setMaxCoefficientValue(maxCoefficientValue);
        }
        else
        {
            // update internal node's coefficient values
            uint leftNodeIndex = nodeIndex + 1;
            uint rightNodeIndex = nodeIndex + node.getRightChildOffset();
            ReflectanceBvhNode leftNode = nodes[leftNodeIndex];
            ReflectanceBvhNode rightNode = nodes[rightNodeIndex];
            nodes[nodeIndex].setMinCoefficientValue(min(leftNode.getMinCoefficientValue(),
                                                        rightNode.getMinCoefficientValue()));
            nodes[nodeIndex].setMaxCoefficientValue(max(leftNode.getMaxCoefficientValue(),
                                                        rightNode.getMaxCoefficientValue()));
        }
    }

    // intersects aggregate geometry with ray
    public bool intersect(inout Ray r, bool checkForOcclusion, inout Interaction i)
    {
        TraversalStack traversalStack[FCPW_BVH_MAX_DEPTH];
        float4 distToChildNodes = float4(0.0, 0.0, 0.0, 0.0);
        BoundingBox rootBox = nodes[0].getBoundingBox();
        bool didIntersect = false;

        if (rootBox.intersect(r, distToChildNodes[0], distToChildNodes[1]))
        {
            traversalStack[0].node = 0;
            traversalStack[0].distance = distToChildNodes[0];
            int stackPtr = 0;

            while (stackPtr >= 0)
            {
                // pop off the next node to work on
                uint currentNodeIndex = traversalStack[stackPtr].node;
                float currentDist = traversalStack[stackPtr].distance;
                stackPtr--;

                // if this node is further than the closest found intersection, continue
                if (currentDist > r.tMax)
                {
                    continue;
                }

                ReflectanceBvhNode node = nodes[currentNodeIndex];
                if (node.isLeaf())
                {
                    // intersect primitives in leaf node
                    uint nPrimitives = node.getNumPrimitives();
                    for (uint p = 0; p < nPrimitives; p++)
                    {
                        Interaction c;
                        uint primitiveIndex = node.getPrimitiveOffset() + p;
                        bool didIntersectPrimitive = primitives[primitiveIndex].intersect(r, checkForOcclusion, c);

                        if (didIntersectPrimitive)
                        {
                            if (checkForOcclusion)
                            {
                                i.index = c.index;
                                return true;
                            }

                            didIntersect = true;
                            r.tMax = min(r.tMax, c.d);
                            i = c;
                        }
                    }
                }
                else
                {
                    // intersect child nodes
                    uint leftNodeIndex = currentNodeIndex + 1;
                    BoundingBox leftBox = nodes[leftNodeIndex].getBoundingBox();
                    bool didIntersectLeft = leftBox.intersect(r, distToChildNodes[0], distToChildNodes[1]);

                    uint rightNodeIndex = currentNodeIndex + node.getRightChildOffset();
                    BoundingBox rightBox = nodes[rightNodeIndex].getBoundingBox();
                    bool didIntersectRight = rightBox.intersect(r, distToChildNodes[2], distToChildNodes[3]);

                    // which nodes did we intersect?
                    if (didIntersectLeft && didIntersectRight)
                    {
                        // assume that the left child is closer
                        uint closer = leftNodeIndex;
                        uint other = rightNodeIndex;

                        // ... if the right child was actually closer, swap the relavent values
                        if (distToChildNodes[2] < distToChildNodes[0])
                        {
                            float tmpDist = distToChildNodes[0];
                            distToChildNodes[0] = distToChildNodes[2];
                            distToChildNodes[2] = tmpDist;

                            uint tmpNodeIndex = closer;
                            closer = other;
                            other = tmpNodeIndex;
                        }

                        // it's possible that the nearest primitive is still in the other node,
                        // but we'll check the farther-away node later.

                        // push the further node first, then the closer node
                        stackPtr++;
                        traversalStack[stackPtr].node = other;
                        traversalStack[stackPtr].distance = distToChildNodes[2];

                        stackPtr++;
                        traversalStack[stackPtr].node = closer;
                        traversalStack[stackPtr].distance = distToChildNodes[0];
                    }
                    else if (didIntersectLeft)
                    {
                        stackPtr++;
                        traversalStack[stackPtr].node = leftNodeIndex;
                        traversalStack[stackPtr].distance = distToChildNodes[0];
                    }
                    else if (didIntersectRight)
                    {
                        stackPtr++;
                        traversalStack[stackPtr].node = rightNodeIndex;
                        traversalStack[stackPtr].distance = distToChildNodes[2];
                    }
                }
            }
        }

        return didIntersect;
    }

    // intersects aggregate geometry with sphere
    public bool intersect<T : IBranchTraversalWeight>(BoundingSphere s, float3 randNums,
                                                      T branchTraversalWeight,
                                                      inout Interaction i)
    {
        float4 distToChildNodes = float4(0.0, 0.0, 0.0, 0.0);
        BoundingBox rootBox = nodes[0].getBoundingBox();
        uint currentNodeIndex = 0;
        uint selectedPrimitiveIndex = UINT_MAX;
        bool didIntersect = false;

        if (rootBox.overlap(s, distToChildNodes[0], distToChildNodes[1]))
        {
            float maxDistToChildNode = distToChildNodes[1];
            float traversalPdf = 1.0;
            float u = randNums[0];
            int stackPtr = 0;

            while (stackPtr >= 0)
            {
                // pop off the next node to work on
                stackPtr--;

                ReflectanceBvhNode node = nodes[currentNodeIndex];
                if (node.isLeaf())
                {
                    // probabilistically select a primitive
                    float totalPrimitiveWeight = 0.0;
                    uint nPrimitives = node.getNumPrimitives();
                    for (uint p = 0; p < nPrimitives; p++)
                    {
                        Interaction c;
                        bool didIntersectPrimitive = false;
                        uint primitiveIndex = node.getPrimitiveOffset() + p;
                        Primitive primitive = primitives[primitiveIndex];

                        if (maxDistToChildNode <= s.r2)
                        {
                            didIntersectPrimitive = true;
                            c.d = primitive.getSurfaceArea();
                            c.index = primitive.getIndex();
                        }
                        else
                        {
                            didIntersectPrimitive = primitive.intersect(s, c);
                        }

                        if (didIntersectPrimitive)
                        {
                            didIntersect = true;
                            totalPrimitiveWeight += c.d;
                            float selectionProb = c.d / totalPrimitiveWeight;

                            if (u < selectionProb)
                            {
                                u = u / selectionProb; // rescale to [0,1)
                                i = c;
                                i.d *= traversalPdf;
                                selectedPrimitiveIndex = primitiveIndex;
                            }
                            else
                            {
                                u = (u - selectionProb) / (1.0 - selectionProb);
                            }
                        }
                    }

                    if (totalPrimitiveWeight > 0.0)
                    {
                        i.d /= totalPrimitiveWeight;
                    }
                }
                else
                {
                    // probabilistically select one child node to traverse
                    uint leftNodeIndex = currentNodeIndex + 1;
                    BoundingBox leftBox = nodes[leftNodeIndex].getBoundingBox();
                    bool overlapsLeft = leftBox.overlap(s, distToChildNodes[0], distToChildNodes[1]);
                    float weightLeft = overlapsLeft ? 1.0 : 0.0;
                    if (weightLeft > 0.0)
                    {
                        float3 u = s.c - leftBox.getCentroid();
                        weightLeft *= branchTraversalWeight.compute(dot(u, u));
                    }

                    uint rightNodeIndex = currentNodeIndex + node.getRightChildOffset();
                    BoundingBox rightBox = nodes[rightNodeIndex].getBoundingBox();
                    bool overlapsRight = rightBox.overlap(s, distToChildNodes[2], distToChildNodes[3]);
                    float weightRight = overlapsRight ? 1.0 : 0.0;
                    if (weightRight > 0.0)
                    {
                        float3 u = s.c - rightBox.getCentroid();
                        weightRight *= branchTraversalWeight.compute(dot(u, u));
                    }

                    float totalTraversalWeight = weightLeft + weightRight;
                    if (totalTraversalWeight > 0.0)
                    {
                        stackPtr++;
                        float traversalProbLeft = weightLeft / totalTraversalWeight;
                        float traversalProbRight = 1.0 - traversalProbLeft;

                        if (u < traversalProbLeft)
                        {
                            u = u / traversalProbLeft; // rescale to [0,1)
                            currentNodeIndex = leftNodeIndex;
                            traversalPdf *= traversalProbLeft;
                            maxDistToChildNode = distToChildNodes[1];
                        }
                        else
                        {
                            u = (u - traversalProbLeft) / traversalProbRight; // rescale to [0,1)
                            currentNodeIndex = rightNodeIndex;
                            traversalPdf *= traversalProbRight;
                            maxDistToChildNode = distToChildNodes[3];
                        }
                    }
                }
            }
        }

        if (didIntersect)
        {
            if (i.index == UINT_MAX || selectedPrimitiveIndex == UINT_MAX)
            {
                didIntersect = false;
            }
            else
            {
                // sample a point on the selected geometric primitive
                float samplingPdf = primitives[selectedPrimitiveIndex].samplePoint(randNums.yz, i.uv, i.p, i.n);
                i.d *= samplingPdf;
            }
        }

        return didIntersect;
    }

    // finds closest point on aggregate geometry from sphere center
    public bool findClosestPoint(inout BoundingSphere s, inout Interaction i,
                                 bool recordNormal = false)
    {
        uint traversalStack[FCPW_BVH_MAX_DEPTH];
        float4 distToChildNodes = float4(0.0, 0.0, 0.0, 0.0);
        BoundingBox rootBox = nodes[0].getBoundingBox();
        uint selectedPrimitiveIndex = UINT_MAX;
        bool notFound = true;

        if (rootBox.overlap(s, distToChildNodes[0], distToChildNodes[1]))
        {
            s.r2 = min(s.r2, distToChildNodes[1]);
            traversalStack[0] = 0;
            int stackPtr = 0;

            while (stackPtr >= 0)
            {
                // pop off the next node to work on
                uint currentNodeIndex = traversalStack[stackPtr];
                ReflectanceBvhNode node = nodes[currentNodeIndex];
                stackPtr--;

                // if this node is further than the closest found primitive, continue
                BoundingBox currentBox = node.getBoundingBox();
                if (!currentBox.overlap(s, distToChildNodes[0]))
                {
                    continue;
                }

                if (node.isLeaf())
                {
                    // compute distance to primitives in leaf node
                    uint nPrimitives = node.getNumPrimitives();
                    for (uint p = 0; p < nPrimitives; p++)
                    {
                        Interaction c;
                        uint primitiveIndex = node.getPrimitiveOffset() + p;
                        bool found = primitives[primitiveIndex].findClosestPoint(s, c);

                        // keep the closest point only
                        if (found)
                        {
                            notFound = false;
                            s.r2 = min(s.r2, c.d * c.d);
                            i = c;
                            selectedPrimitiveIndex = primitiveIndex;
                        }
                    }
                }
                else
                {
                    // find distance to child nodes
                    uint leftNodeIndex = currentNodeIndex + 1;
                    BoundingBox leftBox = nodes[leftNodeIndex].getBoundingBox();
                    bool overlapsLeft = leftBox.overlap(s, distToChildNodes[0], distToChildNodes[1]);
                    s.r2 = min(s.r2, distToChildNodes[1]);

                    uint rightNodeIndex = currentNodeIndex + node.getRightChildOffset();
                    BoundingBox rightBox = nodes[rightNodeIndex].getBoundingBox();
                    bool overlapsRight = rightBox.overlap(s, distToChildNodes[2], distToChildNodes[3]);
                    s.r2 = min(s.r2, distToChildNodes[3]);

                    // which nodes do we overlap?
                    if (overlapsLeft && overlapsRight)
                    {
                        // assume that the left child is closer
                        uint closer = leftNodeIndex;
                        uint other = rightNodeIndex;

                        // ... if the right child was actually closer, swap the relavent values
                        if (distToChildNodes[2] < distToChildNodes[0])
                        {
                            uint tmpNodeIndex = closer;
                            closer = other;
                            other = tmpNodeIndex;
                        }

                        // it's possible that the nearest primitive is still in the other node,
                        // but we'll check the farther-away node later.

                        // push the further node first, then the closer node
                        stackPtr++;
                        traversalStack[stackPtr] = other;

                        stackPtr++;
                        traversalStack[stackPtr] = closer;
                    }
                    else if (overlapsLeft)
                    {
                        stackPtr++;
                        traversalStack[stackPtr] = leftNodeIndex;
                    }
                    else if (overlapsRight)
                    {
                        stackPtr++;
                        traversalStack[stackPtr] = rightNodeIndex;
                    }
                }
            }
        }

        if (!notFound && recordNormal)
        {
            i.n = primitives[selectedPrimitiveIndex].getNormal();
        }

        return !notFound;
    }

    // checks whether the node should be visited during traversal
    internal bool visitNode(BoundingSphere s,
                            uint nodeIndex,
                            inout float r2MinBound,
                            inout float r2MaxBound,
                            out bool hasSilhouette)
    {
        hasSilhouette = true;
        ReflectanceBvhNode node = nodes[nodeIndex];
        BoundingBox box = node.getBoundingBox();
        if (box.overlap(s, r2MinBound, r2MaxBound))
        {
            float minCoefficientValue = node.getMinCoefficientValue();
            if (minCoefficientValue < FLT_MAX) // early out for perfectly absorbing case
            {
                // perform silhouette test for reflecting boundaries
                float maximalAngles[2];
                BoundingCone cone = node.getBoundingCone();
                if (cone.overlap(s.c, box, r2MinBound, maximalAngles[0], maximalAngles[1]))
                {
                    r2MaxBound = FLT_MAX;
                }
                else
                {
                    hasSilhouette = false;
                    float maxCoefficientValue = node.getMaxCoefficientValue();
                    if (maxCoefficientValue > FLT_EPSILON)
                    {
                        // Reflectance case: compute radius bounds
                        float rMin = sqrt(r2MinBound);
                        float rMax = sqrt(r2MaxBound);
                        float minAbsCosTheta = min(abs(cos(maximalAngles[0])), abs(cos(maximalAngles[1])));
                        float maxAbsCosTheta = 1.0; // assume maxCosTheta = 1.0 for simplicity
                        r2MinBound = NodeBound.computeMinSquaredStarRadiusBound(
                            rMin, rMax, minCoefficientValue, maxCoefficientValue,
                            minAbsCosTheta, maxAbsCosTheta);
                        r2MaxBound = NodeBound.computeMaxSquaredStarRadiusBound(
                            rMin, rMax, minCoefficientValue, maxCoefficientValue,
                            minAbsCosTheta, maxAbsCosTheta);
                    }
                    else
                    {
                        // Perfectly reflecting case: r2MinBound becomes infinite,
                        // which means the node will not be visited
                        return false;
                    }
                }
            }

            return true;
        }

        return false;
    }

    // computes the squared reflectance star radius
    public void computeSquaredStarRadius(inout BoundingSphere s,
                                         bool flipNormalOrientation,
                                         float silhouettePrecision)
    {
        TraversalStack traversalStack[FCPW_BVH_MAX_DEPTH];
        float4 distToChildNodes = float4(0.0, 0.0, 0.0, 0.0);
        bool hasSilhouettes[2];

        if (visitNode(s, 0, distToChildNodes[0], distToChildNodes[1], hasSilhouettes[0]))
        {
            traversalStack[0].node = 0;
            traversalStack[0].distance = distToChildNodes[0];
            int stackPtr = 0;

            while (stackPtr >= 0)
            {
                // pop off the next node to work on
                uint currentNodeIndex = traversalStack[stackPtr].node;
                float currentDist = traversalStack[stackPtr].distance;
                stackPtr--;

                // if this node is further than the closest found primitive, continue
                if (abs(currentDist) > s.r2)
                {
                    continue;
                }

                ReflectanceBvhNode node = nodes[currentNodeIndex];
                if (node.isLeaf())
                {
                    uint nPrimitives = node.getNumPrimitives();
                    for (uint p = 0; p < nPrimitives; p++)
                    {
                        uint primitiveIndex = node.getPrimitiveOffset() + p;
                        Primitive primitive = primitives[primitiveIndex];
                        primitive.computeSquaredStarRadius(s, flipNormalOrientation, 
                                                           silhouettePrecision,
                                                           currentDist >= 0.0f);
                    }
                }
                else
                {
                    uint leftNodeIndex = currentNodeIndex + 1;
                    bool visitLeft = visitNode(s, leftNodeIndex, distToChildNodes[0],
                                               distToChildNodes[1], hasSilhouettes[0]);
                    if (visitLeft)
                    {
                        s.r2 = min(s.r2, distToChildNodes[1]);
                    }

                    uint rightNodeIndex = currentNodeIndex + node.getRightChildOffset();
                    bool visitRight = visitNode(s, rightNodeIndex, distToChildNodes[2],
                                                distToChildNodes[3], hasSilhouettes[1]);
                    if (visitRight)
                    {
                        s.r2 = min(s.r2, distToChildNodes[3]);
                    }

                    // is there overlap with both nodes?
                    if (visitLeft && visitRight)
                    {
                        // assume that the left child is closer
                        uint closer = leftNodeIndex;
                        uint other = rightNodeIndex;

                        // ... if the right child was actually closer, swap the relavent values
                        if (distToChildNodes[0] == 0.0 && distToChildNodes[2] == 0.0)
                        {
                            if (distToChildNodes[3] < distToChildNodes[1])
                            {
                                bool tmpHasSilhouettes = hasSilhouettes[0];
                                hasSilhouettes[0] = hasSilhouettes[1];
                                hasSilhouettes[1] = tmpHasSilhouettes;

                                uint tmpNodeIndex = closer;
                                closer = other;
                                other = tmpNodeIndex;
                            }
                        }
                        else if (distToChildNodes[2] < distToChildNodes[0])
                        {
                            float tmpDist = distToChildNodes[0];
                            distToChildNodes[0] = distToChildNodes[2];
                            distToChildNodes[2] = tmpDist;

                            bool tmpHasSilhouettes = hasSilhouettes[0];
                            hasSilhouettes[0] = hasSilhouettes[1];
                            hasSilhouettes[1] = tmpHasSilhouettes;

                            uint tmpNodeIndex = closer;
                            closer = other;
                            other = tmpNodeIndex;
                        }

                        // it's possible that the nearest primitive is still in the other node,
                        // but we'll check the farther-away node later.

                        // push the further node first, then the closer node
                        stackPtr++;
                        traversalStack[stackPtr].node = other;
                        traversalStack[stackPtr].distance = distToChildNodes[2] * (hasSilhouettes[1] ? 1.0 : -1.0);

                        stackPtr++;
                        traversalStack[stackPtr].node = closer;
                        traversalStack[stackPtr].distance = distToChildNodes[0] * (hasSilhouettes[0] ? 1.0 : -1.0);
                    }
                    else if (visitLeft)
                    {
                        stackPtr++;
                        traversalStack[stackPtr].node = leftNodeIndex;
                        traversalStack[stackPtr].distance = distToChildNodes[0] * (hasSilhouettes[0] ? 1.0 : -1.0);
                    }
                    else if (visitRight)
                    {
                        stackPtr++;
                        traversalStack[stackPtr].node = rightNodeIndex;
                        traversalStack[stackPtr].distance = distToChildNodes[2] * (hasSilhouettes[1] ? 1.0 : -1.0);
                    }
                }
            }
        }
    }

    // finds closest silhouette point on aggregate geometry from sphere center
    public bool findClosestSilhouettePoint(inout BoundingSphere s, bool flipNormalOrientation,
                                           float squaredMinRadius, float precision,
                                           inout Interaction i)
    {
        // NOTE: distance returned by computeSquaredStarRadius will in general
        // be smaller than the actual distance to the closest silhouette point
        computeSquaredStarRadius(s, flipNormalOrientation, precision);
        i.d = sqrt(s.r2);

        return true;
    }
};
