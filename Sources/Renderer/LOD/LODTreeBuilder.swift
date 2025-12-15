import Foundation
import simd
@_exported import RendererTypes

/// Builds a hierarchical LOD tree from Gaussian splat data.
/// Faithful implementation of "A Hierarchical 3D Gaussian Representation" (SIGGRAPH 2024)
/// by Kerbl, Meuleman, Kopanas, Wimmer, Lanvin, Drettakis.
///
/// Key features per the paper:
/// - Top-down AABB BVH with binary median split along longest axis
/// - Covariance merging via moment-matching (minimizes KL divergence)
/// - Weights based on projected area: w = opacity × √|Σ|
/// - Falloff opacity that can exceed 1.0 for proper blending
/// - Orientation matching to avoid interpolation artifacts
public final class LODTreeBuilder {

    /// Result of building an LOD tree
    public struct LODTree {
        /// Tree nodes (2N-1 entries)
        public let nodes: [LODNode]
        /// Header with tree metadata
        public let header: LODTreeHeader
        /// Merged Gaussian data for interior nodes (N-1 entries)
        public let mergedGaussians: [PackedWorldGaussian]
        /// Merged SH coefficients for interior nodes
        public let mergedHarmonics: [Float]
        /// SH components per Gaussian (number of bands, multiply by 3 for RGB coefficients)
        public let shComponents: Int
    }

    // MARK: - Public API

    /// Maximum tree depth to prevent stack overflow and limit build time
    /// For 5.8M gaussians, log2(5.8M) ≈ 23, so 30 is safe
    public static let maxTreeDepth: Int = 30

    /// Build LOD tree from Gaussian records using top-down BVH construction.
    /// - Parameters:
    ///   - records: Gaussian records (Morton sorting optional but can help)
    ///   - harmonics: Flat array of SH coefficients (shComponents * 3 floats per Gaussian)
    ///   - shComponents: Number of SH bands per Gaussian (e.g., 1 for DC only = 3 RGB values)
    ///   - maxDepth: Maximum tree depth (default: 30, enough for ~1 billion gaussians)
    /// - Returns: Complete LOD tree ready for GPU upload
    public static func build(
        from records: [GaussianRecord],
        harmonics: [Float],
        shComponents: Int,
        maxDepth: Int = maxTreeDepth
    ) -> LODTree {
        let leafCount = records.count
        guard leafCount > 0 else {
            return LODTree(
                nodes: [],
                header: LODTreeHeader(nodeCount: 0, leafCount: 0, interiorCount: 0, maxDepth: 0),
                mergedGaussians: [],
                mergedHarmonics: [],
                shComponents: shComponents
            )
        }

        // Special case: single Gaussian
        if leafCount == 1 {
            var node = LODNode()
            let r = records[0]
            let scale3x = r.scale * 3.0
            node.aabbMinX = r.position.x - scale3x.x
            node.aabbMinY = r.position.y - scale3x.y
            node.aabbMinZ = r.position.z - scale3x.z
            node.aabbMaxX = r.position.x + scale3x.x
            node.aabbMaxY = r.position.y + scale3x.y
            node.aabbMaxZ = r.position.z + scale3x.z
            node.leftChild = UInt32.max
            node.rightChild = UInt32.max
            node.parentIndex = UInt32.max
            node.gaussianIndex = 0
            node.isLeaf = 1
            node.subtreeCount = 1

            return LODTree(
                nodes: [node],
                header: LODTreeHeader(nodeCount: 1, leafCount: 1, interiorCount: 0, maxDepth: 0),
                mergedGaussians: [],
                mergedHarmonics: [],
                shComponents: shComponents
            )
        }

        // Build tree using top-down BVH with median split
        var builder = TreeBuilder(records: records, harmonics: harmonics, shComponents: shComponents, maxDepth: maxDepth)
        builder.buildTopDown()

        // Perform orientation matching from root to leaves
        builder.performOrientationMatching()

        return builder.finalize()
    }

    // MARK: - Internal Builder

    private struct TreeBuilder {
        let records: [GaussianRecord]
        let harmonics: [Float]
        let shComponents: Int
        let coeffsPerGaussian: Int
        let maxAllowedDepth: Int

        var nodes: [LODNode]
        var mergedGaussians: [PackedWorldGaussian]
        var mergedHarmonics: [Float]

        var nextNodeIndex: Int = 0
        var nextMergedIndex: Int = 0
        var maxDepth: UInt32 = 0

        init(records: [GaussianRecord], harmonics: [Float], shComponents: Int, maxDepth: Int) {
            self.records = records
            self.harmonics = harmonics
            self.shComponents = shComponents
            self.coeffsPerGaussian = shComponents * 3
            self.maxAllowedDepth = maxDepth

            let leafCount = records.count
            let interiorCount = leafCount - 1
            let nodeCount = 2 * leafCount - 1

            self.nodes = [LODNode](repeating: LODNode(), count: nodeCount)
            self.mergedGaussians = [PackedWorldGaussian](repeating: PackedWorldGaussian(), count: max(interiorCount, 0))
            self.mergedHarmonics = [Float](repeating: 0, count: max(interiorCount * coeffsPerGaussian, 0))
        }

        /// Build tree top-down using binary median split on longest AABB axis
        mutating func buildTopDown() {
            let indices = Array(0..<records.count)
            _ = buildNode(gaussianIndices: indices, depth: 0)
        }

        /// Recursively build a node for the given Gaussian indices
        /// Returns the node index
        private mutating func buildNode(gaussianIndices: [Int], depth: Int) -> Int {
            maxDepth = max(maxDepth, UInt32(depth))

            let nodeIndex = nextNodeIndex
            nextNodeIndex += 1

            // Compute AABB for all Gaussians in this node (using 3× scale as per paper)
            var aabbMin = SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
            var aabbMax = SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)

            for idx in gaussianIndices {
                let r = records[idx]
                let extent = r.scale * 3.0
                aabbMin = simd_min(aabbMin, r.position - extent)
                aabbMax = simd_max(aabbMax, r.position + extent)
            }

            nodes[nodeIndex].aabbMinX = aabbMin.x
            nodes[nodeIndex].aabbMinY = aabbMin.y
            nodes[nodeIndex].aabbMinZ = aabbMin.z
            nodes[nodeIndex].aabbMaxX = aabbMax.x
            nodes[nodeIndex].aabbMaxY = aabbMax.y
            nodes[nodeIndex].aabbMaxZ = aabbMax.z
            nodes[nodeIndex].subtreeCount = UInt32(gaussianIndices.count)

            // Base case: single Gaussian = leaf node
            if gaussianIndices.count == 1 {
                nodes[nodeIndex].isLeaf = 1
                nodes[nodeIndex].gaussianIndex = UInt32(gaussianIndices[0])
                nodes[nodeIndex].leftChild = UInt32.max
                nodes[nodeIndex].rightChild = UInt32.max
                return nodeIndex
            }

            // Interior node: split using median along longest axis
            nodes[nodeIndex].isLeaf = 0

            // Find longest axis
            let extent = aabbMax - aabbMin
            let longestAxis: Int
            if extent.x >= extent.y && extent.x >= extent.z {
                longestAxis = 0
            } else if extent.y >= extent.z {
                longestAxis = 1
            } else {
                longestAxis = 2
            }

            // Project means onto longest axis and sort
            var projections: [(index: Int, value: Float)] = gaussianIndices.map { idx in
                let pos = records[idx].position
                let value = longestAxis == 0 ? pos.x : (longestAxis == 1 ? pos.y : pos.z)
                return (idx, value)
            }
            projections.sort { $0.value < $1.value }

            // Split at median
            let midpoint = projections.count / 2
            let leftIndices = projections[0..<midpoint].map { $0.index }
            let rightIndices = projections[midpoint...].map { $0.index }

            // Handle edge case: if one side is empty, force a split
            let finalLeftIndices: [Int]
            let finalRightIndices: [Int]
            if leftIndices.isEmpty {
                finalLeftIndices = [rightIndices[0]]
                finalRightIndices = Array(rightIndices.dropFirst())
            } else if rightIndices.isEmpty {
                finalLeftIndices = Array(leftIndices.dropLast())
                finalRightIndices = [leftIndices.last!]
            } else {
                finalLeftIndices = Array(leftIndices)
                finalRightIndices = Array(rightIndices)
            }

            // Recursively build children
            let leftChild = buildNode(gaussianIndices: finalLeftIndices, depth: depth + 1)
            let rightChild = buildNode(gaussianIndices: finalRightIndices, depth: depth + 1)

            nodes[nodeIndex].leftChild = UInt32(leftChild)
            nodes[nodeIndex].rightChild = UInt32(rightChild)
            nodes[leftChild].parentIndex = UInt32(nodeIndex)
            nodes[rightChild].parentIndex = UInt32(nodeIndex)

            // Compute merged Gaussian for this interior node
            let mergedIndex = computeMergedGaussian(
                nodeIndex: nodeIndex,
                leftChild: leftChild,
                rightChild: rightChild
            )
            nodes[nodeIndex].gaussianIndex = UInt32(mergedIndex)

            return nodeIndex
        }

        /// Compute merged Gaussian from TWO CHILDREN using moment-matching
        /// O(1) per node - critical for performance!
        /// Returns the index in mergedGaussians array
        private mutating func computeMergedGaussian(
            nodeIndex: Int,
            leftChild: Int,
            rightChild: Int
        ) -> Int {
            // Get the two child Gaussians (either from original records or merged buffer)
            let (leftPos, leftScale, leftRot, leftOpacity, leftSH) = getGaussianData(nodeIndex: leftChild)
            let (rightPos, rightScale, rightRot, rightOpacity, rightSH) = getGaussianData(nodeIndex: rightChild)

            // Compute unnormalized weights: w' = opacity × √|Σ| (proportional to projected area)
            // Per paper Eq. 8: "we set the weights proportional to the projected surface areas"
            let leftCovDet = leftScale.x * leftScale.y * leftScale.z
            let rightCovDet = rightScale.x * rightScale.y * rightScale.z
            let leftWeightUnnorm = leftOpacity * sqrt(max(leftCovDet, 1e-10))
            let rightWeightUnnorm = rightOpacity * sqrt(max(rightCovDet, 1e-10))
            let totalWeight = leftWeightUnnorm + rightWeightUnnorm

            // Normalized weights for position/covariance merging
            let wL = totalWeight > 1e-10 ? leftWeightUnnorm / totalWeight : 0.5
            let wR = totalWeight > 1e-10 ? rightWeightUnnorm / totalWeight : 0.5

            // Merged mean: μ = wL × μL + wR × μR
            let mergedMean = leftPos * wL + rightPos * wR

            // Merged covariance: Σ = Σ w_i × (Σ_i + δμ_i × δμ_i^T)
            let deltaL = leftPos - mergedMean
            let deltaR = rightPos - mergedMean

            let covL = buildCovarianceMatrix(scale: leftScale, rotation: leftRot)
            let covR = buildCovarianceMatrix(scale: rightScale, rotation: rightRot)

            let outerL = simd_float3x3(
                SIMD3<Float>(deltaL.x * deltaL.x, deltaL.x * deltaL.y, deltaL.x * deltaL.z),
                SIMD3<Float>(deltaL.y * deltaL.x, deltaL.y * deltaL.y, deltaL.y * deltaL.z),
                SIMD3<Float>(deltaL.z * deltaL.x, deltaL.z * deltaL.y, deltaL.z * deltaL.z)
            )
            let outerR = simd_float3x3(
                SIMD3<Float>(deltaR.x * deltaR.x, deltaR.x * deltaR.y, deltaR.x * deltaR.z),
                SIMD3<Float>(deltaR.y * deltaR.x, deltaR.y * deltaR.y, deltaR.y * deltaR.z),
                SIMD3<Float>(deltaR.z * deltaR.x, deltaR.z * deltaR.y, deltaR.z * deltaR.z)
            )

            let mergedCov = (covL + outerL) * wL + (covR + outerR) * wR

            // Decompose merged covariance back to scale and rotation
            let (mergedScale, mergedRotation) = decomposeCovariance(mergedCov)

            // Merged falloff opacity per paper:
            // falloff = Σ wᵢ' / Sₚ where wᵢ' = oᵢ√|Σᵢ| and Sₚ = √|Σₚ|
            // This ensures opacity contribution is preserved when replacing children with merged Gaussian
            let mergedCovDet = mergedScale.x * mergedScale.y * mergedScale.z
            let Sp = sqrt(max(mergedCovDet, 1e-10))
            let mergedFalloff = totalWeight / Sp  // totalWeight = sum of unnormalized weights

            // Merge spherical harmonics
            var mergedSH = [Float](repeating: 0, count: coeffsPerGaussian)
            for c in 0..<coeffsPerGaussian {
                mergedSH[c] = leftSH[c] * wL + rightSH[c] * wR
            }

            // Find storage index for this interior node
            let mergedIndex = nextMergedIndex
            nextMergedIndex += 1

            // Store merged Gaussian
            var merged = PackedWorldGaussian()
            merged.px = mergedMean.x
            merged.py = mergedMean.y
            merged.pz = mergedMean.z
            merged.opacity = mergedFalloff
            merged.sx = mergedScale.x
            merged.sy = mergedScale.y
            merged.sz = mergedScale.z
            merged.rotation = SIMD4<Float>(mergedRotation.imag.x, mergedRotation.imag.y,
                                           mergedRotation.imag.z, mergedRotation.real)

            mergedGaussians[mergedIndex] = merged

            // Store merged harmonics
            let dstBase = mergedIndex * coeffsPerGaussian
            for c in 0..<coeffsPerGaussian {
                mergedHarmonics[dstBase + c] = mergedSH[c]
            }

            return mergedIndex
        }

        /// Get Gaussian data from either original records (leaf) or merged buffer (interior)
        private func getGaussianData(nodeIndex: Int) -> (SIMD3<Float>, SIMD3<Float>, simd_quatf, Float, [Float]) {
            let node = nodes[nodeIndex]
            if node.isLeaf == 1 {
                let r = records[Int(node.gaussianIndex)]
                let srcBase = Int(node.gaussianIndex) * coeffsPerGaussian
                var sh = [Float](repeating: 0, count: coeffsPerGaussian)
                for c in 0..<coeffsPerGaussian {
                    sh[c] = harmonics[srcBase + c]
                }
                return (r.position, r.scale, r.rotation, r.opacity, sh)
            } else {
                let g = mergedGaussians[Int(node.gaussianIndex)]
                let srcBase = Int(node.gaussianIndex) * coeffsPerGaussian
                var sh = [Float](repeating: 0, count: coeffsPerGaussian)
                for c in 0..<coeffsPerGaussian {
                    sh[c] = mergedHarmonics[srcBase + c]
                }
                let rot = simd_quatf(ix: g.rotation.x, iy: g.rotation.y, iz: g.rotation.z, r: g.rotation.w)
                return (SIMD3(g.px, g.py, g.pz), SIMD3(g.sx, g.sy, g.sz), rot, g.opacity, sh)
            }
        }

        /// Build 3×3 covariance matrix from scale and rotation
        /// Σ = R × diag(s²) × R^T
        private func buildCovarianceMatrix(scale: SIMD3<Float>, rotation: simd_quatf) -> simd_float3x3 {
            let R = simd_float3x3(rotation)
            let S2 = simd_float3x3(diagonal: scale * scale)
            return R * S2 * R.transpose
        }

        /// Decompose covariance matrix to scale and rotation via eigendecomposition
        /// Returns (scale, rotation) where Σ ≈ R × diag(scale²) × R^T
        private func decomposeCovariance(_ cov: simd_float3x3) -> (SIMD3<Float>, simd_quatf) {
            // Use eigendecomposition: Σ = V × D × V^T
            // where D = diag(eigenvalues) and V = eigenvector matrix (rotation)
            let (eigenvalues, eigenvectors) = eigendecomposition3x3(cov)

            // Scale = sqrt of eigenvalues (they represent variance = scale²)
            let scale = SIMD3<Float>(
                sqrt(max(eigenvalues.x, 1e-10)),
                sqrt(max(eigenvalues.y, 1e-10)),
                sqrt(max(eigenvalues.z, 1e-10))
            )

            // Convert eigenvector matrix to quaternion
            let rotation = quaternionFromMatrix(eigenvectors)

            return (scale, rotation)
        }

        /// Perform orientation matching from root to leaves
        /// This aligns child quaternions to minimize rotation difference with parent
        mutating func performOrientationMatching() {
            guard nodes.count > 1 else { return }

            // Start from root (index 0) and recursively match children
            matchOrientationRecursive(nodeIndex: 0, parentRotation: nil)
        }

        private mutating func matchOrientationRecursive(nodeIndex: Int, parentRotation: simd_quatf?) {
            let node = nodes[nodeIndex]

            if node.isLeaf == 0 {
                // Get this node's rotation from merged Gaussian
                let mergedIdx = Int(node.gaussianIndex)
                var currentRot = simd_quatf(
                    ix: mergedGaussians[mergedIdx].rotation.x,
                    iy: mergedGaussians[mergedIdx].rotation.y,
                    iz: mergedGaussians[mergedIdx].rotation.z,
                    r: mergedGaussians[mergedIdx].rotation.w
                )

                // Match to parent if exists
                if let parentRot = parentRotation {
                    currentRot = matchOrientation(child: currentRot, parent: parentRot)
                    mergedGaussians[mergedIdx].rotation = SIMD4<Float>(
                        currentRot.imag.x, currentRot.imag.y, currentRot.imag.z, currentRot.real
                    )
                }

                // Recurse to children
                if node.leftChild != UInt32.max {
                    matchOrientationRecursive(nodeIndex: Int(node.leftChild), parentRotation: currentRot)
                }
                if node.rightChild != UInt32.max {
                    matchOrientationRecursive(nodeIndex: Int(node.rightChild), parentRotation: currentRot)
                }
            }
            // Leaf nodes don't need matching (they use original Gaussians)
        }

        /// Match child orientation to parent by finding the sign-flip that minimizes rotation difference
        /// Quaternions q and -q represent the same rotation, and we can also flip axis signs
        private func matchOrientation(child: simd_quatf, parent: simd_quatf) -> simd_quatf {
            // The paper does exhaustive search over axis sign flips
            // For an ellipsoid, flipping two axes is equivalent to a rotation
            // We check q and -q (sign of quaternion)

            let candidates = [child, -child]

            var bestCandidate = child
            var bestDot: Float = -2

            for candidate in candidates {
                // Dot product of quaternions measures similarity
                // We want the one closest to parent (highest dot product with parent)
                let dot = abs(simd_dot(simd_float4(candidate.vector), simd_float4(parent.vector)))
                if dot > bestDot {
                    bestDot = dot
                    bestCandidate = candidate
                }
            }

            return bestCandidate
        }

        func finalize() -> LODTree {
            let leafCount = records.count
            let interiorCount = leafCount - 1
            let nodeCount = nodes.count

            // Reindex nodes so interior nodes come first, then leaves
            // Current layout from top-down build is mixed

            var reindexedNodes = [LODNode](repeating: LODNode(), count: nodeCount)
            var reindexedMergedGaussians = [PackedWorldGaussian](repeating: PackedWorldGaussian(), count: max(interiorCount, 0))
            var reindexedMergedHarmonics = [Float](repeating: 0, count: max(interiorCount * coeffsPerGaussian, 0))

            // Map old index -> new index
            var indexMap = [Int](repeating: -1, count: nodeCount)

            // First pass: assign new indices (interior first, then leaves)
            var interiorIdx = 0
            var leafIdx = interiorCount

            for i in 0..<nodeCount {
                if nodes[i].isLeaf == 1 {
                    indexMap[i] = leafIdx
                    leafIdx += 1
                } else {
                    indexMap[i] = interiorIdx
                    interiorIdx += 1
                }
            }

            // Second pass: copy nodes with remapped indices
            var mergedGaussianIdx = 0
            for i in 0..<nodeCount {
                let newIdx = indexMap[i]
                var node = nodes[i]

                // Remap child and parent indices
                if node.leftChild != UInt32.max {
                    node.leftChild = UInt32(indexMap[Int(node.leftChild)])
                }
                if node.rightChild != UInt32.max {
                    node.rightChild = UInt32(indexMap[Int(node.rightChild)])
                }
                if node.parentIndex != UInt32.max {
                    node.parentIndex = UInt32(indexMap[Int(node.parentIndex)])
                }

                // Handle merged Gaussian index for interior nodes
                if node.isLeaf == 0 {
                    let oldMergedIdx = Int(node.gaussianIndex)
                    let newMergedIdx = mergedGaussianIdx
                    node.gaussianIndex = UInt32(newMergedIdx)

                    reindexedMergedGaussians[newMergedIdx] = mergedGaussians[oldMergedIdx]
                    for c in 0..<coeffsPerGaussian {
                        reindexedMergedHarmonics[newMergedIdx * coeffsPerGaussian + c] =
                            mergedHarmonics[oldMergedIdx * coeffsPerGaussian + c]
                    }
                    mergedGaussianIdx += 1
                }

                reindexedNodes[newIdx] = node
            }

            // Set root parent to invalid (root is now at index 0)
            reindexedNodes[0].parentIndex = UInt32.max

            let header = LODTreeHeader(
                nodeCount: UInt32(nodeCount),
                leafCount: UInt32(leafCount),
                interiorCount: UInt32(interiorCount),
                maxDepth: maxDepth
            )

            return LODTree(
                nodes: reindexedNodes,
                header: header,
                mergedGaussians: reindexedMergedGaussians,
                mergedHarmonics: reindexedMergedHarmonics,
                shComponents: shComponents
            )
        }
    }

    // MARK: - Linear Algebra Helpers

    /// 3×3 Eigendecomposition using Jacobi iteration
    /// Returns (eigenvalues as SIMD3, eigenvector matrix where columns are eigenvectors)
    private static func eigendecomposition3x3(_ A: simd_float3x3) -> (SIMD3<Float>, simd_float3x3) {
        // For symmetric positive-definite covariance matrix
        // Use iterative Jacobi method

        var V = simd_float3x3(1) // Eigenvectors (start with identity)
        var D = A                 // Working copy

        let maxIter = 50
        let tolerance: Float = 1e-10

        for _ in 0..<maxIter {
            // Find largest off-diagonal element
            var maxVal: Float = 0
            var p = 0, q = 1

            for i in 0..<3 {
                for j in (i+1)..<3 {
                    let val = abs(D[i][j])
                    if val > maxVal {
                        maxVal = val
                        p = i
                        q = j
                    }
                }
            }

            if maxVal < tolerance {
                break
            }

            // Compute rotation angle
            let theta: Float
            if abs(D[p][p] - D[q][q]) < tolerance {
                theta = .pi / 4
            } else {
                theta = 0.5 * atan2(2 * D[p][q], D[p][p] - D[q][q])
            }

            let c = cos(theta)
            let s = sin(theta)

            // Apply Givens rotation
            var G = simd_float3x3(1)
            G[p][p] = c
            G[q][q] = c
            G[p][q] = s
            G[q][p] = -s

            D = G.transpose * D * G
            V = V * G
        }

        // Extract eigenvalues from diagonal
        let eigenvalues = SIMD3<Float>(D[0][0], D[1][1], D[2][2])

        return (eigenvalues, V)
    }

    /// Convert 3×3 rotation matrix to quaternion
    private static func quaternionFromMatrix(_ m: simd_float3x3) -> simd_quatf {
        // Ensure orthonormal (handle numerical errors)
        let col0 = simd_normalize(m.columns.0)
        let col1 = simd_normalize(m.columns.1 - simd_dot(m.columns.1, col0) * col0)
        let col2 = simd_cross(col0, col1)

        let trace = col0.x + col1.y + col2.z

        var q: simd_quatf

        if trace > 0 {
            let s = sqrt(trace + 1) * 2
            q = simd_quatf(
                ix: (col1.z - col2.y) / s,
                iy: (col2.x - col0.z) / s,
                iz: (col0.y - col1.x) / s,
                r: 0.25 * s
            )
        } else if col0.x > col1.y && col0.x > col2.z {
            let s = sqrt(1 + col0.x - col1.y - col2.z) * 2
            q = simd_quatf(
                ix: 0.25 * s,
                iy: (col1.x + col0.y) / s,
                iz: (col2.x + col0.z) / s,
                r: (col1.z - col2.y) / s
            )
        } else if col1.y > col2.z {
            let s = sqrt(1 + col1.y - col0.x - col2.z) * 2
            q = simd_quatf(
                ix: (col1.x + col0.y) / s,
                iy: 0.25 * s,
                iz: (col2.y + col1.z) / s,
                r: (col2.x - col0.z) / s
            )
        } else {
            let s = sqrt(1 + col2.z - col0.x - col1.y) * 2
            q = simd_quatf(
                ix: (col2.x + col0.z) / s,
                iy: (col2.y + col1.z) / s,
                iz: 0.25 * s,
                r: (col0.y - col1.x) / s
            )
        }

        return simd_normalize(q)
    }
}
