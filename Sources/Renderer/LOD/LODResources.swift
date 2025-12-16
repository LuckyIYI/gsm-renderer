import Metal
import simd
@_exported import RendererTypes

/// Manages GPU resources for LOD tree rendering.
/// Used with LODProjectEncoder for combined LOD+projection.
public final class LODResources {

    /// GPU buffer containing LOD tree nodes
    public let nodeBuffer: MTLBuffer

    /// GPU buffer containing a per-frame LOD traversal frontier (disjoint subtree roots)
    /// Used to seed GPU cut building without scanning all nodes.
    public let frontierNodeBuffer: MTLBuffer

    /// Number of frontier nodes in `frontierNodeBuffer`
    public let frontierNodeCount: UInt32

    /// GPU buffer containing merged Gaussians for interior nodes
    public let mergedGaussiansBuffer: MTLBuffer

    /// GPU buffer containing merged SH coefficients
    public let mergedHarmonicsBuffer: MTLBuffer

    /// GPU buffer mapping each LOD node index -> a representative leaf gaussian index.
    /// Used by "stamped" LOD selection to write one unique slot per selected cut node.
    public let representativeLeafBuffer: MTLBuffer

    /// Tree metadata
    public let header: LODTreeHeader

    /// Number of SH components per Gaussian
    public let shComponents: Int

    /// Precision used for merged buffers (must match the renderer's world/harmonics precision)
    public let precision: RenderPrecision

    /// Initialize LOD resources from a built LOD tree
    /// - Parameters:
    ///   - device: Metal device
    ///   - tree: LOD tree from LODTreeBuilder
    ///   - precision: Precision used for merged buffers (matches renderer input precision)
    ///   - targetFrontierCount: Approximate number of frontier nodes to seed GPU traversal
    /// - Throws: RendererError if buffer allocation fails
    public init(
        device: MTLDevice,
        tree: LODTreeBuilder.LODTree,
        precision: RenderPrecision = .float32,
        targetFrontierCount: Int = 8192
    ) throws {
        self.header = tree.header
        self.shComponents = tree.shComponents
        self.precision = precision

        // Allocate node buffer
        let nodeSize = MemoryLayout<LODNode>.stride * tree.nodes.count
        guard let nodeBuffer = device.makeBuffer(
            bytes: tree.nodes,
            length: max(nodeSize, 64),
            options: .storageModeShared
        ) else {
            throw RendererError.failedToAllocateBuffer(label: "LOD Nodes", size: nodeSize)
        }
        nodeBuffer.label = "LOD Tree Nodes"
        self.nodeBuffer = nodeBuffer

        // Build and upload a fixed traversal frontier (disjoint subtree roots).
        // This lets selection run in O(S) without scanning all nodes.
        let frontierNodes = Self.buildTraversalFrontier(nodes: tree.nodes, targetCount: targetFrontierCount)
        self.frontierNodeCount = UInt32(frontierNodes.count)
        let frontierSize = MemoryLayout<UInt32>.stride * max(frontierNodes.count, 1)
        guard let frontierBuffer = device.makeBuffer(
            bytes: frontierNodes.isEmpty ? [UInt32(0)] : frontierNodes,
            length: frontierSize,
            options: .storageModeShared
        ) else {
            throw RendererError.failedToAllocateBuffer(label: "LOD Frontier Nodes", size: frontierSize)
        }
        frontierBuffer.label = "LOD Frontier Nodes"
        self.frontierNodeBuffer = frontierBuffer

        // Build and upload representative leaf mapping (nodeIndex -> leaf gaussian index).
        let repLeaves = Self.buildRepresentativeLeaves(nodes: tree.nodes)
        let repSize = MemoryLayout<UInt32>.stride * max(repLeaves.count, 1)
        guard let repBuffer = device.makeBuffer(
            bytes: repLeaves.isEmpty ? [UInt32(0)] : repLeaves,
            length: repSize,
            options: .storageModeShared
        ) else {
            throw RendererError.failedToAllocateBuffer(label: "LOD Representative Leaves", size: repSize)
        }
        repBuffer.label = "LOD Representative Leaves"
        self.representativeLeafBuffer = repBuffer

        // Allocate merged Gaussians buffer
        switch precision {
        case .float32:
            let mergedGaussianSize = MemoryLayout<PackedWorldGaussian>.stride * max(tree.mergedGaussians.count, 1)
            guard let mergedBuffer = device.makeBuffer(
                bytes: tree.mergedGaussians.isEmpty ? [PackedWorldGaussian()] : tree.mergedGaussians,
                length: mergedGaussianSize,
                options: .storageModeShared
            ) else {
                throw RendererError.failedToAllocateBuffer(label: "LOD Merged Gaussians", size: mergedGaussianSize)
            }
            mergedBuffer.label = "LOD Merged Gaussians (float32)"
            self.mergedGaussiansBuffer = mergedBuffer

        case .float16:
            var mergedHalf = [PackedWorldGaussianHalf]()
            mergedHalf.reserveCapacity(max(tree.mergedGaussians.count, 1))
            if tree.mergedGaussians.isEmpty {
                mergedHalf.append(PackedWorldGaussianHalf())
            } else {
                for g in tree.mergedGaussians {
                    mergedHalf.append(
                        PackedWorldGaussianHalf(
                            position: SIMD3(g.px, g.py, g.pz),
                            scale: SIMD3(g.sx, g.sy, g.sz),
                            rotation: SIMD4(g.rotation.x, g.rotation.y, g.rotation.z, g.rotation.w),
                            opacity: g.opacity
                        )
                    )
                }
            }
            let mergedGaussianSize = MemoryLayout<PackedWorldGaussianHalf>.stride * mergedHalf.count
            guard let mergedBuffer = device.makeBuffer(
                bytes: mergedHalf,
                length: mergedGaussianSize,
                options: .storageModeShared
            ) else {
                throw RendererError.failedToAllocateBuffer(label: "LOD Merged Gaussians (half)", size: mergedGaussianSize)
            }
            mergedBuffer.label = "LOD Merged Gaussians (float16)"
            self.mergedGaussiansBuffer = mergedBuffer
        }

        // Allocate merged harmonics buffer
        switch precision {
        case .float32:
            let mergedHarmonicsSize = MemoryLayout<Float>.stride * max(tree.mergedHarmonics.count, 1)
            guard let harmonicsBuffer = device.makeBuffer(
                bytes: tree.mergedHarmonics.isEmpty ? [Float(0)] : tree.mergedHarmonics,
                length: mergedHarmonicsSize,
                options: .storageModeShared
            ) else {
                throw RendererError.failedToAllocateBuffer(label: "LOD Merged Harmonics", size: mergedHarmonicsSize)
            }
            harmonicsBuffer.label = "LOD Merged Harmonics (float32)"
            self.mergedHarmonicsBuffer = harmonicsBuffer

        case .float16:
            let mergedHalf: [Float16]
            if tree.mergedHarmonics.isEmpty {
                mergedHalf = [Float16(0)]
            } else {
                mergedHalf = tree.mergedHarmonics.map { Float16($0) }
            }
            let mergedHarmonicsSize = MemoryLayout<Float16>.stride * mergedHalf.count
            guard let harmonicsBuffer = device.makeBuffer(
                bytes: mergedHalf,
                length: mergedHarmonicsSize,
                options: .storageModeShared
            ) else {
                throw RendererError.failedToAllocateBuffer(label: "LOD Merged Harmonics (half)", size: mergedHarmonicsSize)
            }
            harmonicsBuffer.label = "LOD Merged Harmonics (float16)"
            self.mergedHarmonicsBuffer = harmonicsBuffer
        }
    }

    private static func buildTraversalFrontier(nodes: [LODNode], targetCount: Int) -> [UInt32] {
        guard !nodes.isEmpty else { return [] }
        let target = max(1, targetCount)

        var frontier: [UInt32] = [0]
        frontier.reserveCapacity(min(target, nodes.count))

        while frontier.count < target {
            guard let splitIndex = frontier.firstIndex(where: { nodes[Int($0)].isLeaf == 0 }) else { break }
            let nodeIndex = frontier[splitIndex]
            let node = nodes[Int(nodeIndex)]
            if node.leftChild == UInt32.max || node.rightChild == UInt32.max { break }

            frontier.remove(at: splitIndex)
            frontier.append(node.leftChild)
            frontier.append(node.rightChild)

            if frontier.count >= nodes.count { break }
        }

        return frontier
    }

    private static func buildRepresentativeLeaves(nodes: [LODNode]) -> [UInt32] {
        guard !nodes.isEmpty else { return [] }
        let invalid = UInt32.max
        var rep = [UInt32](repeating: invalid, count: nodes.count)

        // Post-order traversal with an explicit stack to avoid recursion.
        var stack: [(index: UInt32, expanded: Bool)] = [(0, false)]
        stack.reserveCapacity(64)

        while let top = stack.popLast() {
            let idx = Int(top.index)
            if idx < 0 || idx >= nodes.count { continue }
            if rep[idx] != invalid { continue }

            let node = nodes[idx]
            if node.isLeaf != 0 {
                rep[idx] = node.gaussianIndex
                continue
            }

            if !top.expanded {
                stack.append((top.index, true))
                if node.rightChild != invalid && rep[Int(node.rightChild)] == invalid {
                    stack.append((node.rightChild, false))
                }
                if node.leftChild != invalid && rep[Int(node.leftChild)] == invalid {
                    stack.append((node.leftChild, false))
                }
                continue
            }

            let left = node.leftChild != invalid ? rep[Int(node.leftChild)] : invalid
            let right = node.rightChild != invalid ? rep[Int(node.rightChild)] : invalid
            rep[idx] = left != invalid ? left : (right != invalid ? right : 0)
        }

        return rep
    }
}

// MARK: - LODTreeBuilder Extension for GPU Upload

public extension LODTreeBuilder.LODTree {

    /// Create GPU resources from this tree
    func createGPUResources(device: MTLDevice) throws -> LODResources {
        try LODResources(device: device, tree: self)
    }
}
