import Metal
import simd
@_exported import RendererTypes

/// High-level manager for LOD tree operations.
/// Handles tree building and GPU resource management.
///
/// Usage:
/// 1. Build LOD tree from Gaussian records: `buildTree(from:harmonics:shComponents:)`
/// 2. Use LODProjectEncoder for combined LOD+projection in render pipeline
public final class LODManager {

    private let device: MTLDevice

    /// GPU resources for the current LOD tree (nil until tree is built)
    public private(set) var resources: LODResources?

    /// CPU-side tree data (for debugging/inspection)
    private(set) var tree: LODTreeBuilder.LODTree?

    /// Current granularity threshold
    public var granularityThreshold: Float = 1.0

    /// Initialize LOD manager
    /// - Parameter device: Metal device
    public init(device: MTLDevice) throws {
        self.device = device
    }

    /// Build LOD tree from Gaussian records
    /// - Parameters:
    ///   - records: Gaussian records (preferably Morton-sorted for best tree quality)
    ///   - harmonics: Spherical harmonics coefficients
    ///   - shComponents: Number of SH components per Gaussian
    ///   - precision: Precision used for merged buffers (must match renderer input precision)
    /// - Returns: Tree header with metadata
    @discardableResult
    public func buildTree(
        from records: [GaussianRecord],
        harmonics: [Float],
        shComponents: Int,
        precision: RenderPrecision = .float32
    ) throws -> LODTreeHeader {
        let tree = LODTreeBuilder.build(from: records, harmonics: harmonics, shComponents: shComponents)
        self.tree = tree

        let resources = try LODResources(device: device, tree: tree, precision: precision)
        self.resources = resources

        return tree.header
    }

    /// Check if LOD tree is built and ready
    public var isReady: Bool {
        resources != nil
    }

    /// Get tree metadata
    public var treeHeader: LODTreeHeader? {
        tree?.header
    }
}

// MARK: - GPU Buffer Access

public extension LODManager {

    /// Access to LOD node buffer for combined LOD+projection
    var nodeBuffer: MTLBuffer? {
        resources?.nodeBuffer
    }

    /// Access to merged Gaussians buffer for combined LOD+projection
    var mergedGaussiansBuffer: MTLBuffer? {
        resources?.mergedGaussiansBuffer
    }

    /// Access to merged harmonics buffer for combined LOD+projection
    var mergedHarmonicsBuffer: MTLBuffer? {
        resources?.mergedHarmonicsBuffer
    }
}
