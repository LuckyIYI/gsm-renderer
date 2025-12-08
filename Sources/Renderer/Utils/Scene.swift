import Foundation
import simd

public struct GaussianRecord {
    public var position: SIMD3<Float>
    public var scale: SIMD3<Float>
    public var rotation: simd_quatf
    public var opacity: Float
    public var sh: [Float] // Not strictly used in struct if we separate harmonics

    public init(position: SIMD3<Float>, scale: SIMD3<Float>, rotation: simd_quatf, opacity: Float, sh: [Float]) {
        self.position = position
        self.scale = scale
        self.rotation = rotation
        self.opacity = opacity
        self.sh = sh
    }
}

public struct CameraPose {
    public var viewMatrix: simd_float4x4
    public var projectionMatrix: simd_float4x4
    public var cameraCenter: SIMD3<Float>
    public var focalX: Float
    public var focalY: Float
    public var pixelFactor: Float
    public var near: Float
    public var far: Float
    public var viewportSize: CGSize

    public init(viewMatrix: simd_float4x4, projectionMatrix: simd_float4x4, cameraCenter: SIMD3<Float>, focalX: Float, focalY: Float, pixelFactor: Float, near: Float, far: Float, viewportSize: CGSize) {
        self.viewMatrix = viewMatrix
        self.projectionMatrix = projectionMatrix
        self.cameraCenter = cameraCenter
        self.focalX = focalX
        self.focalY = focalY
        self.pixelFactor = pixelFactor
        self.near = near
        self.far = far
        self.viewportSize = viewportSize
    }
}

// MARK: - Morton Code Sorting

/// Expands a 21-bit integer into 63 bits by inserting two zeros between each bit.
/// Used for 3D Morton code interleaving.
@inline(__always)
private func expandBits(_ v: UInt64) -> UInt64 {
    var x = v & 0x1FFFFF // Keep only 21 bits
    x = (x | (x << 32)) & 0x1F_0000_0000_FFFF
    x = (x | (x << 16)) & 0x1F_0000_FF00_00FF
    x = (x | (x << 8)) & 0x100F_00F0_0F00_F00F
    x = (x | (x << 4)) & 0x10C3_0C30_C30C_30C3
    x = (x | (x << 2)) & 0x1249_2492_4924_9249
    return x
}

/// Computes a 64-bit Morton code from normalized [0,1] coordinates.
/// Coordinates are quantized to 21 bits each, then interleaved as Z-Y-X.
@inline(__always)
private func mortonCode(x: Float, y: Float, z: Float) -> UInt64 {
    let scale = Float((1 << 21) - 1) // 2^21 - 1
    let xi = UInt64(max(0, min(scale, x * scale)))
    let yi = UInt64(max(0, min(scale, y * scale)))
    let zi = UInt64(max(0, min(scale, z * scale)))
    return expandBits(xi) | (expandBits(yi) << 1) | (expandBits(zi) << 2)
}

public extension GaussianSceneBuilder {
    /// Sorts Gaussians by Morton code for improved spatial locality.
    /// This groups nearby Gaussians together in memory, improving cache coherence during rendering.
    static func sortByMortonCode(
        records: inout [GaussianRecord],
        harmonics: inout [Float],
        shComponents: Int
    ) {
        guard records.count > 1 else { return }

        // 1. Compute bounding box for normalization
        var minP = records[0].position
        var maxP = records[0].position
        for r in records {
            minP = SIMD3<Float>(
                min(minP.x, r.position.x),
                min(minP.y, r.position.y),
                min(minP.z, r.position.z)
            )
            maxP = SIMD3<Float>(
                max(maxP.x, r.position.x),
                max(maxP.y, r.position.y),
                max(maxP.z, r.position.z)
            )
        }

        let extent = maxP - minP
        let invExtent = SIMD3<Float>(
            extent.x > 1e-6 ? 1.0 / extent.x : 0,
            extent.y > 1e-6 ? 1.0 / extent.y : 0,
            extent.z > 1e-6 ? 1.0 / extent.z : 0
        )

        // 2. Compute Morton codes for each Gaussian
        let count = records.count
        var mortonCodes = [UInt64](repeating: 0, count: count)
        var indices = Array(0 ..< count)

        for i in 0 ..< count {
            let pos = records[i].position
            let normalized = (pos - minP) * invExtent
            mortonCodes[i] = mortonCode(x: normalized.x, y: normalized.y, z: normalized.z)
        }

        // 3. Sort indices by Morton code
        indices.sort { mortonCodes[$0] < mortonCodes[$1] }

        // 4. Reorder records
        var sortedRecords = [GaussianRecord]()
        sortedRecords.reserveCapacity(count)
        for i in indices {
            sortedRecords.append(records[i])
        }
        records = sortedRecords

        // 5. Reorder harmonics if present
        let coeffsPerGaussian = shComponents * 3
        if coeffsPerGaussian > 0, harmonics.count == count * coeffsPerGaussian {
            var sortedHarmonics = [Float](repeating: 0, count: harmonics.count)
            for (newIdx, oldIdx) in indices.enumerated() {
                let srcBase = oldIdx * coeffsPerGaussian
                let dstBase = newIdx * coeffsPerGaussian
                for c in 0 ..< coeffsPerGaussian {
                    sortedHarmonics[dstBase + c] = harmonics[srcBase + c]
                }
            }
            harmonics = sortedHarmonics
        }
    }
}

public struct GaussianDataset {
    public var records: [GaussianRecord]
    public let harmonics: [Float]
    public let shComponents: Int
    public var cameraPoses: [CameraPose]
    public var cameraNames: [String]
    public var imageSize: CGSize?

    public init(records: [GaussianRecord], harmonics: [Float], shComponents: Int, cameraPoses: [CameraPose], cameraNames: [String], imageSize: CGSize?) {
        self.records = records
        self.harmonics = harmonics
        self.shComponents = shComponents
        self.cameraPoses = cameraPoses
        self.cameraNames = cameraNames
        self.imageSize = imageSize
    }
}

public class GaussianSceneBuilder {
    public static func centroid(of records: [GaussianRecord]) -> SIMD3<Float> {
        guard !records.isEmpty else { return .zero }
        var sum = SIMD3<Float>(0, 0, 0)
        for r in records {
            sum += r.position
        }
        return sum / Float(records.count)
    }

    public static func bounds(of records: [GaussianRecord]) -> (center: SIMD3<Float>, radius: Float) {
        guard let first = records.first else { return (.zero, 1.0) }

        var minP = first.position
        var maxP = first.position
        for r in records {
            minP = SIMD3<Float>(
                min(minP.x, r.position.x),
                min(minP.y, r.position.y),
                min(minP.z, r.position.z)
            )
            maxP = SIMD3<Float>(
                max(maxP.x, r.position.x),
                max(maxP.y, r.position.y),
                max(maxP.z, r.position.z)
            )
        }

        let center = (minP + maxP) * 0.5
        var radius: Float = 0
        for r in records {
            let offset = r.position - center
            let scaleMax = max(r.scale.x, max(r.scale.y, r.scale.z))
            radius = max(radius, simd_length(offset) + scaleMax)
        }
        radius = max(radius, simd_length(maxP - center))
        return (center, max(radius, 0.5))
    }
}
