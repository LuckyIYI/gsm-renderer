import Foundation
import simd

struct GaussianRecord {
    var position: SIMD3<Float>
    var scale: SIMD3<Float>
    var rotation: simd_quatf
    var opacity: Float
    var sh: [Float] // Not strictly used in struct if we separate harmonics
}

struct CameraPose {
    var viewMatrix: simd_float4x4
    var projectionMatrix: simd_float4x4
    var cameraCenter: SIMD3<Float>
    var focalX: Float
    var focalY: Float
    var pixelFactor: Float
    var near: Float
    var far: Float
    var viewportSize: CGSize
}

// MARK: - Morton Code Sorting

/// Expands a 21-bit integer into 63 bits by inserting two zeros between each bit.
/// Used for 3D Morton code interleaving.
@inline(__always)
private func expandBits(_ v: UInt64) -> UInt64 {
    var x = v & 0x1FFFFF // Keep only 21 bits
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8))  & 0x100F00F00F00F00F
    x = (x | (x << 4))  & 0x10C30C30C30C30C3
    x = (x | (x << 2))  & 0x1249249249249249
    return x
}

/// Computes a 64-bit Morton code from normalized [0,1] coordinates.
/// Coordinates are quantized to 21 bits each, then interleaved as Z-Y-X.
@inline(__always)
private func mortonCode(x: Float, y: Float, z: Float) -> UInt64 {
    let scale: Float = Float((1 << 21) - 1) // 2^21 - 1
    let xi = UInt64(max(0, min(scale, x * scale)))
    let yi = UInt64(max(0, min(scale, y * scale)))
    let zi = UInt64(max(0, min(scale, z * scale)))
    return expandBits(xi) | (expandBits(yi) << 1) | (expandBits(zi) << 2)
}

extension GaussianSceneBuilder {
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
        var indices = Array(0..<count)

        for i in 0..<count {
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
        if coeffsPerGaussian > 0 && harmonics.count == count * coeffsPerGaussian {
            var sortedHarmonics = [Float](repeating: 0, count: harmonics.count)
            for (newIdx, oldIdx) in indices.enumerated() {
                let srcBase = oldIdx * coeffsPerGaussian
                let dstBase = newIdx * coeffsPerGaussian
                for c in 0..<coeffsPerGaussian {
                    sortedHarmonics[dstBase + c] = harmonics[srcBase + c]
                }
            }
            harmonics = sortedHarmonics
        }
    }
}

struct GaussianDataset {
    var records: [GaussianRecord]
    let harmonics: [Float]
    let shComponents: Int
    var cameraPoses: [CameraPose]
    var cameraNames: [String]
    var imageSize: CGSize?
}

class GaussianSceneBuilder {
    static func centroid(of records: [GaussianRecord]) -> SIMD3<Float> {
        guard !records.isEmpty else { return .zero }
        var sum = SIMD3<Float>(0, 0, 0)
        for r in records { sum += r.position }
        return sum / Float(records.count)
    }

    static func bounds(of records: [GaussianRecord]) -> (center: SIMD3<Float>, radius: Float) {
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
