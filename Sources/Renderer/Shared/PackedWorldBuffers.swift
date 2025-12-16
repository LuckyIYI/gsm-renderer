import Metal

/// GPU buffers containing packed Gaussian attributes and (optionally) SH harmonics.
public struct PackedWorldBuffers {
    public let packedGaussians: MTLBuffer
    public let harmonics: MTLBuffer

    public init(packedGaussians: MTLBuffer, harmonics: MTLBuffer) {
        self.packedGaussians = packedGaussians
        self.harmonics = harmonics
    }
}

