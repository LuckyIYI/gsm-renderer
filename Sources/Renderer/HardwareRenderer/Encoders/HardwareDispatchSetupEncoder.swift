import Metal
import RendererTypes

/// Configuration for dispatch preparation
struct HardwareDispatchSetupConfig {
    let radixBlockSize: Int
    let radixGrainSize: Int
    let reorderThreadgroupSize: Int
    let numBlocks: Int

    var radixAlignment: UInt32 { UInt32(radixBlockSize * radixGrainSize) }
    var sortThreadgroupSize: UInt32 { UInt32(radixBlockSize) }
    var reorderTGSize: UInt32 { UInt32(reorderThreadgroupSize) }
    var maxNumBlocks: UInt32 { UInt32(numBlocks) }

    init(radixBlockSize: Int = 256, radixGrainSize: Int = 4, reorderThreadgroupSize: Int = 256, numBlocks: Int) {
        self.radixBlockSize = radixBlockSize
        self.radixGrainSize = radixGrainSize
        self.reorderThreadgroupSize = reorderThreadgroupSize
        self.numBlocks = numBlocks
    }
}

/// Encoder for preparing indirect dispatch arguments based on visible Gaussian counts.
/// Handles hardware stereo and mono rendering modes.
final class HardwareDispatchSetupEncoder {
    private let stereoPipeline: MTLComputePipelineState
    private let monoSortPipeline: MTLComputePipelineState
    private let stereoDrawArgsPipeline: MTLComputePipelineState
    private let meshDrawArgsPipeline: MTLComputePipelineState?

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let centerFn = library.makeFunction(name: "prepareStereoSortKernel"),
              let monoFn = library.makeFunction(name: "prepareMonoSortKernel"),
              let drawArgsFn = library.makeFunction(name: "prepareStereoDrawArgsKernel") else {
            throw RendererError.failedToCreatePipeline("Prepare dispatch kernels not found")
        }

        self.stereoPipeline = try device.makeComputePipelineState(function: centerFn)
        self.monoSortPipeline = try device.makeComputePipelineState(function: monoFn)
        self.stereoDrawArgsPipeline = try device.makeComputePipelineState(function: drawArgsFn)

        // Mesh draw args pipeline (optional - only available if mesh shaders are supported)
        if let meshDrawArgsFn = library.makeFunction(name: "prepareMeshDrawArgsKernel") {
            self.meshDrawArgsPipeline = try device.makeComputePipelineState(function: meshDrawArgsFn)
        } else {
            self.meshDrawArgsPipeline = nil
        }
    }

    /// Encode preparation for hardware stereo (single shared sort)
    func encodeStereo(
        commandBuffer: MTLCommandBuffer,
        visibleCount: MTLBuffer,
        header: MTLBuffer,
        sortHeader: MTLBuffer,
        sortDispatch: MTLBuffer,
        scanDispatch: MTLBuffer,
        reorderDispatch: MTLBuffer,
        config: HardwareDispatchSetupConfig
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PrepareStereoSort"
        encoder.setComputePipelineState(stereoPipeline)

        encoder.setBuffer(visibleCount, offset: 0, index: 0)
        encoder.setBuffer(header, offset: 0, index: 1)
        encoder.setBuffer(sortHeader, offset: 0, index: 2)
        encoder.setBuffer(sortDispatch, offset: 0, index: 3)
        encoder.setBuffer(scanDispatch, offset: 0, index: 4)
        encoder.setBuffer(reorderDispatch, offset: 0, index: 5)

        var radixAlignment = config.radixAlignment
        var sortTGSize = config.sortThreadgroupSize
        var reorderTGSize = config.reorderTGSize
        var maxNumBlocks = config.maxNumBlocks
        encoder.setBytes(&radixAlignment, length: 4, index: 6)
        encoder.setBytes(&sortTGSize, length: 4, index: 7)
        encoder.setBytes(&reorderTGSize, length: 4, index: 8)
        encoder.setBytes(&maxNumBlocks, length: 4, index: 9)

        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Encode preparation for mono rendering (single eye)
    func encodeMono(
        commandBuffer: MTLCommandBuffer,
        counter: MTLBuffer,
        header: MTLBuffer,
        sortHeader: MTLBuffer,
        sortDispatch: MTLBuffer,
        scanDispatch: MTLBuffer,
        reorderDispatch: MTLBuffer,
        drawArgs: MTLBuffer,
        config: HardwareDispatchSetupConfig
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PrepareMonoSort"
        encoder.setComputePipelineState(monoSortPipeline)

        encoder.setBuffer(counter, offset: 0, index: 0)
        encoder.setBuffer(header, offset: 0, index: 1)
        encoder.setBuffer(sortHeader, offset: 0, index: 2)
        encoder.setBuffer(sortDispatch, offset: 0, index: 3)
        encoder.setBuffer(scanDispatch, offset: 0, index: 4)
        encoder.setBuffer(reorderDispatch, offset: 0, index: 5)
        encoder.setBuffer(drawArgs, offset: 0, index: 6)

        var radixAlignment = config.radixAlignment
        var sortTGSize = config.sortThreadgroupSize
        var reorderTGSize = config.reorderTGSize
        var maxNumBlocks = config.maxNumBlocks
        encoder.setBytes(&radixAlignment, length: 4, index: 7)
        encoder.setBytes(&sortTGSize, length: 4, index: 8)
        encoder.setBytes(&reorderTGSize, length: 4, index: 9)
        encoder.setBytes(&maxNumBlocks, length: 4, index: 10)

        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Encode preparation of draw args for hardware stereo (instanced backend)
    func encodeStereoDrawArgs(
        commandBuffer: MTLCommandBuffer,
        header: MTLBuffer,
        drawArgs: MTLBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PrepareStereoDrawArgs"
        encoder.setComputePipelineState(stereoDrawArgsPipeline)

        encoder.setBuffer(header, offset: 0, index: 0)
        encoder.setBuffer(drawArgs, offset: 0, index: 1)

        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Encode preparation of mesh draw args for center-sort stereo
    func encodeMeshDrawArgs(
        commandBuffer: MTLCommandBuffer,
        header: MTLBuffer,
        meshDrawArgs: MTLBuffer
    ) {
        guard let pipeline = meshDrawArgsPipeline,
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PrepareMeshDrawArgs"
        encoder.setComputePipelineState(pipeline)

        encoder.setBuffer(header, offset: 0, index: 0)
        encoder.setBuffer(meshDrawArgs, offset: 0, index: 1)

        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
