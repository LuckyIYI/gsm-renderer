import Metal
import RendererTypes

/// Dispatch slot indices matching the Metal kernel constants
enum DepthFirstDispatchSlot: Int {
    case depthHistogram = 0
    case depthScanBlocks = 1
    case depthExclusive = 2
    case depthApply = 3
    case depthScatter = 4
    case applyDepthOrder = 5
    case prefixSum = 6
    case prefixLevel2Reduce = 7
    case prefixLevel2Scan = 8
    case prefixLevel2Apply = 9
    case prefixFinalScan = 10
    case createInstances = 11
    // Stable radix sort for tile IDs (preserves depth order within tiles)
    case tileHistogram = 12
    case tileScanBlocks = 13
    case tileExclusive = 14
    case tileApply = 15
    case tileScatter = 16
    case extractRanges = 17
    case render = 18

    static let count = 19

    var offset: Int {
        rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
    }
}

/// Configuration for depth-first dispatch preparation
struct DepthFirstDispatchConfigSwift {
    var maxGaussians: UInt32
    var maxInstances: UInt32
    var radixBlockSize: UInt32
    var radixGrainSize: UInt32
    var instanceExpansionTGSize: UInt32
    var prefixSumTGSize: UInt32
    var tileRangeTGSize: UInt32
    var tileCount: UInt32
}

/// Encoder that prepares indirect dispatch arguments for depth-first pipeline
final class DepthFirstDispatchEncoder {
    private let pipeline: MTLComputePipelineState
    private let prepareRenderPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "prepareDepthFirstDispatchKernel") else {
            throw RendererError.failedToCreatePipeline("prepareDepthFirstDispatchKernel not found")
        }
        guard let renderFn = library.makeFunction(name: "prepareRenderDispatchKernel") else {
            throw RendererError.failedToCreatePipeline("prepareRenderDispatchKernel not found")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
        self.prepareRenderPipeline = try device.makeComputePipelineState(function: renderFn)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        visibleCount: MTLBuffer,
        totalInstances: MTLBuffer,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer,
        config: DepthFirstDispatchConfigSwift
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PrepareDepthFirstDispatch"

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(visibleCount, offset: 0, index: 0)
        encoder.setBuffer(totalInstances, offset: 0, index: 1)
        encoder.setBuffer(header, offset: 0, index: 2)
        encoder.setBuffer(dispatchArgs, offset: 0, index: 3)

        var cfg = config
        encoder.setBytes(&cfg, length: MemoryLayout<DepthFirstDispatchConfigSwift>.stride, index: 4)

        // Single thread - reads atomics, writes dispatch args
        let threads = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
        encoder.endEncoding()
    }

    /// Prepare render dispatch args based on actual active tile count
    func encodePrepareRenderDispatch(
        commandBuffer: MTLCommandBuffer,
        activeTileCount: MTLBuffer,
        dispatchArgs: MTLBuffer,
        tileCount: UInt32
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PrepareRenderDispatch"

        encoder.setComputePipelineState(prepareRenderPipeline)
        encoder.setBuffer(activeTileCount, offset: 0, index: 0)
        encoder.setBuffer(dispatchArgs, offset: 0, index: 1)

        var tc = tileCount
        encoder.setBytes(&tc, length: MemoryLayout<UInt32>.stride, index: 2)

        let threads = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
        encoder.endEncoding()
    }
}
