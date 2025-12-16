import Metal
import RendererTypes

/// Buffers required for radix sort scratch space
struct RadixSortBuffers {
    let histogram: MTLBuffer
    let blockSums: MTLBuffer
    let scannedHistogram: MTLBuffer
    let scratchKeys: MTLBuffer
    let scratchIndices: MTLBuffer
}

/// Key precision for depth radix sort.
/// - Note: Both modes operate on `uint` key buffers; `.bits16` stores a 16-bit quantized key in the low 16 bits and runs 2 passes.
public enum RadixSortKeyPrecision: Sendable {
    case bits16 // 2 passes (low 16 bits)
    case bits32 // 4 passes (full 32 bits)

    var numPasses: Int {
        switch self {
        case .bits16: 2
        case .bits32: 4
        }
    }
}

/// Radix sort encoder for depth-based Gaussian sorting.
/// Supports both 16-bit (2 passes) and 32-bit (4 passes) key variants.
/// Uses indirect dispatch based on GPU-computed visible count.
final class DepthRadixSortEncoder {
    static let blockSize: Int = 256
    static let grainSize: Int = 4

    private let histogramPipeline: MTLComputePipelineState
    private let scanBlocksPipeline: MTLComputePipelineState
    private let exclusiveScanPipeline: MTLComputePipelineState
    private let applyOffsetsPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState

    let precision: RadixSortKeyPrecision

    /// Initialize with specified key precision
    init(device: MTLDevice, library: MTLLibrary, precision: RadixSortKeyPrecision = .bits32) throws {
        self.precision = precision
        // Depth sort buffers are `uint` (even for .bits16, which uses the low 16 bits), so always bind the 32-bit kernels.
        let suffix = "32"

        guard let histFn = library.makeFunction(name: "depthSortHistogramKernel\(suffix)"),
              let scanFn = library.makeFunction(name: "depthSortScanBlocksKernel"),
              let exclusiveFn = library.makeFunction(name: "depthSortExclusiveScanKernel"),
              let applyFn = library.makeFunction(name: "depthSortApplyOffsetsKernel"),
              let scatterFn = library.makeFunction(name: "depthSortScatterKernel\(suffix)")
        else {
            throw RendererError.failedToCreatePipeline("Depth radix sort kernels not found")
        }

        self.histogramPipeline = try device.makeComputePipelineState(function: histFn)
        self.scanBlocksPipeline = try device.makeComputePipelineState(function: scanFn)
        self.exclusiveScanPipeline = try device.makeComputePipelineState(function: exclusiveFn)
        self.applyOffsetsPipeline = try device.makeComputePipelineState(function: applyFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)
    }

    /// Dispatch offsets for radix sort stages
    struct DispatchOffsets {
        let histogram: Int
        let scanBlocks: Int
        let exclusive: Int
        let apply: Int
        let scatter: Int

        /// Default offsets (all zero, for separate dispatch buffers)
        static let zero = DispatchOffsets(histogram: 0, scanBlocks: 0, exclusive: 0, apply: 0, scatter: 0)

        /// Create offsets from slot indices (multiplied by stride)
        static func fromSlots(
            histogram: Int,
            scanBlocks: Int,
            exclusive: Int,
            apply: Int,
            scatter: Int,
            stride: Int
        ) -> DispatchOffsets {
            DispatchOffsets(
                histogram: histogram * stride,
                scanBlocks: scanBlocks * stride,
                exclusive: exclusive * stride,
                apply: apply * stride,
                scatter: scatter * stride
            )
        }
    }

    /// Encode a radix sort on depth keys with corresponding index reordering.
    /// Number of passes depends on `precision` (2 for `.bits16`, 4 for `.bits32`).
    /// - Parameters:
    ///   - commandBuffer: Command buffer to encode into
    ///   - depthKeys: Input/output buffer of `uint` depth keys
    ///   - sortedIndices: Input/output buffer of UInt32 indices
    ///   - sortBuffers: Scratch buffers for histogram and intermediate results
    ///   - sortHeader: DepthFirstHeader buffer containing paddedCount and numBlocks
    ///   - sortDispatch: Indirect dispatch arguments for histogram/scatter
    ///   - scanDispatch: Indirect dispatch arguments for scan operations
    ///   - label: Label prefix for encoder debugging
    func encode(
        commandBuffer: MTLCommandBuffer,
        depthKeys: MTLBuffer,
        sortedIndices: MTLBuffer,
        sortBuffers: RadixSortBuffers,
        sortHeader: MTLBuffer,
        sortDispatch: MTLBuffer,
        scanDispatch: MTLBuffer,
        label: String
    ) {
        self.encode(
            commandBuffer: commandBuffer,
            depthKeys: depthKeys,
            sortedIndices: sortedIndices,
            sortBuffers: sortBuffers,
            sortHeader: sortHeader,
            dispatchBuffer: sortDispatch,
            scanDispatchBuffer: scanDispatch,
            offsets: .zero,
            label: label
        )
    }

    /// Encode a radix sort with configurable dispatch offsets (for unified dispatch buffer layouts)
    /// - Parameters:
    ///   - commandBuffer: Command buffer to encode into
    ///   - depthKeys: Input/output buffer of depth keys
    ///   - sortedIndices: Input/output buffer of UInt32 indices
    ///   - sortBuffers: Scratch buffers for histogram and intermediate results
    ///   - sortHeader: DepthFirstHeader buffer containing paddedCount and numBlocks
    ///   - dispatchBuffer: Indirect dispatch arguments buffer
    ///   - scanDispatchBuffer: Optional separate scan dispatch buffer (if nil, uses dispatchBuffer)
    ///   - offsets: Byte offsets into dispatch buffer for each kernel stage
    ///   - label: Label prefix for encoder debugging
    func encode(
        commandBuffer: MTLCommandBuffer,
        depthKeys: MTLBuffer,
        sortedIndices: MTLBuffer,
        sortBuffers: RadixSortBuffers,
        sortHeader: MTLBuffer,
        dispatchBuffer: MTLBuffer,
        scanDispatchBuffer: MTLBuffer? = nil,
        offsets: DispatchOffsets,
        label: String
    ) {
        let scanDispatch = scanDispatchBuffer ?? dispatchBuffer
        let numPasses = self.precision.numPasses

        for pass in 0 ..< numPasses {
            let isEvenPass = pass % 2 == 0
            let inputKeys = isEvenPass ? depthKeys : sortBuffers.scratchKeys
            let inputIndices = isEvenPass ? sortedIndices : sortBuffers.scratchIndices
            let outputKeys = isEvenPass ? sortBuffers.scratchKeys : depthKeys
            let outputIndices = isEvenPass ? sortBuffers.scratchIndices : sortedIndices

            self.encodeHistogram(
                commandBuffer: commandBuffer,
                inputKeys: inputKeys,
                histogram: sortBuffers.histogram,
                sortHeader: sortHeader,
                sortDispatch: dispatchBuffer,
                dispatchOffset: offsets.histogram,
                pass: pass,
                label: label
            )

            self.encodeScanBlocks(
                commandBuffer: commandBuffer,
                histogram: sortBuffers.histogram,
                blockSums: sortBuffers.blockSums,
                sortHeader: sortHeader,
                scanDispatch: scanDispatch,
                dispatchOffset: offsets.scanBlocks,
                pass: pass,
                label: label
            )

            self.encodeExclusiveScan(
                commandBuffer: commandBuffer,
                blockSums: sortBuffers.blockSums,
                scannedHistogram: sortBuffers.scannedHistogram,
                sortHeader: sortHeader,
                pass: pass,
                label: label
            )

            self.encodeApplyOffsets(
                commandBuffer: commandBuffer,
                histogram: sortBuffers.histogram,
                blockSums: sortBuffers.blockSums,
                scannedHistogram: sortBuffers.scannedHistogram,
                sortHeader: sortHeader,
                scanDispatch: scanDispatch,
                dispatchOffset: offsets.apply,
                pass: pass,
                label: label
            )

            self.encodeScatter(
                commandBuffer: commandBuffer,
                inputKeys: inputKeys,
                inputIndices: inputIndices,
                scannedHistogram: sortBuffers.scannedHistogram,
                outputKeys: outputKeys,
                outputIndices: outputIndices,
                sortHeader: sortHeader,
                sortDispatch: dispatchBuffer,
                dispatchOffset: offsets.scatter,
                pass: pass,
                label: label
            )
        }
    }

    // MARK: - Private Encoding Methods

    private func encodeHistogram(
        commandBuffer: MTLCommandBuffer,
        inputKeys: MTLBuffer,
        histogram: MTLBuffer,
        sortHeader: MTLBuffer,
        sortDispatch: MTLBuffer,
        dispatchOffset: Int,
        pass: Int,
        label: String
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "\(label)_RadixHist_Pass\(pass)"
        encoder.setComputePipelineState(self.histogramPipeline)
        encoder.setBuffer(inputKeys, offset: 0, index: 0)
        encoder.setBuffer(histogram, offset: 0, index: 1)
        var passVal = UInt32(pass)
        encoder.setBytes(&passVal, length: 4, index: 3)
        encoder.setBuffer(sortHeader, offset: 0, index: 4)
        encoder.dispatchThreadgroups(
            indirectBuffer: sortDispatch,
            indirectBufferOffset: dispatchOffset,
            threadsPerThreadgroup: MTLSize(width: Self.blockSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    private func encodeScanBlocks(
        commandBuffer: MTLCommandBuffer,
        histogram: MTLBuffer,
        blockSums: MTLBuffer,
        sortHeader: MTLBuffer,
        scanDispatch: MTLBuffer,
        dispatchOffset: Int,
        pass: Int,
        label: String
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "\(label)_RadixScan_Pass\(pass)"
        encoder.setComputePipelineState(self.scanBlocksPipeline)
        encoder.setBuffer(histogram, offset: 0, index: 0)
        encoder.setBuffer(blockSums, offset: 0, index: 1)
        encoder.setBuffer(sortHeader, offset: 0, index: 2)
        encoder.dispatchThreadgroups(
            indirectBuffer: scanDispatch,
            indirectBufferOffset: dispatchOffset,
            threadsPerThreadgroup: MTLSize(width: Self.blockSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    private func encodeExclusiveScan(
        commandBuffer: MTLCommandBuffer,
        blockSums: MTLBuffer,
        scannedHistogram _: MTLBuffer,
        sortHeader: MTLBuffer,
        pass: Int,
        label: String
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "\(label)_RadixExcl_Pass\(pass)"
        encoder.setComputePipelineState(self.exclusiveScanPipeline)
        encoder.setBuffer(blockSums, offset: 0, index: 0)
        encoder.setBuffer(sortHeader, offset: 0, index: 1)
        encoder.setThreadgroupMemoryLength(Self.blockSize * MemoryLayout<UInt32>.stride, index: 0)
        encoder.dispatchThreads(
            MTLSize(width: 256, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    private func encodeApplyOffsets(
        commandBuffer: MTLCommandBuffer,
        histogram: MTLBuffer,
        blockSums: MTLBuffer,
        scannedHistogram: MTLBuffer,
        sortHeader: MTLBuffer,
        scanDispatch: MTLBuffer,
        dispatchOffset: Int,
        pass: Int,
        label: String
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "\(label)_RadixApply_Pass\(pass)"
        encoder.setComputePipelineState(self.applyOffsetsPipeline)
        encoder.setBuffer(histogram, offset: 0, index: 0)
        encoder.setBuffer(blockSums, offset: 0, index: 1)
        encoder.setBuffer(scannedHistogram, offset: 0, index: 2)
        encoder.setBuffer(sortHeader, offset: 0, index: 3)
        encoder.setThreadgroupMemoryLength(Self.blockSize * MemoryLayout<UInt32>.stride, index: 0)
        encoder.dispatchThreadgroups(
            indirectBuffer: scanDispatch,
            indirectBufferOffset: dispatchOffset,
            threadsPerThreadgroup: MTLSize(width: Self.blockSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    private func encodeScatter(
        commandBuffer: MTLCommandBuffer,
        inputKeys: MTLBuffer,
        inputIndices: MTLBuffer,
        scannedHistogram: MTLBuffer,
        outputKeys: MTLBuffer,
        outputIndices: MTLBuffer,
        sortHeader: MTLBuffer,
        sortDispatch: MTLBuffer,
        dispatchOffset: Int,
        pass: Int,
        label: String
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "\(label)_RadixScatter_Pass\(pass)"
        encoder.setComputePipelineState(self.scatterPipeline)
        encoder.setBuffer(outputKeys, offset: 0, index: 0)
        encoder.setBuffer(inputKeys, offset: 0, index: 1)
        encoder.setBuffer(outputIndices, offset: 0, index: 2)
        encoder.setBuffer(inputIndices, offset: 0, index: 3)
        encoder.setBuffer(scannedHistogram, offset: 0, index: 5)
        var passVal = UInt32(pass)
        encoder.setBytes(&passVal, length: 4, index: 6)
        encoder.setBuffer(sortHeader, offset: 0, index: 7)
        encoder.dispatchThreadgroups(
            indirectBuffer: sortDispatch,
            indirectBufferOffset: dispatchOffset,
            threadsPerThreadgroup: MTLSize(width: Self.blockSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }
}
