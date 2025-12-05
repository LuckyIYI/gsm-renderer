import Metal

/// Encodes projection stage: ProjectStore → Visibility PrefixSum → CompactCount
public final class LocalProjectEncoder {
    // Projection pipelines by (shDegree, useClusterCull)
    private var projectStoreFloatPipelines: [UInt64: MTLComputePipelineState] = [:]
    private var projectStoreHalfPipelines: [UInt64: MTLComputePipelineState] = [:]

    // Compaction (no counting - scatter atomics give count for free)
    private let compactPipeline: MTLComputePipelineState
    private let writeVisibleCountPipeline: MTLComputePipelineState

    // Hierarchical prefix sum
    private let blockReducePipeline: MTLComputePipelineState
    private let singleBlockScanPipeline: MTLComputePipelineState
    private let blockScanPipeline: MTLComputePipelineState

    private let prefixBlockSize = 256

    public init(LocalLibrary: MTLLibrary, mainLibrary: MTLLibrary, device: MTLDevice) throws {
        // Load compaction kernels (LocalCompact - just copies, no counting)
        guard let compactFn = LocalLibrary.makeFunction(name: "LocalCompact"),
              let writeVisibleCountFn = LocalLibrary.makeFunction(name: "LocalWriteVisibleCount") else {
            fatalError("Missing compaction kernels")
        }
        self.compactPipeline = try device.makeComputePipelineState(function: compactFn)
        self.writeVisibleCountPipeline = try device.makeComputePipelineState(function: writeVisibleCountFn)

        // Load hierarchical prefix sum from main library
        guard let reduceFn = mainLibrary.makeFunction(name: "blockReduceKernel"),
              let singleScanFn = mainLibrary.makeFunction(name: "singleBlockScanKernel"),
              let blockScanFn = mainLibrary.makeFunction(name: "blockScanKernel") else {
            fatalError("Missing hierarchical prefix sum kernels")
        }
        self.blockReducePipeline = try device.makeComputePipelineState(function: reduceFn)
        self.singleBlockScanPipeline = try device.makeComputePipelineState(function: singleScanFn)
        self.blockScanPipeline = try device.makeComputePipelineState(function: blockScanFn)

        // Create projection pipeline variants: SH degree (0-3) x cluster cull (true/false)
        for degree: UInt32 in 0...3 {
            for useClusterCull in [false, true] {
                let constantValues = MTLFunctionConstantValues()
                var shDegree = degree
                var cullEnabled = useClusterCull
                constantValues.setConstantValue(&shDegree, type: .uint, index: 0)
                constantValues.setConstantValue(&cullEnabled, type: .bool, index: 1)

                let key = Self.pipelineKey(shDegree: degree, useClusterCull: useClusterCull)

                if let fn = try? LocalLibrary.makeFunction(name: "LocalProjectStoreFloat", constantValues: constantValues) {
                    self.projectStoreFloatPipelines[key] = try? device.makeComputePipelineState(function: fn)
                }
                if let fn = try? LocalLibrary.makeFunction(name: "LocalProjectStoreHalfHalfSh", constantValues: constantValues) {
                    self.projectStoreHalfPipelines[key] = try? device.makeComputePipelineState(function: fn)
                }
            }
        }
    }

    private static func pipelineKey(shDegree: UInt32, useClusterCull: Bool) -> UInt64 {
        return UInt64(shDegree) | (useClusterCull ? 0x100 : 0)
    }

    public static func shDegree(from shComponents: UInt32) -> UInt32 {
        switch shComponents {
        case 0, 1: return 0
        case 2...4: return 1
        case 5...9: return 2
        default: return 3
        }
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        worldGaussians: MTLBuffer,
        harmonics: MTLBuffer,
        camera: CameraUniformsSwift,
        params: TileBinningParams,
        gaussianCount: Int,
        // Output
        tempProjectionBuffer: MTLBuffer,
        visibilityMarks: MTLBuffer,
        visibilityPartialSums: MTLBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        // Options
        useHalfWorld: Bool,
        clusterVisibility: MTLBuffer?,
        clusterSize: UInt32
    ) {
        var cameraUniforms = camera
        var binParams = params
        var gaussianCountU = UInt32(gaussianCount)

        let shDegree = Self.shDegree(from: camera.shComponents)
        let useClusterCull = clusterVisibility != nil
        let key = Self.pipelineKey(shDegree: shDegree, useClusterCull: useClusterCull)

        // === PROJECT + STORE ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_ProjectStore"
            let pipeline = useHalfWorld
                ? projectStoreHalfPipelines[key]!
                : projectStoreFloatPipelines[key]!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(worldGaussians, offset: 0, index: 0)
            encoder.setBuffer(harmonics, offset: 0, index: 1)
            encoder.setBuffer(tempProjectionBuffer, offset: 0, index: 2)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 3)
            encoder.setBytes(&cameraUniforms, length: MemoryLayout<CameraUniformsSwift>.stride, index: 4)
            encoder.setBytes(&binParams, length: MemoryLayout<TileBinningParams>.stride, index: 5)
            if let clusterVis = clusterVisibility {
                encoder.setBuffer(clusterVis, offset: 0, index: 6)
                var cs = clusterSize
                encoder.setBytes(&cs, length: 4, index: 7)
            }
            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            let tg = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === HIERARCHICAL PREFIX SUM ON VISIBILITY ===
        encodeVisibilityPrefixSum(
            commandBuffer: commandBuffer,
            visibilityMarks: visibilityMarks,
            visibilityPartialSums: visibilityPartialSums,
            gaussianCount: gaussianCount
        )

        // === WRITE VISIBLE COUNT TO HEADER ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_WriteVisibleCount"
            encoder.setComputePipelineState(writeVisibleCountPipeline)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBytes(&gaussianCountU, length: 4, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // === COMPACT (no counting - scatter atomics give count for free) ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_Compact"
            encoder.setComputePipelineState(compactPipeline)
            encoder.setBuffer(tempProjectionBuffer, offset: 0, index: 0)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 1)  // prefix sum offsets
            encoder.setBuffer(compactedGaussians, offset: 0, index: 2)
            encoder.setBytes(&gaussianCountU, length: 4, index: 3)
            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            let tg = MTLSize(width: compactPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    private func encodeVisibilityPrefixSum(
        commandBuffer: MTLCommandBuffer,
        visibilityMarks: MTLBuffer,
        visibilityPartialSums: MTLBuffer,
        gaussianCount: Int
    ) {
        let visScanCount = gaussianCount + 1
        let visBlocks = (visScanCount + prefixBlockSize - 1) / prefixBlockSize
        var visScanCountU = UInt32(visScanCount)
        var visBlocksU = UInt32(visBlocks)

        // Block reduce
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_VisBlockReduce"
            encoder.setComputePipelineState(blockReducePipeline)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 0)
            encoder.setBuffer(visibilityPartialSums, offset: 0, index: 1)
            encoder.setBytes(&visScanCountU, length: 4, index: 2)
            let threads = MTLSize(width: visBlocks * prefixBlockSize, height: 1, depth: 1)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Scan block sums
        if visBlocks <= prefixBlockSize {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Local_VisScanBlockSums"
                encoder.setComputePipelineState(singleBlockScanPipeline)
                encoder.setBuffer(visibilityPartialSums, offset: 0, index: 0)
                encoder.setBytes(&visBlocksU, length: 4, index: 1)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }
        } else {
            // Two-level hierarchical scan
            let level2Blocks = (visBlocks + prefixBlockSize - 1) / prefixBlockSize
            let level2Offset = (visBlocks + 1) * 4
            var level2BlocksU = UInt32(level2Blocks)

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Local_VisLevel2Reduce"
                encoder.setComputePipelineState(blockReducePipeline)
                encoder.setBuffer(visibilityPartialSums, offset: 0, index: 0)
                encoder.setBuffer(visibilityPartialSums, offset: level2Offset, index: 1)
                encoder.setBytes(&visBlocksU, length: 4, index: 2)
                let threads = MTLSize(width: level2Blocks * prefixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Local_VisLevel2Scan"
                encoder.setComputePipelineState(singleBlockScanPipeline)
                encoder.setBuffer(visibilityPartialSums, offset: level2Offset, index: 0)
                encoder.setBytes(&level2BlocksU, length: 4, index: 1)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Local_VisLevel1Scan"
                encoder.setComputePipelineState(blockScanPipeline)
                encoder.setBuffer(visibilityPartialSums, offset: 0, index: 0)
                encoder.setBuffer(visibilityPartialSums, offset: 0, index: 1)
                encoder.setBuffer(visibilityPartialSums, offset: level2Offset, index: 2)
                encoder.setBytes(&visBlocksU, length: 4, index: 3)
                let threads = MTLSize(width: level2Blocks * prefixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }
        }

        // Final scan - apply block offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_VisFinalScan"
            encoder.setComputePipelineState(blockScanPipeline)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 0)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 1)
            encoder.setBuffer(visibilityPartialSums, offset: 0, index: 2)
            encoder.setBytes(&visScanCountU, length: 4, index: 3)
            let threads = MTLSize(width: visBlocks * prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }
}
