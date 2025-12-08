import Metal

final class LocalProjectEncoder {
    private var projectStoreFloatPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var projectStoreHalfPipelines: [UInt32: MTLComputePipelineState] = [:]

    private let compactPipeline: MTLComputePipelineState
    private let writeVisibleCountPipeline: MTLComputePipelineState

    private let blockReducePipeline: MTLComputePipelineState
    private let singleBlockScanPipeline: MTLComputePipelineState
    private let blockScanPipeline: MTLComputePipelineState

    private let prefixBlockSize = 256

    init(LocalLibrary: MTLLibrary, mainLibrary: MTLLibrary, device: MTLDevice) throws {
        guard let compactFn = LocalLibrary.makeFunction(name: "localCompact"),
              let writeVisibleCountFn = LocalLibrary.makeFunction(name: "localWriteVisibleCount")
        else {
            throw RendererError.failedToCreatePipeline("Missing compaction kernels")
        }
        self.compactPipeline = try device.makeComputePipelineState(function: compactFn)
        self.writeVisibleCountPipeline = try device.makeComputePipelineState(function: writeVisibleCountFn)

        guard let reduceFn = mainLibrary.makeFunction(name: "blockReduceKernel"),
              let singleScanFn = mainLibrary.makeFunction(name: "singleBlockScanKernel"),
              let blockScanFn = mainLibrary.makeFunction(name: "blockScanKernel")
        else {
            throw RendererError.failedToCreatePipeline("Missing hierarchical prefix sum kernels")
        }
        self.blockReducePipeline = try device.makeComputePipelineState(function: reduceFn)
        self.singleBlockScanPipeline = try device.makeComputePipelineState(function: singleScanFn)
        self.blockScanPipeline = try device.makeComputePipelineState(function: blockScanFn)

        for degree: UInt32 in 0 ... 3 {
            let constantValues = MTLFunctionConstantValues()
            var shDegree = degree
            var cullEnabled = false
            constantValues.setConstantValue(&shDegree, type: .uint, index: 0)
            constantValues.setConstantValue(&cullEnabled, type: .bool, index: 1)

            if let fn = try? LocalLibrary.makeFunction(name: "LocalProjectStoreFloat", constantValues: constantValues) {
                self.projectStoreFloatPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
            if let fn = try? LocalLibrary.makeFunction(name: "LocalProjectStoreHalfHalfSh", constantValues: constantValues) {
                self.projectStoreHalfPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
        }
    }

    static func shDegree(from shComponents: UInt32) -> UInt32 {
        switch shComponents {
        case 0, 1: 0
        case 2 ... 4: 1
        case 5 ... 9: 2
        default: 3
        }
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        worldGaussians: MTLBuffer,
        harmonics: MTLBuffer,
        camera: CameraUniformsSwift,
        params: TileBinningParams,
        gaussianCount: Int,
        tempProjectionBuffer: MTLBuffer,
        visibilityMarks: MTLBuffer,
        visibilityPartialSums: MTLBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        useHalfWorld: Bool
    ) {
        var cameraUniforms = camera
        var binParams = params
        var gaussianCountU = UInt32(gaussianCount)

        let shDegree = Self.shDegree(from: camera.shComponents)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_ProjectStore"
            let pipeline: MTLComputePipelineState = if useHalfWorld {
                self.projectStoreHalfPipelines[shDegree]
                    ?? self.projectStoreHalfPipelines[0]
                    ?? self.projectStoreFloatPipelines[shDegree]
                    ?? self.projectStoreFloatPipelines[0]!
            } else {
                self.projectStoreFloatPipelines[shDegree]
                    ?? self.projectStoreFloatPipelines[0]!
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(worldGaussians, offset: 0, index: 0)
            encoder.setBuffer(harmonics, offset: 0, index: 1)
            encoder.setBuffer(tempProjectionBuffer, offset: 0, index: 2)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 3)
            encoder.setBytes(&cameraUniforms, length: MemoryLayout<CameraUniformsSwift>.stride, index: 4)
            encoder.setBytes(&binParams, length: MemoryLayout<TileBinningParams>.stride, index: 5)
            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            let tg = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        self.encodeVisibilityPrefixSum(
            commandBuffer: commandBuffer,
            visibilityMarks: visibilityMarks,
            visibilityPartialSums: visibilityPartialSums,
            gaussianCount: gaussianCount
        )

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_WriteVisibleCount"
            encoder.setComputePipelineState(self.writeVisibleCountPipeline)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBytes(&gaussianCountU, length: 4, index: 2)
            encoder.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_Compact"
            encoder.setComputePipelineState(self.compactPipeline)
            encoder.setBuffer(tempProjectionBuffer, offset: 0, index: 0)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 1)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 2)
            encoder.setBytes(&gaussianCountU, length: 4, index: 3)
            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            let tg = MTLSize(width: self.compactPipeline.threadExecutionWidth, height: 1, depth: 1)
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
        let visBlocks = (visScanCount + self.prefixBlockSize - 1) / self.prefixBlockSize
        var visScanCountU = UInt32(visScanCount)
        var visBlocksU = UInt32(visBlocks)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_VisBlockReduce"
            encoder.setComputePipelineState(self.blockReducePipeline)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 0)
            encoder.setBuffer(visibilityPartialSums, offset: 0, index: 1)
            encoder.setBytes(&visScanCountU, length: 4, index: 2)
            let threads = MTLSize(width: visBlocks * self.prefixBlockSize, height: 1, depth: 1)
            let tg = MTLSize(width: self.prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        if visBlocks <= self.prefixBlockSize {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Local_VisScanBlockSums"
                encoder.setComputePipelineState(self.singleBlockScanPipeline)
                encoder.setBuffer(visibilityPartialSums, offset: 0, index: 0)
                encoder.setBytes(&visBlocksU, length: 4, index: 1)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: self.prefixBlockSize, height: 1, depth: 1)
                )
                encoder.endEncoding()
            }
        } else {
            let level2Blocks = (visBlocks + self.prefixBlockSize - 1) / self.prefixBlockSize
            let level2Offset = (visBlocks + 1) * 4
            var level2BlocksU = UInt32(level2Blocks)

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Local_VisLevel2Reduce"
                encoder.setComputePipelineState(self.blockReducePipeline)
                encoder.setBuffer(visibilityPartialSums, offset: 0, index: 0)
                encoder.setBuffer(visibilityPartialSums, offset: level2Offset, index: 1)
                encoder.setBytes(&visBlocksU, length: 4, index: 2)
                let threads = MTLSize(width: level2Blocks * self.prefixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: self.prefixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Local_VisLevel2Scan"
                encoder.setComputePipelineState(self.singleBlockScanPipeline)
                encoder.setBuffer(visibilityPartialSums, offset: level2Offset, index: 0)
                encoder.setBytes(&level2BlocksU, length: 4, index: 1)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: self.prefixBlockSize, height: 1, depth: 1)
                )
                encoder.endEncoding()
            }

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Local_VisLevel1Scan"
                encoder.setComputePipelineState(self.blockScanPipeline)
                encoder.setBuffer(visibilityPartialSums, offset: 0, index: 0)
                encoder.setBuffer(visibilityPartialSums, offset: 0, index: 1)
                encoder.setBuffer(visibilityPartialSums, offset: level2Offset, index: 2)
                encoder.setBytes(&visBlocksU, length: 4, index: 3)
                let threads = MTLSize(width: level2Blocks * self.prefixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: self.prefixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_VisFinalScan"
            encoder.setComputePipelineState(self.blockScanPipeline)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 0)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 1)
            encoder.setBuffer(visibilityPartialSums, offset: 0, index: 2)
            encoder.setBytes(&visScanCountU, length: 4, index: 3)
            let threads = MTLSize(width: visBlocks * self.prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: self.prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }
}
