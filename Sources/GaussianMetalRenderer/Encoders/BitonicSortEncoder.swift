import Metal

final class BitonicSortEncoder {
    private let firstPassPipeline: MTLComputePipelineState
    private let generalPassPipeline: MTLComputePipelineState
    private let finalPassPipeline: MTLComputePipelineState
    private let icbFillPipeline: MTLComputePipelineState
    private let icbArgEncoder: MTLArgumentEncoder
    private let device: MTLDevice

    private var icb: MTLIndirectCommandBuffer?
    private var icbArgBuffer: MTLBuffer?
    private var passTypeBuffer: MTLBuffer?
    private var firstParamsBuffer: MTLBuffer?
    private var generalParamsBuffer: MTLBuffer?
    private var debugParamsBuffer: MTLBuffer?
    private let firstParamStride = 256
    private let generalParamStride = 256
    private var icbMaxCommands: Int = 0
    private var icbMaxPadded: Int = 0

    let unitSize: Int

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        guard
            let firstFn = library.makeFunction(name: "bitonicSortFirstPass"),
            let generalFn = library.makeFunction(name: "bitonicSortGeneralPass"),
            let finalFn = library.makeFunction(name: "bitonicSortFinalPass"),
            let icbFillFn = library.makeFunction(name: "encodeBitonicICB")
        else {
            fatalError("Bitonic sort functions missing")
        }
        
        func makeICBSafePipeline(_ fn: MTLFunction) throws -> MTLComputePipelineState {
            let desc = MTLComputePipelineDescriptor()
            desc.computeFunction = fn
            desc.supportIndirectCommandBuffers = true
            return try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
        }
        
        self.firstPassPipeline = try makeICBSafePipeline(firstFn)
        self.generalPassPipeline = try makeICBSafePipeline(generalFn)
        self.finalPassPipeline = try makeICBSafePipeline(finalFn)
        self.icbFillPipeline = try makeICBSafePipeline(icbFillFn)
        self.icbArgEncoder = icbFillFn.makeArgumentEncoder(bufferIndex: 0)
        
        let bitonicMax = min(firstPassPipeline.maxTotalThreadsPerThreadgroup, 512)
        self.unitSize = BitonicSortEncoder.largestPowerOfTwo(lessThanOrEqualTo: bitonicMax)
    }
    
    static func largestPowerOfTwo(lessThanOrEqualTo value: Int) -> Int {
        var v = value
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return (v + 1) >> 1
    }
    
    private func log2PowerOfTwo(_ value: Int) -> Int {
        return Int.bitWidth - value.leadingZeroBitCount - 1
    }
    
    private func maxPassCount(for paddedCapacity: Int) -> Int {
        let m = self.log2PowerOfTwo(paddedCapacity)
        let u = self.log2PowerOfTwo(self.unitSize)
        let g = max(0, m - u) // number of stages beyond the first
        // total = 1(first) + g finals + g*(g-1)/2 generals
        return 1 + g + (g * (g - 1)) / 2
    }
    
    private func buildScheduleForCapacity(_ paddedCapacity: Int) {
        guard let icb = self.icb,
              let passTypeBuffer = self.passTypeBuffer,
              let firstParamsBuffer = self.firstParamsBuffer,
              let generalParamsBuffer = self.generalParamsBuffer else {
            return
        }

        let maxCommands = self.icbMaxCommands
        let passTypes = passTypeBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCommands)

        func writeFirstParam(_ slot: Int, value: UInt32) {
            let raw = firstParamsBuffer.contents().advanced(by: slot * firstParamStride)
            raw.storeBytes(of: value, as: UInt32.self)
        }

        func writeGeneralParam(_ slot: Int, value: SIMD2<UInt32>) {
            let raw = generalParamsBuffer.contents().advanced(by: slot * generalParamStride)
            raw.storeBytes(of: value, as: SIMD2<UInt32>.self)
        }

        var slot = 0

        if slot < maxCommands {
            let cmd = icb.indirectComputeCommandAt(slot)
            cmd.setComputePipelineState(self.firstPassPipeline)
            cmd.setKernelBuffer(firstParamsBuffer, offset: slot * firstParamStride, at: 3)
            cmd.setThreadgroupMemoryLength(self.unitSize * 16, index: 0)
            cmd.setThreadgroupMemoryLength(self.unitSize * 8, index: 1)
            cmd.setBarrier()
            writeFirstParam(slot, value: UInt32(min(self.unitSize, paddedCapacity / 2)))
            passTypes[slot] = 0
            slot += 1
        }

        var k = self.unitSize << 1
        while k <= paddedCapacity && slot < maxCommands {
            var j = k >> 1
            while j > 0 && slot < maxCommands {
                let params = SIMD2<UInt32>(UInt32(k), UInt32(j))
                if self.unitSize < j {
                    let cmd = icb.indirectComputeCommandAt(slot)
                    cmd.setComputePipelineState(self.generalPassPipeline)
                    cmd.setKernelBuffer(generalParamsBuffer, offset: slot * generalParamStride, at: 3)
                    cmd.setBarrier()
                    writeGeneralParam(slot, value: params)
                    passTypes[slot] = 1
                    slot += 1
                    j >>= 1
                } else {
                    let cmd = icb.indirectComputeCommandAt(slot)
                    cmd.setComputePipelineState(self.finalPassPipeline)
                    cmd.setKernelBuffer(generalParamsBuffer, offset: slot * generalParamStride, at: 3)
                    cmd.setThreadgroupMemoryLength((self.unitSize * MemoryLayout<SIMD2<UInt32>>.stride) << 1, index: 0)
                    cmd.setThreadgroupMemoryLength((self.unitSize * MemoryLayout<Int32>.stride) << 1, index: 1)
                    cmd.setBarrier()
                    writeGeneralParam(slot, value: params)
                    passTypes[slot] = 2
                    slot += 1
                    j = 0
                }
            }
            k <<= 1
        }

        while slot < maxCommands {
            icb.indirectComputeCommandAt(slot).setComputePipelineState(self.firstPassPipeline)
            passTypes[slot] = 3
            slot += 1
        }
    }
    
    private func ensureICB(maxPaddedCapacity: Int) {
        if let _ = self.icb, maxPaddedCapacity <= self.icbMaxPadded {
            return
        }

        let commandCount = self.maxPassCount(for: maxPaddedCapacity)

        let descriptor = MTLIndirectCommandBufferDescriptor()
        descriptor.commandTypes = .concurrentDispatch // matches concurrent_dispatch_threadgroups
        descriptor.inheritBuffers = false
        descriptor.inheritPipelineState = false
        descriptor.maxKernelBufferBindCount = 16

        guard
            let icb = device.makeIndirectCommandBuffer(
                descriptor: descriptor,
                maxCommandCount: commandCount,
                options: .storageModePrivate
            )
        else {
            fatalError("Failed to create indirect command buffer for bitonic")
        }

        self.icb = icb
        self.icbMaxCommands = commandCount
        self.icbMaxPadded   = maxPaddedCapacity

        // Argument buffer for the ICB container
        let argBuffer = device.makeBuffer(
            length: icbArgEncoder.encodedLength,
            options: .storageModePrivate
        )!
        icbArgEncoder.setArgumentBuffer(argBuffer, offset: 0)
        icbArgEncoder.setIndirectCommandBuffer(icb, index: 0)
        self.icbArgBuffer = argBuffer

        let firstBytes = commandCount * firstParamStride
        let generalBytes = commandCount * generalParamStride
        self.firstParamsBuffer = device.makeBuffer(length: firstBytes, options: .storageModeShared)
        self.generalParamsBuffer = device.makeBuffer(length: generalBytes, options: .storageModeShared)
        self.debugParamsBuffer = device.makeBuffer(length: commandCount * MemoryLayout<SIMD4<UInt32>>.stride,
                                                   options: .storageModeShared)
        self.passTypeBuffer = device.makeBuffer(length: commandCount * MemoryLayout<UInt32>.stride,
                                                options: .storageModeShared)

        // Build static schedule (pipelines, params, pass types) for this capacity
        self.buildScheduleForCapacity(maxPaddedCapacity)
    }
    func encode(
        commandBuffer: MTLCommandBuffer,
        sortKeys: MTLBuffer,
        sortedIndices: MTLBuffer,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer,
        offsets: (first: Int, general: Int, final: Int),
        paddedCapacity: Int,
        debugParams: MTLBuffer? = nil
    ) {
        guard paddedCapacity > 1 else { return }
        self.ensureICB(maxPaddedCapacity: paddedCapacity)
        guard let icb = self.icb,
              let icbArgBuffer = self.icbArgBuffer,
              let passTypeBuffer = self.passTypeBuffer else { return }

        let commandsToRun = self.icbMaxCommands

        // Bind per-pass buffers for this frame (small loop, no k/j math on CPU).
        for slot in 0..<commandsToRun {
            let cmd = icb.indirectComputeCommandAt(slot)
            cmd.setKernelBuffer(sortKeys, offset: 0, at: 0)
            cmd.setKernelBuffer(sortedIndices, offset: 0, at: 1)
            cmd.setKernelBuffer(header, offset: 0, at: 2)
        }

        // Mask dispatch counts on GPU (single thread)
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "BitonicICBFill"
            enc.setComputePipelineState(self.icbFillPipeline)
            enc.setBuffer(icbArgBuffer,    offset: 0, index: 0)
            enc.setBuffer(header,          offset: 0, index: 1)
            enc.setBuffer(dispatchArgs,    offset: 0, index: 2)
            enc.setBuffer(passTypeBuffer,  offset: 0, index: 3)
            if let firstParams = self.firstParamsBuffer {
                enc.setBuffer(firstParams, offset: 0, index: 9)
                enc.useResource(firstParams, usage: .write)
            }
            var u  = UInt32(self.unitSize)
            var first = UInt32(offsets.first)
            var gen   = UInt32(offsets.general)
            var fin   = UInt32(offsets.final)
            var maxCmds = UInt32(commandsToRun)
            enc.setBytes(&u,       length: 4, index: 4)
            enc.setBytes(&maxCmds, length: 4, index: 5)
            enc.setBytes(&first,   length: 4, index: 6)
            enc.setBytes(&gen,     length: 4, index: 7)
            enc.setBytes(&fin,     length: 4, index: 8)
            if let dbg = debugParams ?? self.debugParamsBuffer {
                enc.setBuffer(dbg, offset: 0, index: 10)
                enc.useResource(dbg, usage: .write)
            }

            enc.useResource(icb,          usage: .write)
            enc.useResource(dispatchArgs, usage: .read)
            enc.useResource(header,       usage: .read)
            enc.useResource(passTypeBuffer, usage: .read)

            let one = MTLSize(width: 1, height: 1, depth: 1)
            enc.dispatchThreads(one, threadsPerThreadgroup: one)
            enc.endEncoding()
        }

        // Optional optimize
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "BitonicICBOptimize"
            blit.optimizeIndirectCommandBuffer(icb, range: 0..<commandsToRun)
            blit.endEncoding()
        }

        // Execute ICB
        if let exec = commandBuffer.makeComputeCommandEncoder() {
            exec.label = "BitonicICBExecute"
            if let firstParams = self.firstParamsBuffer { exec.useResource(firstParams, usage: .read) }
            if let generalParams = self.generalParamsBuffer { exec.useResource(generalParams, usage: .read) }
            exec.useResource(sortKeys,      usage: [.read, .write])
            exec.useResource(sortedIndices, usage: [.read, .write])
            exec.useResource(header,        usage: .read)
            exec.useResource(dispatchArgs,  usage: .read)
            exec.useResource(icb,           usage: .read)
            exec.executeCommandsInBuffer(icb, range: 0..<commandsToRun)
            exec.endEncoding()
        }
    }


    func encodeDirect(
        commandBuffer: MTLCommandBuffer,
        sortKeys: MTLBuffer,
        sortedIndices: MTLBuffer,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer,
        offsets: (first: Int, general: Int, final: Int),
        paddedCapacity: Int
    ) {
        guard paddedCapacity > 1 else { return }
        let gridSize = paddedCapacity / 2
        guard gridSize > 0 else { return }
        
        // First Pass
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "BitonicFirst"
            encoder.setComputePipelineState(self.firstPassPipeline)
            encoder.setBuffer(sortKeys, offset: 0, index: 0)
            encoder.setBuffer(sortedIndices, offset: 0, index: 1)
            encoder.setBuffer(header, offset: 0, index: 2)
            
            var logicalUnitSize = UInt32(min(self.unitSize, paddedCapacity / 2))
            encoder.setBytes(&logicalUnitSize, length: 4, index: 3)
            
            encoder.setThreadgroupMemoryLength(self.unitSize * 16, index: 0)
            encoder.setThreadgroupMemoryLength(self.unitSize * 8, index: 1)
            
            let tg = MTLSize(width: self.unitSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.first,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
        
        // Subsequent Passes
        var k = UInt32(self.unitSize << 1)
        while k <= UInt32(paddedCapacity) {
            var j = k >> 1
            while j > 0 {
                let params = SIMD2<UInt32>(k, j)
                if UInt32(self.unitSize) < j {
                    // General Pass
                    if let encoder = commandBuffer.makeComputeCommandEncoder() {
                        encoder.label = "BitonicGeneral"
                        var p = params
                        encoder.setComputePipelineState(self.generalPassPipeline)
                        encoder.setBuffer(sortKeys, offset: 0, index: 0)
                        encoder.setBuffer(sortedIndices, offset: 0, index: 1)
                        encoder.setBuffer(header, offset: 0, index: 2)
                        encoder.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 3)
                        let tg = MTLSize(width: self.unitSize, height: 1, depth: 1)
                        encoder.dispatchThreadgroups(
                            indirectBuffer: dispatchArgs,
                            indirectBufferOffset: offsets.general,
                            threadsPerThreadgroup: tg
                        )
                        encoder.endEncoding()
                    }
                    j >>= 1
                } else {
                    // Final Pass
                    if let encoder = commandBuffer.makeComputeCommandEncoder() {
                        encoder.label = "BitonicFinal"
                        var p = params
                        encoder.setComputePipelineState(self.finalPassPipeline)
                        encoder.setBuffer(sortKeys, offset: 0, index: 0)
                        encoder.setBuffer(sortedIndices, offset: 0, index: 1)
                        encoder.setBuffer(header, offset: 0, index: 2)
                        encoder.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 3)
                        
                        encoder.setThreadgroupMemoryLength(
                            (self.unitSize * MemoryLayout<SIMD2<UInt32>>.stride) << 1,
                            index: 0
                        )
                        encoder.setThreadgroupMemoryLength(
                            (self.unitSize * MemoryLayout<Int32>.stride) << 1,
                            index: 1
                        )
                        
                        let tg = MTLSize(width: self.unitSize, height: 1, depth: 1)
                        encoder.dispatchThreadgroups(
                            indirectBuffer: dispatchArgs,
                            indirectBufferOffset: offsets.final,
                            threadsPerThreadgroup: tg
                        )
                        encoder.endEncoding()
                    }
                    j = 0
                }
            }
            k <<= 1
        }
    }

    // Debug helpers
    var debugFirstParamsBuffer: MTLBuffer? { firstParamsBuffer }
    var debugGeneralParamsBuffer: MTLBuffer? { generalParamsBuffer }
}
