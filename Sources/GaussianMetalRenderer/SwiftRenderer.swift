import Foundation
import simd
@preconcurrency import Metal


// MARK: - Frame Resources

final class FrameResources {
    // Tile Builder Buffers
    var boundsBuffer: MTLBuffer?
    var coverageBuffer: MTLBuffer?
    var offsetsBuffer: MTLBuffer?
    var partialSumsBuffer: MTLBuffer?
    var scatterDispatchBuffer: MTLBuffer?
    var tileAssignmentHeader: MTLBuffer?
    var tileIndices: MTLBuffer?
    var tileIds: MTLBuffer?
    var tileAssignmentMaxAssignments: Int = 0
    // Capacity-based power-of-two padding for buffer sizing (GPU will set actual paddedCount per frame).
    var tileAssignmentPaddedCapacity: Int = 1
    
    // Ordered Buffers
    var orderedHeaders: MTLBuffer?
    var packedMeans: MTLBuffer?
    var packedConics: MTLBuffer?
    var packedColors: MTLBuffer?
    var packedOpacities: MTLBuffer?
    var packedDepths: MTLBuffer?
    var activeTileIndices: MTLBuffer?
    var activeTileCount: MTLBuffer?
    
    // Sort Buffers
    var sortKeys: MTLBuffer?
    var sortedIndices: MTLBuffer?
    
    // Radix Buffers
    var radixHistogram: MTLBuffer?
    var radixBlockSums: MTLBuffer?
    var radixScannedHistogram: MTLBuffer?
    var radixFusedKeys: MTLBuffer?
    var radixKeysScratch: MTLBuffer?
    var radixPayloadScratch: MTLBuffer?
    
    // Dispatch Args (Per-frame to avoid race on indirect dispatch)
    var dispatchArgs: MTLBuffer?
    
    // Output Buffers
    var outputBuffers: RenderOutputBuffers?
    
    init(device: MTLDevice) {
        let dispatchLength = 4096
        if let buf = device.makeBuffer(length: dispatchLength, options: .storageModePrivate) {
            buf.label = "FrameDispatchArgs"
            self.dispatchArgs = buf
        }
    }
}

// MARK: - Renderer

public final class Renderer: @unchecked Sendable {
    public static let shared = Renderer()

    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    private var nextFrameCapturePath: String?
    
    // Encoders
    let tileBoundsEncoder: TileBoundsEncoder
    let coverageEncoder: CoverageEncoder
    let scatterEncoder: ScatterEncoder
    let sortKeyGenEncoder: SortKeyGenEncoder
    let bitonicSortEncoder: BitonicSortEncoder
    let radixSortEncoder: RadixSortEncoder
    let packEncoder: PackEncoder
    let renderEncoder: RenderEncoder
    let projectEncoder: ProjectEncoder
    let dispatchEncoder: DispatchEncoder
    
    // Frame Resources (Single Buffering for now)
    private var frameResources: [FrameResources?] = []
    private var frameInUse: [Bool] = []
    private var frameCursor: Int = 0
    private let maxInFlightFrames = 1 // Enforce single buffering initially
    private let frameLock = NSLock()
    
    // World Buffers (Input - Shared/Transient)
    private var worldPositionsCache: MTLBuffer?
    private var worldScalesCache: MTLBuffer?
    private var worldRotationsCache: MTLBuffer?
    private var worldHarmonicsCache: MTLBuffer?
    private var worldOpacitiesCache: MTLBuffer?
    
    // Gaussian input (projection stage) buffers (Shared/Transient)
    private var gaussianMeansCache: MTLBuffer?
    private var gaussianRadiiCache: MTLBuffer?
    private var gaussianMaskCache: MTLBuffer?
    private var gaussianDepthsCache: MTLBuffer?
    private var gaussianConicsCache: MTLBuffer?
        private var gaussianColorsCache: MTLBuffer?
        private var gaussianOpacitiesCache: MTLBuffer?
        
    private let sortAlgorithm: SortAlgorithm = .radix
    private let precision: Precision
    private let effectivePrecision: Precision
    public var precisionSetting: Precision { self.effectivePrecision }
            
    private enum SortAlgorithm {
        case bitonic
        case radix
    }

    public init(precision: Precision = .float32, useIndirectBitonic: Bool = true) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device unavailable")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        self.queue = queue
        
        do {
            // Try compiling from bundled .metal source first to pick up new kernels; fall back to prebuilt metallib.
            let currentFileURL = URL(fileURLWithPath: #filePath)
            let moduleDir = currentFileURL.deletingLastPathComponent()
            let metallibURL = moduleDir.appendingPathComponent("GaussianMetalRenderer.metallib")
            var chosen: MTLLibrary?
            if
                let metalURL = Bundle.module.url(forResource: "GaussianMetalRenderer", withExtension: "metal"),
                let source = try? String(contentsOf: metalURL, encoding: .utf8) {
                do {
                    chosen = try device.makeLibrary(source: source, options: nil)
                } catch {
                    print("[Renderer] Failed to compile Metal source from bundle: \(error)")
                }
            }
            if chosen == nil {
                chosen = try device.makeLibrary(URL: metallibURL)
            }
            var library = chosen!
            if precision == .float16 {
                // Ensure half kernels are available; if not, attempt a direct compile from source next to the binary.
                if library.makeFunction(name: "packTileDataKernelHalf") == nil || library.makeFunction(name: "renderTilesHalf") == nil {
                    let fallbackSourceURL = moduleDir.appendingPathComponent("GaussianMetalRenderer.metal")
                    if let source = try? String(contentsOf: fallbackSourceURL, encoding: .utf8) {
                        do {
                            let rebuilt = try device.makeLibrary(source: source, options: nil)
                            library = rebuilt
                        } catch {
                            print("[Renderer] Failed to compile Metal source for half precision: \(error)")
                        }
                    }
                }
                if library.makeFunction(name: "packTileDataKernelHalf") == nil || library.makeFunction(name: "renderTilesHalf") == nil {
                    let halfOnlySource = """
                    #include <metal_stdlib>
                    using namespace metal;
                    struct GaussianHeader { uint offset; uint count; };
                    struct TileAssignmentHeader { uint totalAssignments; uint maxAssignments; uint paddedCount; uint overflow; };
                    struct PackParams { uint totalAssignments; uint padding; };
                    struct RenderParams { uint width; uint height; uint tileWidth; uint tileHeight; uint tilesX; uint tilesY; uint maxPerTile; uint whiteBackground; uint activeTileCount; uint gaussianCount; };
                    kernel void packTileDataKernelHalf(
                        const device int* sortedIndices [[buffer(0)]],
                        const device float2* means [[buffer(1)]],
                        const device float4* conics [[buffer(2)]],
                        const device packed_float3* colors [[buffer(3)]],
                        const device float* opacities [[buffer(4)]],
                        const device float* depths [[buffer(5)]],
                        device half2* outMeans [[buffer(6)]],
                        device half4* outConics [[buffer(7)]],
                        device half4* outColors [[buffer(8)]],
                        device half* outOpacities [[buffer(9)]],
                        device half* outDepths [[buffer(10)]],
                        const device TileAssignmentHeader* header [[buffer(11)]],
                        const device int* tileIndices [[buffer(12)]],
                        const device int* tileIds [[buffer(13)]],
                        constant PackParams& params [[buffer(14)]],
                        uint gid [[thread_position_in_grid]]
                    ) {
                        uint total = header->totalAssignments;
                        if (gid >= total) { return; }
                        int src = sortedIndices[gid];
                        if (src < 0) { return; }
                        float2 m = means[src];
                        float4 c = conics[src];
                        float3 col = float3(colors[src]);
                        outMeans[gid] = half2(m);
                        outConics[gid] = half4(c);
                        outColors[gid] = half4(half3(col), half(0.0));
                        outOpacities[gid] = half(opacities[src]);
                        outDepths[gid] = half(depths[src]);
                    }
                    kernel void renderTilesHalf(
                        const device GaussianHeader* headers [[buffer(0)]],
                        const device half2* means [[buffer(1)]],
                        const device half4* conics [[buffer(2)]],
                        const device half4* colors [[buffer(3)]],
                        const device half* opacities [[buffer(4)]],
                        const device half* depths [[buffer(5)]],
                        device float* colorOut [[buffer(6)]],
                        device float* depthOut [[buffer(7)]],
                        device float* alphaOut [[buffer(8)]],
                        constant RenderParams& params [[buffer(9)]],
                        const device uint* activeTiles [[buffer(10)]],
                        const device uint* activeTileCount [[buffer(11)]],
                        uint3 localPos3 [[thread_position_in_threadgroup]],
                        uint3 tileCoord [[threadgroup_position_in_grid]]
                    ) {
                        uint tilesX = params.tilesX;
                        uint tilesY = params.tilesY;
                        uint tileId = tileCoord.x;
                        if (tileId >= tilesX * tilesY) { return; }
                        uint tileWidth = params.tileWidth;
                        uint tileHeight = params.tileHeight;
                        uint localX = localPos3.x;
                        uint localY = localPos3.y;
                        uint tileX = tileId % tilesX;
                        uint tileY = tileId / tilesX;
                        uint px = tileX * tileWidth + localX;
                        uint py = tileY * tileHeight + localY;
                        bool inBounds = (px < params.width) && (py < params.height);
                        float3 accumColor = float3(0.0f);
                        float accumDepth = 0.0f;
                        float accumAlpha = 0.0f;
                        float trans = 1.0f;
                        GaussianHeader header = headers[tileId];
                        uint start = header.offset;
                        uint count = header.count;
                        if (count == 0) { return; }
                        for (uint i = 0; i < count; ++i) {
                            uint gIdx = start + i;
                            if (inBounds) {
                                float2 mean = float2(means[gIdx]);
                                float4 conic = float4(conics[gIdx]);
                                float3 color = float3(colors[gIdx].xyz);
                                float baseOpacity = metal::min(float(opacities[gIdx]), 0.99f);
                                if (baseOpacity > 0.0f) {
                                    float fx = float(px);
                                    float fy = float(py);
                                    float dx = fx - mean.x;
                                    float dy = fy - mean.y;
                                    float quad = dx * dx * conic.x + dy * dy * conic.z + 2.0f * dx * dy * conic.y;
                                    if (quad < 20.0f && (conic.x != 0.0f || conic.z != 0.0f)) {
                                        float weight = metal::exp(-0.5f * quad);
                                        float alpha = weight * baseOpacity;
                                        if (alpha > 1e-4f) {
                                            float contrib = trans * alpha;
                                            trans *= (1.0f - alpha);
                                            accumAlpha += contrib;
                                            accumColor += color * contrib;
                                            accumDepth += float(depths[gIdx]) * contrib;
                                            if (trans < 1e-3f) { break; }
                                        }
                                    }
                                }
                            }
                        }
                        if (inBounds) {
                            if (params.whiteBackground != 0) {
                                accumColor += float3(trans);
                            }
                            uint pixelIndex = py * params.width + px;
                            uint base = pixelIndex * 3;
                            colorOut[base + 0] = accumColor.x;
                            colorOut[base + 1] = accumColor.y;
                            colorOut[base + 2] = accumColor.z;
                            depthOut[pixelIndex] = accumDepth;
                            alphaOut[pixelIndex] = accumAlpha;
                        }
                    }
                    """
                    do {
                        let halfLib = try device.makeLibrary(source: halfOnlySource, options: nil)
                        library = halfLib
                    } catch {
                        print("[Renderer] Failed to compile inline half-precision Metal source: \(error)")
                    }
                }
            }
            let halfAvailable = library.makeFunction(name: "packTileDataKernelHalf") != nil && library.makeFunction(name: "renderTilesHalf") != nil
            self.library = library
            self.effectivePrecision = (precision == .float16 && halfAvailable) ? .float16 : .float32
            
            self.tileBoundsEncoder = try TileBoundsEncoder(device: device, library: library)
            self.coverageEncoder = try CoverageEncoder(device: device, library: library)
            self.scatterEncoder = try ScatterEncoder(device: device, library: library)
        self.sortKeyGenEncoder = try SortKeyGenEncoder(device: device, library: library)
        self.bitonicSortEncoder = try BitonicSortEncoder(device: device, library: library, useIndirect: useIndirectBitonic)
            self.radixSortEncoder = try RadixSortEncoder(device: device, library: library)
            self.packEncoder = try PackEncoder(device: device, library: library)
            self.renderEncoder = try RenderEncoder(device: device, library: library)
            self.projectEncoder = try ProjectEncoder(device: device, library: library)
            
            let config = AssignmentDispatchConfigSwift(
                sortThreadgroupSize: UInt32(self.sortKeyGenEncoder.threadgroupSize),
                fuseThreadgroupSize: UInt32(self.radixSortEncoder.fuseThreadgroupSize),
                unpackThreadgroupSize: UInt32(self.radixSortEncoder.unpackThreadgroupSize),
                packThreadgroupSize: UInt32(self.packEncoder.packThreadgroupSize),
                bitonicThreadgroupSize: UInt32(self.bitonicSortEncoder.unitSize),
                radixBlockSize: UInt32(self.radixSortEncoder.blockSize),
                radixGrainSize: UInt32(self.radixSortEncoder.grainSize)
            )
            self.dispatchEncoder = try DispatchEncoder(device: device, library: library, config: config)
            
        } catch {
            fatalError("Failed to load library or initialize encoders: \(error)")
        }
        self.precision = precision

        // Initialize frame slots
        for _ in 0..<maxInFlightFrames {
            self.frameResources.append(nil)
            self.frameInUse.append(false)
        }
    }
    
    func triggerCapture(path: String) {
        self.nextFrameCapturePath = path
    }

    // MARK: - Render Methods
    
    func render(
        headersPtr: UnsafePointer<GaussianHeader>,
        headerCount: Int,
        meansPtr: UnsafePointer<Float>,
        conicsPtr: UnsafePointer<Float>,
        colorsPtr: UnsafePointer<Float>,
        opacityPtr: UnsafePointer<Float>,
        depthsPtr: UnsafePointer<Float>,
        colorOutPtr: UnsafeMutablePointer<Float>?,
        depthOutPtr: UnsafeMutablePointer<Float>?,
        alphaOutPtr: UnsafeMutablePointer<Float>?,
        params: RenderParams
    ) -> Int32 {
        guard headerCount > 0 else {
            return 0
        }
        
        // Helper to create buffers from pointers (Shared mode for CPU->GPU)
        func makeBuf<T>(_ ptr: UnsafePointer<T>, _ count: Int) -> MTLBuffer? {
            self.device.makeBuffer(bytes: ptr, length: count * MemoryLayout<T>.stride, options: .storageModeShared)
        }
        
        let headers = UnsafeBufferPointer(start: headersPtr, count: headerCount)
        guard let last = headers.last else { return 0 }
        let totalAssignments = Int(last.offset + last.count)
        
        guard
            let headersBuf = makeBuf(headersPtr, headerCount),
            let meansBuf = makeBuf(meansPtr, totalAssignments * 2),
            let conicsBuf = makeBuf(conicsPtr, totalAssignments * 4),
            let colorsBuf = makeBuf(colorsPtr, totalAssignments * 3),
            let opacityBuf = makeBuf(opacityPtr, totalAssignments),
            let depthsBuf = makeBuf(depthsPtr, totalAssignments)
        else { return -1 }
        
        // Generate Active Tiles (All tiles)
        guard let activeTileIndices = self.device.makeBuffer(length: headerCount * 4, options: .storageModeShared),
              let activeTileCount = self.device.makeBuffer(length: 4, options: .storageModeShared)
        else { return -2 }
        
        let activeIndicesPtr = activeTileIndices.contents().bindMemory(to: UInt32.self, capacity: headerCount)
        for i in 0..<headerCount {
            activeIndicesPtr[i] = UInt32(i)
        }
        activeTileCount.contents().storeBytes(of: UInt32(headerCount), as: UInt32.self)
        
        let ordered = OrderedGaussianBuffers(
            headers: headersBuf,
            means: meansBuf,
            conics: conicsBuf,
            colors: colorsBuf,
            opacities: opacityBuf,
            depths: depthsBuf,
            tileCount: headerCount,
            activeTileIndices: activeTileIndices,
            activeTileCount: activeTileCount,
            precision: .float32
        )
        
        guard let commandBuffer = self.queue.makeCommandBuffer() else { return -1 }
        return self.encodeAndRunRender(
            commandBuffer: commandBuffer,
            ordered: ordered,
            colorOutPtr: colorOutPtr,
            depthOutPtr: depthOutPtr,
            alphaOutPtr: alphaOutPtr,
            params: params,
            gaussianCount: totalAssignments
        )
    }
    
    func renderRaw(
        gaussianCount: Int,
        meansPtr: UnsafePointer<Float>,
        conicsPtr: UnsafePointer<Float>,
        colorsPtr: UnsafePointer<Float>,
        opacityPtr: UnsafePointer<Float>,
        depthsPtr: UnsafePointer<Float>,
        radiiPtr: UnsafePointer<Float>,
        colorOutPtr: UnsafeMutablePointer<Float>?,
        depthOutPtr: UnsafeMutablePointer<Float>?,
        alphaOutPtr: UnsafeMutablePointer<Float>?,
        params: RenderParams
    ) -> Int32 {
        let pixelCount = Int(params.width * params.height)
        if gaussianCount == 0 || pixelCount == 0 {
            return 0
        }
        
        let count = gaussianCount
        
        // Acquire a frame for rendering
        guard let (frame, slotIndex) = self.acquireFrame(width: Int(params.width), height: Int(params.height)) else { return -5 }

        // Create temporary buffers for input (these are world buffers, not frame resources)
        guard
            let meansBuffer = makeBuffer(ptr: meansPtr, count: count * 2),
            let conicsBuffer = makeBuffer(ptr: conicsPtr, count: count * 4),
            let colorsBuffer = makeBuffer(ptr: colorsPtr, count: count * 3),
            let opacityBuffer = makeBuffer(ptr: opacityPtr, count: count),
            let depthsBuffer = makeBuffer(ptr: depthsPtr, count: count),
            let radiiBuffer = makeBuffer(ptr: radiiPtr, count: count),
            let maskBuffer = self.device.makeBuffer(length: count, options: .storageModeShared)
        else {
            self.releaseFrame(index: slotIndex)
            return -2
        }
        
        // Initialize mask to 1 (valid)
        memset(maskBuffer.contents(), 1, count)
        
        let inputs = GaussianInputBuffers(
            means: meansBuffer,
            radii: radiiBuffer,
            mask: maskBuffer,
            depths: depthsBuffer,
            conics: conicsBuffer,
            colors: colorsBuffer,
            opacities: opacityBuffer
        )
        
        guard let commandBuffer = self.queue.makeCommandBuffer() else { 
            self.releaseFrame(index: slotIndex)
            return -1 
        }
        
        let estimatedAssignments = self.estimateAssignmentCapacity(
            gaussianCount: count,
            meansPtr: meansPtr,
            radiiPtr: radiiPtr,
            params: params
        )
        
        guard let assignment = self.buildTileAssignmentsGPU(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            gaussianBuffers: inputs,
            params: params,
            frame: frame,
            estimatedAssignments: estimatedAssignments
        ) else {
            self.releaseFrame(index: slotIndex)
            return -3
        }
        
        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            assignment: assignment,
            gaussianBuffers: inputs,
            params: params,
            frame: frame,
            precision: self.effectivePrecision
        ) else {
            self.releaseFrame(index: slotIndex)
            return -4
        }

        if ProcessInfo.processInfo.environment["GAUSSIAN_RENDERER_DEBUG"] == "1" {
            let headerPtr = ordered.headers.contents().bindMemory(to: GaussianHeader.self, capacity: assignment.tileCount)
            var nonZero = 0
            var sumCounts = 0
            var maxCount = 0
            for i in 0..<assignment.tileCount {
                let c = Int(headerPtr[i].count)
                sumCounts += c
                if c > 0 { nonZero += 1 }
                if c > maxCount { maxCount = c }
            }
            print("[Renderer][Debug] tiles nonZero=\(nonZero)/\(assignment.tileCount) sum=\(sumCounts) max=\(maxCount)")
        }
        
        return self.encodeAndRunRenderWithFrame(
            commandBuffer: commandBuffer,
            ordered: ordered,
            frame: frame,
            slotIndex: slotIndex,
            colorOutPtr: colorOutPtr,
            depthOutPtr: depthOutPtr,
            alphaOutPtr: alphaOutPtr,
            params: params,
            gaussianCount: count
        )
    }
    
    func renderWorld(
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffers,
        cameraUniforms: CameraUniformsSwift,
        colorOutPtr: UnsafeMutablePointer<Float>?,
        depthOutPtr: UnsafeMutablePointer<Float>?,
        alphaOutPtr: UnsafeMutablePointer<Float>?,
        radiiOutPtr: UnsafeMutablePointer<Float>?,
        maskOutPtr: UnsafeMutablePointer<UInt8>?,
        params: RenderParams
    ) -> Int32 {
         // Metal Capture Start
         if let capturePath = self.nextFrameCapturePath {
             let descriptor = MTLCaptureDescriptor()
             descriptor.captureObject = self.device
             descriptor.destination = .gpuTraceDocument
             descriptor.outputURL = URL(fileURLWithPath: capturePath)
             
             let manager = MTLCaptureManager.shared()
             try? manager.startCapture(with: descriptor)
             self.nextFrameCapturePath = nil
         }

        let count = gaussianCount
        
        // Acquire a frame for rendering
        guard let (frame, slotIndex) = self.acquireFrame(width: Int(params.width), height: Int(params.height)) else { return -5 }

        guard let gaussianBuffers = self.prepareGaussianBuffers(count: count) else {
            self.releaseFrame(index: slotIndex)
            return -2
        }
        
        guard let commandBuffer = self.queue.makeCommandBuffer() else {
            self.releaseFrame(index: slotIndex)
            return -1
        }

        self.projectEncoder.encodeForRender(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            worldBuffers: worldBuffers,
            cameraUniforms: cameraUniforms,
            gaussianBuffers: gaussianBuffers
        )
        
        guard let assignment = self.buildTileAssignmentsGPU(commandBuffer: commandBuffer, gaussianCount: count, gaussianBuffers: gaussianBuffers, params: params, frame: frame) else {
            self.releaseFrame(index: slotIndex)
            return -3
        }
        
        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            assignment: assignment,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            precision: self.effectivePrecision
        ) else {
            self.releaseFrame(index: slotIndex)
            return -4
        }
        
        let result = self.encodeAndRunRenderWithFrame(
            commandBuffer: commandBuffer,
            ordered: ordered,
            frame: frame,
            slotIndex: slotIndex,
            colorOutPtr: colorOutPtr,
            depthOutPtr: depthOutPtr,
            alphaOutPtr: alphaOutPtr,
            params: params,
            gaussianCount: count
        )
        
        if MTLCaptureManager.shared().isCapturing {
            MTLCaptureManager.shared().stopCapture()
        }
        
        return result
    }
    
        public func encodeRender(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffers,
        cameraUniforms: CameraUniformsSwift,
        params: RenderParams
    ) -> RenderOutputBuffers? {
        
        guard let (frame, slotIndex) = self.acquireFrame(width: Int(params.width), height: Int(params.height)) else {
            return nil
        }
    
        if let capturePath = self.nextFrameCapturePath {
            let descriptor = MTLCaptureDescriptor()
            descriptor.captureObject = self.device
            descriptor.destination = .gpuTraceDocument
            descriptor.outputURL = URL(fileURLWithPath: capturePath)
            let manager = MTLCaptureManager.shared()
            try? manager.startCapture(with: descriptor)
            self.nextFrameCapturePath = nil // Consume the capture path
        }

        guard let gaussianBuffers = self.prepareGaussianBuffers(count: gaussianCount) else {
            self.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
            return nil
        }

        self.projectEncoder.encodeForRender(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            worldBuffers: worldBuffers,
            cameraUniforms: cameraUniforms,
            gaussianBuffers: gaussianBuffers
        )

        guard let assignment = self.buildTileAssignmentsGPU(commandBuffer: commandBuffer, gaussianCount: gaussianCount, gaussianBuffers: gaussianBuffers, params: params, frame: frame) else {
            self.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }

            return nil
        }

        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            assignment: assignment,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            precision: self.effectivePrecision
        ) else {
            self.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }

            return nil
        }

        let submission = self.submitRender(
            commandBuffer: commandBuffer,
            ordered: ordered,
            params: params,
            frame: frame,
            slotIndex: slotIndex,
            precision: self.effectivePrecision
        )
        
        guard submission else {
            self.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
            return nil
        }

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
        }
        
        return frame.outputBuffers
    }
    
    // MARK: - Helper Methods
    
    private func estimateAssignmentCapacity(
        gaussianCount: Int,
        meansPtr: UnsafePointer<Float>,
        radiiPtr: UnsafePointer<Float>,
        params: RenderParams
    ) -> Int {
        guard gaussianCount > 0 else { return 1 }
        let tilesX = Int(params.tilesX)
        let tilesY = Int(params.tilesY)
        let tileW = Float(params.tileWidth)
        let tileH = Float(params.tileHeight)
        var total = 0
        for i in 0..<gaussianCount {
            let mx = meansPtr[i * 2 + 0]
            let my = meansPtr[i * 2 + 1]
            let r = radiiPtr[i]
            let minX = max(0, Int(floor((mx - r) / tileW)))
            let maxX = min(tilesX - 1, Int(ceil((mx + r) / tileW)))
            let minY = max(0, Int(floor((my - r) / tileH)))
            let maxY = min(tilesY - 1, Int(ceil((my + r) / tileH)))
            if maxX < minX || maxY < minY { continue }
            let spanX = maxX - minX + 1
            let spanY = maxY - minY + 1
            total += spanX * spanY
        }
        // Clamp to a reasonable upper bound to avoid runaway allocations.
        let conservativeCap = Int(params.tilesX * params.tilesY) * max(1, Int(params.maxPerTile == 0 ? UInt32(gaussianCount) : params.maxPerTile))
        let estimate = min(total, conservativeCap)
        return max(estimate, gaussianCount)
    }
    
    internal func buildTileAssignmentsGPU(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        gaussianBuffers: GaussianInputBuffers,
        params: RenderParams,
        frame: FrameResources,
        estimatedAssignments: Int? = nil
    ) -> TileAssignmentBuffers? {
        let tileCount = Int(params.tilesX * params.tilesY)
        let perTileLimit = (params.maxPerTile == 0) ? UInt32(max(gaussianCount, 1)) : params.maxPerTile
        let baseCapacity = max(tileCount * Int(perTileLimit), 1)
        let estimated = max(estimatedAssignments ?? 0, gaussianCount)
        // Keep a small headroom on the estimate and ensure we never go below the per-tile contract.
        let estimatedWithMargin = Int(Double(estimated) * 1.2)
        let forcedCapacity = max(baseCapacity, estimatedWithMargin)
        guard self.prepareTileBuilderResources(frame: frame, gaussianCount: gaussianCount, tileCount: tileCount, maxPerTile: Int(perTileLimit), forcedCapacity: forcedCapacity) else {
            return nil
        }
        
        self.tileBoundsEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianBuffers: gaussianBuffers,
            boundsBuffer: frame.boundsBuffer!,
            params: params,
            gaussianCount: gaussianCount
        )
        
        self.coverageEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            boundsBuffer: frame.boundsBuffer!,
            opacitiesBuffer: gaussianBuffers.opacities,
            coverageBuffer: frame.coverageBuffer!,
            offsetsBuffer: frame.offsetsBuffer!,
            partialSumsBuffer: frame.partialSumsBuffer!,
            tileAssignmentHeader: frame.tileAssignmentHeader!
        )
        
        
        self.scatterEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            tilesX: Int(params.tilesX),
            offsetsBuffer: frame.offsetsBuffer!,
            dispatchBuffer: frame.scatterDispatchBuffer!,
            boundsBuffer: frame.boundsBuffer!,
            tileIndicesBuffer: frame.tileIndices!,
            tileIdsBuffer: frame.tileIds!,
            tileAssignmentHeader: frame.tileAssignmentHeader!
        )

        if ProcessInfo.processInfo.environment["GAUSSIAN_RENDERER_DEBUG"] == "1" {
            // Copy private offsets/bounds to temporary shared buffers for inspection.
            let offsetsBytes = (gaussianCount + 1) * MemoryLayout<UInt32>.stride
            let boundsBytes = gaussianCount * MemoryLayout<SIMD4<Int32>>.stride
            let offsetsSnapshot = self.device.makeBuffer(length: offsetsBytes, options: .storageModeShared)
            let boundsSnapshot = self.device.makeBuffer(length: boundsBytes, options: .storageModeShared)
            if let offsetsSnapshot, let boundsSnapshot {
                if let blit = commandBuffer.makeBlitCommandEncoder() {
                    blit.copy(from: frame.offsetsBuffer!, sourceOffset: 0, to: offsetsSnapshot, destinationOffset: 0, size: offsetsBytes)
                    blit.copy(from: frame.boundsBuffer!, sourceOffset: 0, to: boundsSnapshot, destinationOffset: 0, size: boundsBytes)
                    blit.endEncoding()
                }
                commandBuffer.addCompletedHandler { [weak frame] _ in
                    guard let headerBuf = frame?.tileAssignmentHeader else { return }
                    let headerPtr = headerBuf.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
                    let h = headerPtr.pointee
                    let offsetPtr = offsetsSnapshot.contents().bindMemory(to: UInt32.self, capacity: max(1, gaussianCount + 1))
                    let totalFromOffsets = gaussianCount > 0 ? Int(offsetPtr[gaussianCount]) : 0
                    let boundsPtr = boundsSnapshot.contents().bindMemory(to: SIMD4<Int32>.self, capacity: max(1, gaussianCount))
                    let sample = (0..<min(4, gaussianCount)).map { i in
                        let b = boundsPtr[i]
                        return "[\(b.x),\(b.y),\(b.z),\(b.w)]"
                    }.joined(separator: " ")
                    print("[Renderer][Debug] tileAssign total=\(h.totalAssignments) padded=\(h.paddedCount) offsetsLast=\(totalFromOffsets) boundsSample=\(sample)")
                    // Small tileId histogram from initial assignments
                    var tileCounts: [Int: Int] = [:]
                    if let tileIdsBuf = frame?.tileIds {
                        let tileIdsPtr = tileIdsBuf.contents().bindMemory(to: Int32.self, capacity: Int(h.totalAssignments))
                        for i in 0..<Int(h.totalAssignments) {
                            let t = Int(tileIdsPtr[i])
                            tileCounts[t, default: 0] += 1
                        }
                        let sampleHist = tileCounts.keys.sorted().prefix(8).map { "\($0):\(tileCounts[$0]!)" }.joined(separator: ", ")
                        print("[Renderer][Debug] tileIds unique=\(tileCounts.count) sample[\(sampleHist)]")
                    }
                }
            }
        }
        
        return TileAssignmentBuffers(
            tileCount: tileCount,
            maxAssignments: frame.tileAssignmentMaxAssignments,
            tileIndices: frame.tileIndices!,
            tileIds: frame.tileIds!,
            header: frame.tileAssignmentHeader!
        )
    }
    
    
    internal func buildOrderedGaussians(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        assignment: TileAssignmentBuffers,
        gaussianBuffers: GaussianInputBuffers,
        params: RenderParams,
        frame: FrameResources,
        precision: Precision
    ) -> OrderedGaussianBuffers? {
        let maxAssignments = assignment.maxAssignments
        guard self.prepareOrderedBuffers(frame: frame, maxAssignments: maxAssignments, tileCount: assignment.tileCount, precision: precision) else { return nil }
        
        guard let sortKeysBuffer = self.ensureBuffer(&frame.sortKeys, length: frame.tileAssignmentPaddedCapacity * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModePrivate, label: "SortKeys"),
              let sortedIndicesBuffer = self.ensureBuffer(&frame.sortedIndices, length: frame.tileAssignmentPaddedCapacity * MemoryLayout<Int32>.stride, options: .storageModePrivate, label: "SortedIndices")
        else { return nil }

        let headerPtr = assignment.header.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        let totalAssignments = Int(headerPtr.pointee.totalAssignments)
        let paddedCount = Int(headerPtr.pointee.paddedCount)

        guard let dispatchArgs = frame.dispatchArgs else { return nil }

        self.dispatchEncoder.encode(
            commandBuffer: commandBuffer,
            header: assignment.header,
            dispatchArgs: dispatchArgs
        )
        
        self.sortKeyGenEncoder.encode(
            commandBuffer: commandBuffer,
            tileIds: assignment.tileIds,
            tileIndices: assignment.tileIndices,
            depths: gaussianBuffers.depths,
            sortKeys: sortKeysBuffer,
            sortedIndices: sortedIndicesBuffer,
            header: assignment.header,
            dispatchArgs: dispatchArgs,
            dispatchOffset: DispatchSlot.sortKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )

        if self.sortAlgorithm == .radix {
            _ = self.ensureRadixBuffers(frame: frame, paddedCapacity: paddedCount)
             let radixBuffers = RadixBufferSet(
                histogram: frame.radixHistogram!,
                blockSums: frame.radixBlockSums!,
                scannedHistogram: frame.radixScannedHistogram!,
                fusedKeys: frame.radixFusedKeys!,
                scratchKeys: frame.radixKeysScratch!,
                scratchPayload: frame.radixPayloadScratch!
            )
            
            let offsets = (
                fuse: DispatchSlot.fuseKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                unpack: DispatchSlot.unpackKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                histogram: DispatchSlot.radixHistogram.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                scanBlocks: DispatchSlot.radixScanBlocks.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                exclusive: DispatchSlot.radixExclusive.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                apply: DispatchSlot.radixApply.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                scatter: DispatchSlot.radixScatter.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
            )
            
            self.radixSortEncoder.encode(
                commandBuffer: commandBuffer,
                keyBuffer: sortKeysBuffer,
                sortedIndices: sortedIndicesBuffer,
                header: assignment.header,
                dispatchArgs: dispatchArgs,
                radixBuffers: radixBuffers,
                offsets: offsets
            )
        } else {
             let offsets = (
                first: DispatchSlot.bitonicFirst.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                general: DispatchSlot.bitonicGeneral.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                final: DispatchSlot.bitonicFinal.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
            )
            
            self.bitonicSortEncoder.encode(
                commandBuffer: commandBuffer,
                sortKeys: sortKeysBuffer,
                sortedIndices: sortedIndicesBuffer,
                header: assignment.header,
                dispatchArgs: dispatchArgs,
                offsets: offsets,
                paddedCapacity: paddedCount
            )
        }
        
        let orderedBuffers = OrderedBufferSet(
            headers: frame.orderedHeaders!,
            means: frame.packedMeans!,
            conics: frame.packedConics!,
            colors: frame.packedColors!,
            opacities: frame.packedOpacities!,
            depths: frame.packedDepths!
        )
        
        self.packEncoder.encode(
            commandBuffer: commandBuffer,
            sortedIndices: sortedIndicesBuffer,
            sortedKeys: sortKeysBuffer,
            gaussianBuffers: gaussianBuffers,
            orderedBuffers: orderedBuffers,
            assignment: assignment,
            totalAssignments: totalAssignments,
            dispatchArgs: dispatchArgs,
            dispatchOffset: DispatchSlot.pack.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            activeTileIndices: frame.activeTileIndices!,
            activeTileCount: frame.activeTileCount!,
            precision: precision
        )

        if ProcessInfo.processInfo.environment["GAUSSIAN_RENDERER_DEBUG"] == "1" {
            commandBuffer.addCompletedHandler { _ in
                let sorted = sortKeysBuffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: frame.tileAssignmentPaddedCapacity)
                let headerPtr = assignment.header.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
                let total = Int(headerPtr.pointee.totalAssignments)
                var tileHistogram: [UInt32: Int] = [:]
                var minTile: UInt32 = .max
                var maxTile: UInt32 = 0
                var sentinelCount = 0
                let paddedCount = Int(headerPtr.pointee.paddedCount)
                for i in 0..<paddedCount {
                    let t = sorted[i].x
                    if t == 0xFFFFFFFF { sentinelCount += 1; continue }
                    tileHistogram[t, default: 0] += 1
                    if t < minTile { minTile = t }
                    if t > maxTile { maxTile = t }
                }
                let sample = tileHistogram.keys.sorted().prefix(8).map { "\($0):\(tileHistogram[$0]!)" }.joined(separator: ", ")
                let tile0 = tileHistogram[0] ?? 0
                print("[Renderer][Debug] sortedKeys tiles \(tileHistogram.count) unique min=\(minTile) max=\(maxTile) sentinel=\(sentinelCount) padded=\(paddedCount) tile0=\(tile0) sample[\(sample)]")
            }
        }
        
        return OrderedGaussianBuffers(
            headers: orderedBuffers.headers,
            means: orderedBuffers.means,
            conics: orderedBuffers.conics,
            colors: orderedBuffers.colors,
            opacities: orderedBuffers.opacities,
            depths: orderedBuffers.depths,
            tileCount: assignment.tileCount,
            activeTileIndices: frame.activeTileIndices!,
            activeTileCount: frame.activeTileCount!,
            precision: precision
        )
    }
        
    
    private func submitRender(
        commandBuffer: MTLCommandBuffer,
        ordered: OrderedGaussianBuffers,
        params: RenderParams,
        frame: FrameResources,
        slotIndex: Int,
        precision: Precision
    ) -> Bool {
        guard let outputs = frame.outputBuffers, let dispatchArgs = frame.dispatchArgs else { return false }
        
        self.renderEncoder.encode(
            commandBuffer: commandBuffer,
            orderedBuffers: ordered,
            outputBuffers: outputs,
            params: params,
            dispatchArgs: dispatchArgs,
            dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            precision: precision
        )

        return true
    }
    
    private func encodeAndRunRender(
        commandBuffer: MTLCommandBuffer,
        ordered: OrderedGaussianBuffers,
        colorOutPtr: UnsafeMutablePointer<Float>?,
        depthOutPtr: UnsafeMutablePointer<Float>?,
        alphaOutPtr: UnsafeMutablePointer<Float>?,
        params: RenderParams,
        gaussianCount: Int = 0
    ) -> Int32 {
        guard let (frame, slotIndex) = self.acquireFrame(width: Int(params.width), height: Int(params.height)) else { return -5 }
        
        return self.encodeAndRunRenderWithFrame(
            commandBuffer: commandBuffer,
            ordered: ordered,
            frame: frame,
            slotIndex: slotIndex,
            colorOutPtr: colorOutPtr,
            depthOutPtr: depthOutPtr,
            alphaOutPtr: alphaOutPtr,
            params: params,
            gaussianCount: gaussianCount
        )
    }
    
    internal func encodeAndRunRenderWithFrame(
        commandBuffer: MTLCommandBuffer,
        ordered: OrderedGaussianBuffers,
        frame: FrameResources,
        slotIndex: Int,
        colorOutPtr: UnsafeMutablePointer<Float>?,
        depthOutPtr: UnsafeMutablePointer<Float>?,
        alphaOutPtr: UnsafeMutablePointer<Float>?,
        params: RenderParams,
        gaussianCount: Int = 0
    ) -> Int32 {
        let cpuReadback = !(colorOutPtr == nil && depthOutPtr == nil && alphaOutPtr == nil)
        guard self.submitRender(
            commandBuffer: commandBuffer,
            ordered: ordered,
            params: params,
            frame: frame,
            slotIndex: slotIndex,
            precision: self.effectivePrecision
        ) else { return -5 }

        if !cpuReadback {
            commandBuffer.addCompletedHandler { [weak self] _ in
                self?.releaseFrame(index: slotIndex)
            }
            commandBuffer.commit()
            return 0
        }
        
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseFrame(index: slotIndex)
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if ProcessInfo.processInfo.environment["GAUSSIAN_RENDERER_DEBUG"] == "1" {
            let headerPtr = ordered.headers.contents().bindMemory(to: GaussianHeader.self, capacity: ordered.tileCount)
            var nonZero = 0
            var sumCounts = 0
            var maxCount = 0
            for i in 0..<ordered.tileCount {
                let c = Int(headerPtr[i].count)
                sumCounts += c
                if c > 0 { nonZero += 1 }
                if c > maxCount { maxCount = c }
            }
            print("[Renderer][Debug] post-render headers nonZero=\(nonZero)/\(ordered.tileCount) sum=\(sumCounts) max=\(maxCount)")
        }
        
        guard let outputs = frame.outputBuffers else { return -6 }
        let pixelCount = Int(params.width * params.height)
        if let p = colorOutPtr { memcpy(p, outputs.colorOutGPU.contents(), pixelCount * 12) }
        if let p = depthOutPtr { memcpy(p, outputs.depthOutGPU.contents(), pixelCount * 4) }
        if let p = alphaOutPtr { memcpy(p, outputs.alphaOutGPU.contents(), pixelCount * 4) }

        return 0
    }
    
    func projectWorldDebug(
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffers,
        cameraUniforms: CameraUniformsSwift,
        meansOutPtr: UnsafeMutablePointer<Float>?,
        conicsOutPtr: UnsafeMutablePointer<Float>?,
        colorsOutPtr: UnsafeMutablePointer<Float>?,
        opacitiesOutPtr: UnsafeMutablePointer<Float>?,
        depthsOutPtr: UnsafeMutablePointer<Float>?,
        radiiOutPtr: UnsafeMutablePointer<Float>?,
        maskOutPtr: UnsafeMutablePointer<UInt8>?
    ) -> Int32 {
        // Setup staging buffers for readback
        guard let commandBuffer = self.queue.makeCommandBuffer() else { return -1 }
        
        // We need buffers to project into. We can use temporary buffers.
        func makeTmp(_ len: Int) -> MTLBuffer? { self.device.makeBuffer(length: len, options: .storageModeShared) }
        
        guard
            let meansOut = makeTmp(gaussianCount * 8), // 2 floats? No projection debug might need 2 or 4.
            let conicsOut = makeTmp(gaussianCount * 16),
            let colorsOut = makeTmp(gaussianCount * 12),
            let opacitiesOut = makeTmp(gaussianCount * 4),
            let depthsOut = makeTmp(gaussianCount * 4),
            let radiiOut = makeTmp(gaussianCount * 4),
            let maskOut = makeTmp(gaussianCount)
        else { return -2 }
        
        let projBuffers = ProjectionReadbackBuffers(
            meansOut: meansOut,
            conicsOut: conicsOut,
            colorsOut: colorsOut,
            opacitiesOut: opacitiesOut,
            depthsOut: depthsOut,
            radiiOut: radiiOut,
            maskOut: maskOut
        )
        
        self.projectEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            worldBuffers: worldBuffers,
            cameraUniforms: cameraUniforms,
            projectionBuffers: projBuffers
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Copy back to pointers
        if let p = meansOutPtr { memcpy(p, meansOut.contents(), gaussianCount * 8) }
        if let p = conicsOutPtr { memcpy(p, conicsOut.contents(), gaussianCount * 16) }
        if let p = colorsOutPtr { memcpy(p, colorsOut.contents(), gaussianCount * 12) }
        if let p = opacitiesOutPtr { memcpy(p, opacitiesOut.contents(), gaussianCount * 4) }
        if let p = depthsOutPtr { memcpy(p, depthsOut.contents(), gaussianCount * 4) }
        if let p = radiiOutPtr { memcpy(p, radiiOut.contents(), gaussianCount * 4) }
        if let p = maskOutPtr { memcpy(p, maskOut.contents(), gaussianCount) }
        
        return 0
    }
    
    // MARK: - Resource Management
    
    internal func ensureBuffer(_ cache: inout MTLBuffer?, length: Int, options: MTLResourceOptions, label: String) -> MTLBuffer? {
        if let existing = cache, existing.length >= length {
            return existing
        }
        guard let newBuf = self.device.makeBuffer(length: length, options: options) else {
            return nil
        }
        newBuf.label = label
        cache = newBuf
        return newBuf
    }
    
    func prepareWorldBuffers(
        count: Int,
        meansPtr: UnsafePointer<Float>,
        scalesPtr: UnsafePointer<Float>,
        rotationsPtr: UnsafePointer<Float>,
        harmonicsPtr: UnsafePointer<Float>,
        opacitiesPtr: UnsafePointer<Float>,
        shComponents: Int
    ) -> WorldGaussianBuffers? {
        let shared: MTLResourceOptions = .storageModeShared
        let positionLength = count * MemoryLayout<SIMD3<Float>>.stride
        let scaleLength = positionLength
        let rotationLength = count * MemoryLayout<SIMD4<Float>>.stride
        let opacityLength = count * MemoryLayout<Float>.stride
        let coeffs = max(shComponents, 0)
        let harmonicsLength = count * MemoryLayout<Float>.stride * (coeffs == 0 ? 3 : coeffs * 3)
        
        guard
            let positions = self.ensureBuffer(&self.worldPositionsCache, length: positionLength, options: shared, label: "WorldPositions"),
            let scales = self.ensureBuffer(&self.worldScalesCache, length: scaleLength, options: shared, label: "WorldScales"),
            let rotations = self.ensureBuffer(&self.worldRotationsCache, length: rotationLength, options: shared, label: "WorldRotations"),
            let harmonics = self.ensureBuffer(&self.worldHarmonicsCache, length: harmonicsLength, options: shared, label: "WorldHarmonics"),
            let opacities = self.ensureBuffer(&self.worldOpacitiesCache, length: opacityLength, options: shared, label: "WorldOpacities")
        else { return nil }
        
        let positionDest = positions.contents().bindMemory(to: SIMD3<Float>.self, capacity: count)
        let scalesDest = scales.contents().bindMemory(to: SIMD3<Float>.self, capacity: count)
        for i in 0..<count {
            let base = i * 3
            positionDest[i] = SIMD3(meansPtr[base + 0], meansPtr[base + 1], meansPtr[base + 2])
            scalesDest[i] = SIMD3(scalesPtr[base + 0], scalesPtr[base + 1], scalesPtr[base + 2])
        }
        memcpy(rotations.contents(), rotationsPtr, rotationLength)
        memcpy(harmonics.contents(), harmonicsPtr, harmonicsLength)
        memcpy(opacities.contents(), opacitiesPtr, opacityLength)
        
        return WorldGaussianBuffers(
            positions: positions,
            scales: scales,
            rotations: rotations,
            harmonics: harmonics,
            opacities: opacities,
            shComponents: shComponents
        )
    }
    
    internal func prepareGaussianBuffers(count: Int) -> GaussianInputBuffers? {
        let shared: MTLResourceOptions = .storageModeShared
        let meansLen = count * 8
        let radiiLen = count * 4
        let maskLen = count
        let depthsLen = count * 4
        let conicsLen = count * 16
        let colorsLen = count * 12
        let opacitiesLen = count * 4
        
        guard
            let means = self.ensureBuffer(&self.gaussianMeansCache, length: meansLen, options: shared, label: "GaussianMeans"),
            let radii = self.ensureBuffer(&self.gaussianRadiiCache, length: radiiLen, options: shared, label: "GaussianRadii"),
            let mask = self.ensureBuffer(&self.gaussianMaskCache, length: maskLen, options: shared, label: "GaussianMask"),
            let depths = self.ensureBuffer(&self.gaussianDepthsCache, length: depthsLen, options: shared, label: "GaussianDepths"),
            let conics = self.ensureBuffer(&self.gaussianConicsCache, length: conicsLen, options: shared, label: "GaussianConics"),
            let colors = self.ensureBuffer(&self.gaussianColorsCache, length: colorsLen, options: shared, label: "GaussianColors"),
            let opacities = self.ensureBuffer(&self.gaussianOpacitiesCache, length: opacitiesLen, options: shared, label: "GaussianOpacities")
        else { return nil }
        
        return GaussianInputBuffers(
            means: means,
            radii: radii,
            mask: mask,
            depths: depths,
            conics: conics,
            colors: colors,
            opacities: opacities
        )
    }
    
    internal func prepareTileBuilderResources(frame: FrameResources, gaussianCount: Int, tileCount: Int, maxPerTile: Int, forcedCapacity: Int) -> Bool {
        let totalAssignments = max(forcedCapacity, 1)
        let paddedCapacity = self.nextPowerOfTwo(value: totalAssignments)
        frame.tileAssignmentMaxAssignments = totalAssignments
        frame.tileAssignmentPaddedCapacity = paddedCapacity

        let boundsBytes = gaussianCount * MemoryLayout<SIMD4<Int32>>.stride
        let coverageBytes = gaussianCount * MemoryLayout<UInt32>.stride
        let offsetsBytes = (gaussianCount + 1) * MemoryLayout<UInt32>.stride
        
        let prefixBlockSize = 256
        let prefixGrainSize = 4
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let groups = (gaussianCount + elementsPerGroup - 1) / elementsPerGroup
        let partialSumsBytes = max(groups, 1) * MemoryLayout<UInt32>.stride
        let dispatchBytes = 3 * MemoryLayout<UInt32>.stride

        let headerBytes = MemoryLayout<TileAssignmentHeaderSwift>.stride
        let tileIndexBytes = paddedCapacity * MemoryLayout<Int32>.stride
        let tileIdBytes = paddedCapacity * MemoryLayout<Int32>.stride

        guard
            let _ = self.ensureBuffer(&frame.boundsBuffer, length: boundsBytes, options: .storageModePrivate, label: "Bounds"),
            let _ = self.ensureBuffer(&frame.coverageBuffer, length: coverageBytes, options: .storageModePrivate, label: "Coverage"),
            let _ = self.ensureBuffer(&frame.offsetsBuffer, length: offsetsBytes, options: .storageModePrivate, label: "Offsets"),
            let _ = self.ensureBuffer(&frame.partialSumsBuffer, length: partialSumsBytes, options: .storageModePrivate, label: "PartialSums"),
            let _ = self.ensureBuffer(&frame.scatterDispatchBuffer, length: dispatchBytes, options: .storageModePrivate, label: "ScatterDispatch"),
            let _ = self.ensureBuffer(&frame.tileAssignmentHeader, length: headerBytes, options: .storageModeShared, label: "TileHeader"),
            let _ = self.ensureBuffer(&frame.tileIndices, length: tileIndexBytes, options: .storageModePrivate, label: "TileIndices"),
            let _ = self.ensureBuffer(&frame.tileIds, length: tileIdBytes, options: .storageModePrivate, label: "TileIds")
        else {
            return false
        }
        
        // Initialize header
        let headerPtr = frame.tileAssignmentHeader!.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.maxAssignments = UInt32(totalAssignments)
        headerPtr.pointee.paddedCount = UInt32(frame.tileAssignmentPaddedCapacity)
        headerPtr.pointee.totalAssignments = UInt32(0)
        headerPtr.pointee.overflow = 0
        
        return true
    }
    
    internal func prepareOrderedBuffers(frame: FrameResources, maxAssignments: Int, tileCount: Int, precision: Precision) -> Bool {
        let assignmentCount = max(1, maxAssignments)
        let headerCount = max(1, tileCount)
        
        let half2Stride = MemoryLayout<UInt16>.stride * 2 // half2
        let half4Stride = MemoryLayout<UInt16>.stride * 4 // half4
        func strideForMeans() -> Int { precision == .float16 ? half2Stride : 8 }
        func strideForConics() -> Int { precision == .float16 ? half4Stride : 16 }
        func strideForColors() -> Int { precision == .float16 ? MemoryLayout<UInt16>.stride * 3 : 12 }
        func strideForOpacities() -> Int { precision == .float16 ? MemoryLayout<UInt16>.stride : 4 }
        func strideForDepths() -> Int { precision == .float16 ? MemoryLayout<UInt16>.stride : 4 }
        
        guard
            self.ensureBuffer(&frame.orderedHeaders, length: headerCount * MemoryLayout<GaussianHeader>.stride, options: .storageModePrivate, label: "OrderedHeaders") != nil,
            self.ensureBuffer(&frame.packedMeans, length: assignmentCount * strideForMeans(), options: .storageModePrivate, label: "PackedMeans") != nil,
            self.ensureBuffer(&frame.packedConics, length: assignmentCount * strideForConics(), options: .storageModePrivate, label: "PackedConics") != nil,
            self.ensureBuffer(&frame.packedColors, length: assignmentCount * strideForColors(), options: .storageModePrivate, label: "PackedColors") != nil,
            self.ensureBuffer(&frame.packedOpacities, length: assignmentCount * strideForOpacities(), options: .storageModePrivate, label: "PackedOpacities") != nil,
            self.ensureBuffer(&frame.packedDepths, length: assignmentCount * strideForDepths(), options: .storageModePrivate, label: "PackedDepths") != nil,
            self.ensureBuffer(&frame.activeTileIndices, length: headerCount * 4, options: .storageModePrivate, label: "ActiveTileIndices") != nil,
            let _ = self.ensureBuffer(&frame.activeTileCount, length: 4, options: .storageModePrivate, label: "ActiveTileCount")
        else { return false }
        return true
    }
    
    private func prepareRenderOutputResources(frame: FrameResources, width: Int, height: Int) -> Bool {
        let pixelCount = max(1, width * height)
        
        if let out = frame.outputBuffers, out.colorOutGPU.length >= pixelCount * 12, out.depthOutGPU.length >= pixelCount * 4, out.alphaOutGPU.length >= pixelCount * 4 {
             return true
        }
        
        guard
            let color = self.device.makeBuffer(length: pixelCount * 12, options: .storageModeShared),
            let depth = self.device.makeBuffer(length: pixelCount * 4, options: .storageModeShared),
            let alpha = self.device.makeBuffer(length: pixelCount * 4, options: .storageModeShared)
        else { return false }
        
        color.label = "RenderColorOutput"
        depth.label = "RenderDepthOutput"
        alpha.label = "RenderAlphaOutput"
        frame.outputBuffers = RenderOutputBuffers(colorOutGPU: color, depthOutGPU: depth, alphaOutGPU: alpha)
        
        return true
    }
    
    internal func acquireFrame(width: Int, height: Int) -> (FrameResources, Int)? {
        frameLock.lock()
        defer { frameLock.unlock() }
        
        let available = min(maxInFlightFrames, frameResources.count)
        guard available > 0 else { return nil }
        
        for _ in 0..<available {
            let idx = frameCursor % available
            frameCursor = (frameCursor + 1) % max(available, 1)
            if frameInUse[idx] == false {
                frameInUse[idx] = true
                if frameResources[idx] == nil {
                    frameResources[idx] = FrameResources(device: self.device)
                }
                let frame = frameResources[idx]!
                if prepareRenderOutputResources(frame: frame, width: width, height: height),
                   prepareTileBuilderResources(frame: frame, gaussianCount: 1, tileCount: 1, maxPerTile: 1, forcedCapacity: 1) { // Default minimal values to prevent nil
                    return (frame, idx)
                }
                // Fail to prepare output -> release
                frameInUse[idx] = false
                return nil
            }
        }
        return nil
    }
    
    internal func releaseFrame(index: Int) {
        frameLock.lock()
        defer { frameLock.unlock() }
        if index >= 0 && index < frameInUse.count {
            frameInUse[index] = false
        }
    }

    
    internal func ensureRadixBuffers(frame: FrameResources, paddedCapacity: Int) -> Bool {
        let valuesPerGroup = 256 * 4
        let gridSize = max(1, (paddedCapacity + valuesPerGroup - 1) / valuesPerGroup)
        let histogramCount = gridSize * 256
        let blockBytes = gridSize * 4
        let fusedKeyBytes = paddedCapacity * 8
        
        _ = self.ensureBuffer(&frame.radixHistogram, length: histogramCount * 4, options: .storageModePrivate, label: "RadixHist")
        _ = self.ensureBuffer(&frame.radixBlockSums, length: blockBytes, options: .storageModePrivate, label: "RadixBlockSums")
        _ = self.ensureBuffer(&frame.radixScannedHistogram, length: histogramCount * 4, options: .storageModePrivate, label: "RadixScanned")
        _ = self.ensureBuffer(&frame.radixFusedKeys, length: fusedKeyBytes, options: .storageModePrivate, label: "RadixFused")
        _ = self.ensureBuffer(&frame.radixKeysScratch, length: fusedKeyBytes, options: .storageModePrivate, label: "RadixScratch")
        _ = self.ensureBuffer(&frame.radixPayloadScratch, length: paddedCapacity * 4, options: .storageModePrivate, label: "RadixPayload")
        return true
    }
    
    func makeBuffer<T>(ptr: UnsafePointer<T>, count: Int) -> MTLBuffer? {
        self.device.makeBuffer(bytes: ptr, length: count * MemoryLayout<T>.stride, options: .storageModeShared)
    }
    
    internal func nextPowerOfTwo(value: Int) -> Int {
        var result = 1
        while result < value {
            result <<= 1
        }
        return result
    }
    
    // MARK: - Debug Methods

}
