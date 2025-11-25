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
    // Output Textures (Alternative)
    var outputTextures: RenderOutputTextures?
    
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

    public let device: MTLDevice
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

    // Half-precision gaussian buffers cache (for half16 pipeline)
    private var gaussianMeansHalfCache: MTLBuffer?
    private var gaussianDepthsHalfCache: MTLBuffer?
    private var gaussianConicsHalfCache: MTLBuffer?
    private var gaussianColorsHalfCache: MTLBuffer?
    private var gaussianOpacitiesHalfCache: MTLBuffer?

    private let sortAlgorithm: SortAlgorithm
    private let precision: Precision
    private let effectivePrecision: Precision
    public var precisionSetting: Precision { self.effectivePrecision }

    /// Enable multi-pixel rendering (4x2 pixels per thread, 64 threads per 32x16 tile)
    /// This can improve performance by reducing thread divergence and memory loads
    /// When enabled, caller should use tileWidth=32, tileHeight=16 in RenderParams
    public let useMultiPixelRendering: Bool
    public var isMultiPixelAvailable: Bool { self.renderEncoder.isMultiPixelAvailable }

    /// FlashGS-style precise ellipse-tile intersection
    /// Eliminates tiles in AABB that don't actually intersect the gaussian ellipse
    /// Reduces wasted gaussian-tile pairs, especially for elongated/rotated gaussians
    public let usePreciseIntersection: Bool
    public var isPreciseIntersectionAvailable: Bool {
        self.coverageEncoder.isPreciseAvailable && self.scatterEncoder.isPreciseAvailable
    }

    /// Recommended tile size for current rendering mode
    public var recommendedTileWidth: UInt32 { useMultiPixelRendering ? 32 : 16 }
    public var recommendedTileHeight: UInt32 { useMultiPixelRendering ? 16 : 16 }

    public init(precision: Precision = .float32, useIndirectBitonic: Bool = false, sortAlgorithm: SortAlgorithm = .radix, useMultiPixelRendering: Bool = false, usePreciseIntersection: Bool = false) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device unavailable")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        self.queue = queue
        
        do {
            let currentFileURL = URL(fileURLWithPath: #filePath)
            let moduleDir = currentFileURL.deletingLastPathComponent()
            let metallibURL = moduleDir.appendingPathComponent("GaussianMetalRenderer.metallib")
            let library = try device.makeLibrary(URL: metallibURL)

            self.library = library
            self.effectivePrecision = precision
            
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
        self.sortAlgorithm = sortAlgorithm
        self.useMultiPixelRendering = useMultiPixelRendering
        self.usePreciseIntersection = usePreciseIntersection

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
        
        // renderRaw always uses float32 precision because inputs are float32 pointers
        guard let assignment = self.buildTileAssignmentsGPU(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            gaussianBuffers: inputs,
            params: params,
            frame: frame,
            estimatedAssignments: estimatedAssignments,
            precision: .float32
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
            precision: .float32  // Always float32 for renderRaw (float input buffers)
        ) else {
            self.releaseFrame(index: slotIndex)
            return -4
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
            gaussianBuffers: gaussianBuffers,
            precision: self.effectivePrecision
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
            gaussianBuffers: gaussianBuffers,
            precision: self.effectivePrecision
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
    
    // New API for Texture-based rendering
    public func encodeRenderToTextures(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffers,
        cameraUniforms: CameraUniformsSwift,
        params: RenderParams
    ) -> RenderOutputTextures? {
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
            self.nextFrameCapturePath = nil
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
            gaussianBuffers: gaussianBuffers,
            precision: self.effectivePrecision
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

        // Submit render to textures
        guard let textures = frame.outputTextures, let dispatchArgs = frame.dispatchArgs else {
            self.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
            return nil
        }

        // Use multi-pixel if enabled and available
        if self.useMultiPixelRendering && self.renderEncoder.isMultiPixelAvailable {
            _ = self.renderEncoder.encodeMultiPixel(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: textures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
            )
        } else {
            self.renderEncoder.encodeDirect(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: textures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                precision: self.effectivePrecision
            )
        }

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
        }

        return textures
    }

    /// New API for half-precision input rendering directly to textures.
    /// This is the fastest path for GSViewer: half16 world data → texture output.
    public func encodeRenderToTextureHalf(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffersHalf,
        cameraUniforms: CameraUniformsSwift,
        params: RenderParams
    ) -> RenderOutputTextures? {
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
            self.nextFrameCapturePath = nil
        }

        // Prepare gaussian buffers with half-precision layout
        guard let gaussianBuffers = self.prepareGaussianBuffersHalf(count: gaussianCount) else {
            self.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
            return nil
        }

        // Project from half world data to half gaussian data
        self.projectEncoder.encodeForRenderHalf(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            worldBuffers: worldBuffers,
            cameraUniforms: cameraUniforms,
            gaussianBuffers: gaussianBuffers
        )

        guard let assignment = self.buildTileAssignmentsGPU(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            precision: .float16  // Half precision for this path
        ) else {
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
            precision: .float16  // Always half for this path
        ) else {
            self.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
            return nil
        }

        guard let textures = frame.outputTextures, let dispatchArgs = frame.dispatchArgs else {
            self.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
            return nil
        }

        // Use multi-pixel if enabled and available
        if self.useMultiPixelRendering && self.renderEncoder.isMultiPixelAvailable {
            _ = self.renderEncoder.encodeMultiPixel(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: textures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
            )
        } else {
            self.renderEncoder.encodeDirect(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: textures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                precision: .float16
            )
        }

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
        }

        return textures
    }

    /// Render to user-provided texture (most direct path).
    /// Allows GSViewer to pass its own drawable texture for zero-copy rendering.
    public func encodeRenderToTargetTexture(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffersHalf,
        cameraUniforms: CameraUniformsSwift,
        params: RenderParams,
        targetColor: MTLTexture,
        targetDepth: MTLTexture?,
        targetAlpha: MTLTexture?
    ) -> Bool {
        guard let (frame, slotIndex) = self.acquireFrame(width: Int(params.width), height: Int(params.height)) else {
            return false
        }

        guard let gaussianBuffers = self.prepareGaussianBuffersHalf(count: gaussianCount) else {
            self.releaseFrame(index: slotIndex)
            return false
        }

        self.projectEncoder.encodeForRenderHalf(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            worldBuffers: worldBuffers,
            cameraUniforms: cameraUniforms,
            gaussianBuffers: gaussianBuffers
        )

        guard let assignment = self.buildTileAssignmentsGPU(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            precision: .float16  // Half precision for this path
        ) else {
            self.releaseFrame(index: slotIndex)
            return false
        }

        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            assignment: assignment,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            precision: .float16
        ) else {
            self.releaseFrame(index: slotIndex)
            return false
        }

        guard let dispatchArgs = frame.dispatchArgs else {
            self.releaseFrame(index: slotIndex)
            return false
        }

        // Use user's target textures
        let outputTextures = RenderOutputTextures(
            color: targetColor,
            depth: targetDepth ?? frame.outputTextures!.depth,
            alpha: targetAlpha ?? frame.outputTextures!.alpha
        )

        // Use multi-pixel if enabled and available
        if self.useMultiPixelRendering && self.renderEncoder.isMultiPixelAvailable {
            _ = self.renderEncoder.encodeMultiPixel(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: outputTextures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
            )
        } else {
            self.renderEncoder.encodeDirect(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: outputTextures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                precision: .float16
            )
        }

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseFrame(index: slotIndex)
        }

        return true
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
        estimatedAssignments: Int? = nil,
        precision: Precision = .float32
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
            gaussianCount: gaussianCount,
            precision: precision
        )
        
        // Coverage: count tiles per gaussian
        if self.usePreciseIntersection && self.isPreciseIntersectionAvailable {
            // FlashGS precise ellipse-tile intersection
            self.coverageEncoder.encodePrecise(
                commandBuffer: commandBuffer,
                gaussianCount: gaussianCount,
                tileWidth: Int(params.tileWidth),
                tileHeight: Int(params.tileHeight),
                boundsBuffer: frame.boundsBuffer!,
                opacitiesBuffer: gaussianBuffers.opacities,
                meansBuffer: gaussianBuffers.means,
                conicsBuffer: gaussianBuffers.conics,
                coverageBuffer: frame.coverageBuffer!,
                offsetsBuffer: frame.offsetsBuffer!,
                partialSumsBuffer: frame.partialSumsBuffer!,
                tileAssignmentHeader: frame.tileAssignmentHeader!,
                precision: precision
            )
        } else {
            self.coverageEncoder.encode(
                commandBuffer: commandBuffer,
                gaussianCount: gaussianCount,
                boundsBuffer: frame.boundsBuffer!,
                opacitiesBuffer: gaussianBuffers.opacities,
                coverageBuffer: frame.coverageBuffer!,
                offsetsBuffer: frame.offsetsBuffer!,
                partialSumsBuffer: frame.partialSumsBuffer!,
                tileAssignmentHeader: frame.tileAssignmentHeader!,
                precision: precision
            )
        }

        // Scatter: write gaussian-tile pairs
        if self.usePreciseIntersection && self.isPreciseIntersectionAvailable {
            // FlashGS precise scatter (parallel: one threadgroup per gaussian)
            self.scatterEncoder.encodePrecise(
                commandBuffer: commandBuffer,
                gaussianCount: gaussianCount,
                tilesX: Int(params.tilesX),
                tileWidth: Int(params.tileWidth),
                tileHeight: Int(params.tileHeight),
                offsetsBuffer: frame.offsetsBuffer!,
                dispatchBuffer: frame.scatterDispatchBuffer!,
                boundsBuffer: frame.boundsBuffer!,
                tileIndicesBuffer: frame.tileIndices!,
                tileIdsBuffer: frame.tileIds!,
                tileAssignmentHeader: frame.tileAssignmentHeader!,
                meansBuffer: gaussianBuffers.means,
                conicsBuffer: gaussianBuffers.conics,
                opacitiesBuffer: gaussianBuffers.opacities,
                precision: precision
            )
        } else {
            // Load-balanced scatter (binary search) for better GPU utilization
            self.scatterEncoder.encodeBalanced(
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

        let paddedCount = frame.tileAssignmentPaddedCapacity

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
            dispatchOffset: DispatchSlot.sortKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            precision: precision
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
                offsets: offsets,
                tileCount: assignment.tileCount
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
            dispatchArgs: dispatchArgs,
            dispatchOffset: DispatchSlot.pack.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            activeTileIndices: frame.activeTileIndices!,
            activeTileCount: frame.activeTileCount!,
            precision: precision
        )
        
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
            precision: ordered.precision  // Use the precision the buffers were packed with
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

    /// Prepare gaussian buffers with half-precision layout for the half16 pipeline.
    /// Note: radii and mask remain the same size (float and uchar).
    internal func prepareGaussianBuffersHalf(count: Int) -> GaussianInputBuffers? {
        let shared: MTLResourceOptions = .storageModeShared
        // Half precision sizes
        let meansLen = count * 4      // half2 = 4 bytes
        let radiiLen = count * 4      // float (stays float for tile bounds accuracy)
        let maskLen = count           // uchar
        let depthsLen = count * 2     // half = 2 bytes
        let conicsLen = count * 8     // half4 = 8 bytes
        let colorsLen = count * 6     // half3 = 6 bytes
        let opacitiesLen = count * 2  // half = 2 bytes

        guard
            let means = self.ensureBuffer(&self.gaussianMeansHalfCache, length: meansLen, options: shared, label: "GaussianMeansHalf"),
            let radii = self.ensureBuffer(&self.gaussianRadiiCache, length: radiiLen, options: shared, label: "GaussianRadii"),
            let mask = self.ensureBuffer(&self.gaussianMaskCache, length: maskLen, options: shared, label: "GaussianMask"),
            let depths = self.ensureBuffer(&self.gaussianDepthsHalfCache, length: depthsLen, options: shared, label: "GaussianDepthsHalf"),
            let conics = self.ensureBuffer(&self.gaussianConicsHalfCache, length: conicsLen, options: shared, label: "GaussianConicsHalf"),
            let colors = self.ensureBuffer(&self.gaussianColorsHalfCache, length: colorsLen, options: shared, label: "GaussianColorsHalf"),
            let opacities = self.ensureBuffer(&self.gaussianOpacitiesHalfCache, length: opacitiesLen, options: shared, label: "GaussianOpacitiesHalf")
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
            let _ = self.ensureBuffer(&frame.activeTileCount, length: 4, options: .storageModeShared, label: "ActiveTileCount")
        else { return false }
        return true
    }
    
    private func prepareRenderOutputResources(frame: FrameResources, width: Int, height: Int) -> Bool {
        let pixelCount = max(1, width * height)
        
        // 1. Buffers (Standard)
        if let out = frame.outputBuffers, out.colorOutGPU.length >= pixelCount * 12, out.depthOutGPU.length >= pixelCount * 4, out.alphaOutGPU.length >= pixelCount * 4 {
             // Good
        } else {
            guard
                let color = self.device.makeBuffer(length: pixelCount * 12, options: .storageModeShared),
                let depth = self.device.makeBuffer(length: pixelCount * 4, options: .storageModeShared),
                let alpha = self.device.makeBuffer(length: pixelCount * 4, options: .storageModeShared)
            else { return false }
            
            color.label = "RenderColorOutput"
            depth.label = "RenderDepthOutput"
            alpha.label = "RenderAlphaOutput"
            frame.outputBuffers = RenderOutputBuffers(colorOutGPU: color, depthOutGPU: depth, alphaOutGPU: alpha)
        }
        
        // 2. Textures (Alternative)
        let currentW = frame.outputTextures?.color.width ?? 0
        let currentH = frame.outputTextures?.color.height ?? 0
        if currentW != width || currentH != height {
            // Create new textures
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
            desc.usage = [.shaderWrite, .shaderRead]
            desc.storageMode = .private
            
            guard let colorTex = self.device.makeTexture(descriptor: desc) else { return false }
            colorTex.label = "OutputColorTex"
            
            let descF32 = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
            descF32.usage = [.shaderWrite, .shaderRead]
            descF32.storageMode = .private
            
            guard let depthTex = self.device.makeTexture(descriptor: descF32) else { return false }
            depthTex.label = "OutputDepthTex"
            
            guard let alphaTex = self.device.makeTexture(descriptor: descF32) else { return false }
            alphaTex.label = "OutputAlphaTex"
            
            frame.outputTextures = RenderOutputTextures(color: colorTex, depth: depthTex, alpha: alphaTex)
        }
        
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
        let valuesPerGroup = self.radixSortEncoder.blockSize * self.radixSortEncoder.grainSize
        let gridSize = max(1, (paddedCapacity + valuesPerGroup - 1) / valuesPerGroup)
        let histogramCount = gridSize * self.radixSortEncoder.radix
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
