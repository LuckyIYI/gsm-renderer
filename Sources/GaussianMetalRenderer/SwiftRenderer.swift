import Foundation
import simd
@preconcurrency import Metal

// MARK: - C Bridge Functions

@_cdecl("gaussian_renderer_set_capture_path")
public func setNextFrameCapturePath(_ path: String) {
    Renderer.shared.triggerCapture(path: path)
}

private func matrixFromPointer(_ ptr: UnsafePointer<Float>) -> simd_float4x4 {
    var matrix = simd_float4x4()
    for row in 0..<4 {
        for col in 0..<4 {
            matrix[col][row] = ptr[col * 4 + row]
        }
    }
    return matrix
}

@_cdecl("gaussian_renderer_render")
public func gaussian_renderer_render(
    headersPtr: UnsafeRawPointer?,
    headerCount: Int32,
    meansPtr: UnsafePointer<Float>?,
    conicsPtr: UnsafePointer<Float>?,
    colorsPtr: UnsafePointer<Float>?,
    opacityPtr: UnsafePointer<Float>?,
    depthsPtr: UnsafePointer<Float>?,
    colorOutPtr: UnsafeMutablePointer<Float>?,
    depthOutPtr: UnsafeMutablePointer<Float>?,
    alphaOutPtr: UnsafeMutablePointer<Float>?,
    width: Int32,
    height: Int32,
    tileWidth: Int32,
    tileHeight: Int32,
    tilesX: Int32,
    tilesY: Int32,
    maxPerTile: Int32,
    whiteBackground: Int32
) -> Int32 {
    guard
        let headersPtr,
        let meansPtr,
        let conicsPtr,
        let colorsPtr,
        let opacityPtr,
        let depthsPtr
    else {
        return -1
    }

    let headersTyped = headersPtr.bindMemory(to: GaussianHeader.self, capacity: Int(headerCount))
    let count = Int(headerCount)
    
    // Acquire a frame for rendering
    guard let (frame, slotIndex) = Renderer.shared.acquireFrame(width: Int(width), height: Int(height)) else { return -5 }
    
    let ordered = OrderedGaussianBuffers(
        headers: Renderer.shared.makeBuffer(ptr: headersTyped, count: count)!,
        means: Renderer.shared.makeBuffer(ptr: meansPtr, count: count * 2)!,
        conics: Renderer.shared.makeBuffer(ptr: conicsPtr, count: count * 4)!,
        colors: Renderer.shared.makeBuffer(ptr: colorsPtr, count: count * 3)!,
        opacities: Renderer.shared.makeBuffer(ptr: opacityPtr, count: count)!,
        depths: Renderer.shared.makeBuffer(ptr: depthsPtr, count: count)!,
        tileCount: count,
        activeTileIndices: frame.activeTileIndices!,
        activeTileCount: frame.activeTileCount!
    )
    
    guard let commandBuffer = Renderer.shared.queue.makeCommandBuffer() else { 
        Renderer.shared.releaseFrame(index: slotIndex)
        return -1 
    }

    let params = RenderParams(
        width: UInt32(width),
        height: UInt32(height),
        tileWidth: UInt32(tileWidth),
        tileHeight: UInt32(tileHeight),
        tilesX: UInt32(tilesX),
        tilesY: UInt32(tilesY),
        maxPerTile: UInt32(maxPerTile),
        whiteBackground: UInt32(whiteBackground),
        activeTileCount: 0,
        gaussianCount: UInt32(count)
    )

    let result = Renderer.shared.encodeAndRunRenderWithFrame(
        commandBuffer: commandBuffer,
        ordered: ordered,
        frame: frame,
        slotIndex: slotIndex,
        colorOutPtr: colorOutPtr,
        depthOutPtr: depthOutPtr,
        alphaOutPtr: alphaOutPtr,
        params: params
    )
    
    // encodeAndRunRenderWithFrame commits and waits, and releases the frame.
    return result
}

@_cdecl("gaussian_renderer_render_raw")
public func gaussian_renderer_render_raw(
    gaussianCount: Int32,
    meansPtr: UnsafePointer<Float>?,
    conicsPtr: UnsafePointer<Float>?,
    colorsPtr: UnsafePointer<Float>?,
    opacityPtr: UnsafePointer<Float>?,
    depthsPtr: UnsafePointer<Float>?,
    radiiPtr: UnsafePointer<Float>?,
    width: Int32,
    height: Int32,
    tileWidth: Int32,
    tileHeight: Int32,
    tilesX: Int32,
    tilesY: Int32,
    maxPerTile: Int32,
    whiteBackground: Int32,
    colorOutPtr: UnsafeMutablePointer<Float>?,
    depthOutPtr: UnsafeMutablePointer<Float>?,
    alphaOutPtr: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard
        let meansPtr,
        let conicsPtr,
        let colorsPtr,
        let opacityPtr,
        let depthsPtr,
        let radiiPtr
    else {
        return -1
    }

    let pixelCount = Int(width * height)
    if gaussianCount == 0 || pixelCount == 0 {
        return 0
    }
    
    let count = Int(gaussianCount)
    
    // Acquire a frame for rendering
    guard let (frame, slotIndex) = Renderer.shared.acquireFrame(width: Int(width), height: Int(height)) else { return -5 }

    // Create temporary buffers for input (these are world buffers, not frame resources)
    guard
        let meansBuffer = Renderer.shared.makeBuffer(ptr: meansPtr, count: count * 2),
        let conicsBuffer = Renderer.shared.makeBuffer(ptr: conicsPtr, count: count * 4),
        let colorsBuffer = Renderer.shared.makeBuffer(ptr: colorsPtr, count: count * 3),
        let opacityBuffer = Renderer.shared.makeBuffer(ptr: opacityPtr, count: count),
        let depthsBuffer = Renderer.shared.makeBuffer(ptr: depthsPtr, count: count),
        let radiiBuffer = Renderer.shared.makeBuffer(ptr: radiiPtr, count: count),
        let maskBuffer = Renderer.shared.device.makeBuffer(length: count, options: .storageModeShared)
    else {
        Renderer.shared.releaseFrame(index: slotIndex)
        return -2
    }
    
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
    
    guard let commandBuffer = Renderer.shared.queue.makeCommandBuffer() else { 
        Renderer.shared.releaseFrame(index: slotIndex)
        return -1 
    }
    
    let params = RenderParams(
        width: UInt32(width),
        height: UInt32(height),
        tileWidth: UInt32(tileWidth),
        tileHeight: UInt32(tileHeight),
        tilesX: UInt32(tilesX),
        tilesY: UInt32(tilesY),
        maxPerTile: UInt32(maxPerTile),
        whiteBackground: UInt32(whiteBackground),
        activeTileCount: 0,
        gaussianCount: UInt32(count)
    )
    
    guard let assignment = Renderer.shared.buildTileAssignmentsGPU(commandBuffer: commandBuffer, gaussianCount: count, gaussianBuffers: inputs, params: params, frame: frame) else {
        Renderer.shared.releaseFrame(index: slotIndex)
        return -3
    }
    
    guard let ordered = Renderer.shared.buildOrderedGaussians(
        commandBuffer: commandBuffer,
        gaussianCount: count,
        assignment: assignment,
        gaussianBuffers: inputs,
        params: params,
        frame: frame
    ) else {
        Renderer.shared.releaseFrame(index: slotIndex)
        return -4
    }
    
    let result = Renderer.shared.encodeAndRunRenderWithFrame(
        commandBuffer: commandBuffer,
        ordered: ordered,
        frame: frame,
        slotIndex: slotIndex,
        colorOutPtr: colorOutPtr,
        depthOutPtr: depthOutPtr,
        alphaOutPtr: alphaOutPtr,
        params: params
    )
    
    return result
}

@_cdecl("gaussian_renderer_render_world")
public func gaussian_renderer_render_world(
    gaussianCount: Int32,
    meansPtr: UnsafePointer<Float>?,
    scalesPtr: UnsafePointer<Float>?,
    rotationsPtr: UnsafePointer<Float>?,
    harmonicsPtr: UnsafePointer<Float>?,
    opacityPtr: UnsafePointer<Float>?,
    shComponents: Int32,
    viewMatrixPtr: UnsafePointer<Float>?,
    projectionMatrixPtr: UnsafePointer<Float>?,
    cameraCenterPtr: UnsafePointer<Float>?,
    pixelFactor: Float,
    focalX: Float,
    focalY: Float,
    nearPlane: Float,
    farPlane: Float,
    width: Int32,
    height: Int32,
    tileWidth: Int32,
    tileHeight: Int32,
    tilesX: Int32,
    tilesY: Int32,
    maxPerTile: Int32,
    whiteBackground: Int32,
    colorOutPtr: UnsafeMutablePointer<Float>?,
    depthOutPtr: UnsafeMutablePointer<Float>?,
    alphaOutPtr: UnsafeMutablePointer<Float>?,
    radiiOutPtr: UnsafeMutablePointer<Float>?,
    maskOutPtr: UnsafeMutablePointer<UInt8>?
) -> Int32 {
    guard
        let meansPtr,
        let scalesPtr,
        let rotationsPtr,
        let harmonicsPtr,
        let opacityPtr,
        let viewPtr = viewMatrixPtr,
        let projPtr = projectionMatrixPtr,
        let cameraPtr = cameraCenterPtr
    else {
        return -1
    }
    
    let count = Int(gaussianCount)
    
    guard let worldBuffers = Renderer.shared.prepareWorldBuffers(
        count: count,
        meansPtr: meansPtr,
        scalesPtr: scalesPtr,
        rotationsPtr: rotationsPtr,
        harmonicsPtr: harmonicsPtr,
        opacitiesPtr: opacityPtr,
        shComponents: Int(shComponents)
    ) else {
        return -2
    }
    let cameraUniforms = CameraUniformsSwift(
        viewMatrix: matrixFromPointer(viewPtr),
        projectionMatrix: matrixFromPointer(projPtr),
        cameraCenter: SIMD3<Float>(cameraPtr[0], cameraPtr[1], cameraPtr[2]),
        pixelFactor: pixelFactor,
        focalX: focalX,
        focalY: focalY,
        width: Float(width),
        height: Float(height),
        nearPlane: max(max(nearPlane, 0.001), 0.2),
        farPlane: farPlane,
        shComponents: UInt32(max(shComponents, 0)),
        gaussianCount: UInt32(count),
        padding0: 0,
        padding1: 0
    )
    let params = RenderParams(
        width: UInt32(width),
        height: UInt32(height),
        tileWidth: UInt32(tileWidth),
        tileHeight: UInt32(tileHeight),
        tilesX: UInt32(tilesX),
        tilesY: UInt32(tilesY),
        maxPerTile: UInt32(maxPerTile),
        whiteBackground: UInt32(whiteBackground),
        activeTileCount: 0,
        gaussianCount: UInt32(count)
    )
    
    // Acquire a frame for rendering
    guard let (frame, slotIndex) = Renderer.shared.acquireFrame(width: Int(width), height: Int(height)) else { return -5 }
    
    guard let commandBuffer = Renderer.shared.queue.makeCommandBuffer() else {
        Renderer.shared.releaseFrame(index: slotIndex)
        return -1
    }

    // Use shared gaussian buffers (not per-frame)
    guard let gaussianBuffers = Renderer.shared.prepareGaussianBuffers(count: count) else {
        Renderer.shared.releaseFrame(index: slotIndex)
        return -2
    }

    Renderer.shared.projectEncoder.encodeForRender(
        commandBuffer: commandBuffer,
        gaussianCount: count,
        worldBuffers: worldBuffers,
        cameraUniforms: cameraUniforms,
        gaussianBuffers: gaussianBuffers
    )
    
    guard let assignment = Renderer.shared.buildTileAssignmentsGPU(commandBuffer: commandBuffer, gaussianCount: count, gaussianBuffers: gaussianBuffers, params: params, frame: frame) else {
        Renderer.shared.releaseFrame(index: slotIndex)
        return -3
    }
    
    guard let ordered = Renderer.shared.buildOrderedGaussians(
        commandBuffer: commandBuffer,
        gaussianCount: count,
        assignment: assignment,
        gaussianBuffers: gaussianBuffers,
        params: params,
        frame: frame
    ) else {
        Renderer.shared.releaseFrame(index: slotIndex)
        return -4
    }
    
    let result = Renderer.shared.encodeAndRunRenderWithFrame(
        commandBuffer: commandBuffer,
        ordered: ordered,
        frame: frame,
        slotIndex: slotIndex,
        colorOutPtr: colorOutPtr,
        depthOutPtr: depthOutPtr,
        alphaOutPtr: alphaOutPtr,
        params: params
    )
    
    return result
}

@_cdecl("gaussian_renderer_project_world_debug")
public func gaussian_renderer_project_world_debug(
    gaussianCount: Int32,
    meansPtr: UnsafePointer<Float>?,
    scalesPtr: UnsafePointer<Float>?,
    rotationsPtr: UnsafePointer<Float>?,
    harmonicsPtr: UnsafePointer<Float>?,
    opacityPtr: UnsafePointer<Float>?,
    shComponents: Int32,
    viewMatrixPtr: UnsafePointer<Float>?,
    projectionMatrixPtr: UnsafePointer<Float>?,
    cameraCenterPtr: UnsafePointer<Float>?,
    pixelFactor: Float,
    focalX: Float,
    focalY: Float,
    nearPlane: Float,
    farPlane: Float,
    width: Int32,
    height: Int32,
    meansOutPtr: UnsafeMutablePointer<Float>?,
    conicsOutPtr: UnsafeMutablePointer<Float>?,
    colorsOutPtr: UnsafeMutablePointer<Float>?,
    opacitiesOutPtr: UnsafeMutablePointer<Float>?,
    depthsOutPtr: UnsafeMutablePointer<Float>?,
    radiiOutPtr: UnsafeMutablePointer<Float>?,
    maskOutPtr: UnsafeMutablePointer<UInt8>?
) -> Int32 {
    guard
        let meansPtr,
        let scalesPtr,
        let rotationsPtr,
        let harmonicsPtr,
        let opacityPtr,
        let viewPtr = viewMatrixPtr,
        let projPtr = projectionMatrixPtr,
        let cameraPtr = cameraCenterPtr
    else {
        return -1
    }
    let count = Int(gaussianCount)
    guard let worldBuffers = Renderer.shared.prepareWorldBuffers(
        count: count,
        meansPtr: meansPtr,
        scalesPtr: scalesPtr,
        rotationsPtr: rotationsPtr,
        harmonicsPtr: harmonicsPtr,
        opacitiesPtr: opacityPtr,
        shComponents: Int(shComponents)
    ) else {
        return -2
    }
    let cameraUniforms = CameraUniformsSwift(
        viewMatrix: matrixFromPointer(viewPtr),
        projectionMatrix: matrixFromPointer(projPtr),
    cameraCenter: SIMD3<Float>(cameraPtr[0], cameraPtr[1], cameraPtr[2]),
    pixelFactor: pixelFactor,
    focalX: focalX,
    focalY: focalY,
        width: Float(width),
        height: Float(height),
        nearPlane: max(max(nearPlane, 0.001), 0.2),
        farPlane: farPlane,
        shComponents: UInt32(max(shComponents, 0)),
        gaussianCount: UInt32(count),
        padding0: 0,
        padding1: 0
    )
    return Renderer.shared.projectWorldDebug(
        gaussianCount: count,
        worldBuffers: worldBuffers,
        cameraUniforms: cameraUniforms,
        meansOutPtr: meansOutPtr,
        conicsOutPtr: conicsOutPtr,
        colorsOutPtr: colorsOutPtr,
        opacitiesOutPtr: opacitiesOutPtr,
        depthsOutPtr: depthsOutPtr,
        radiiOutPtr: radiiOutPtr,
        maskOutPtr: maskOutPtr
    )
}


@_cdecl("gaussian_renderer_dump_tile_assignment")
public func gaussian_renderer_dump_tile_assignment(
    headerOutPtr: UnsafeMutableRawPointer?,
    headerCapacity: Int32,
    tileIndicesOutPtr: UnsafeMutablePointer<Int32>?,
    tileIndicesCapacity: Int32,
    tileIdsOutPtr: UnsafeMutablePointer<Int32>?,
    tileIdsCapacity: Int32,
    totalAssignmentsOut: UnsafeMutablePointer<Int32>?
) -> Int32 {
    // For debug dumps, we acquire a frame to access its buffers.
    // We're not rendering, so width/height/maxPerTile can be minimal.
    guard let (frame, slotIndex) = Renderer.shared.acquireFrame(width: 1, height: 1) else { return -5 }
    defer { Renderer.shared.releaseFrame(index: slotIndex) }

    guard let orderedH = frame.orderedHeaders,
          let globalH = frame.tileAssignmentHeader,
          let ti = frame.tileIndices,
          let tids = frame.tileIds else {
        return -2
    }
    
    // Header (Per-Tile)
    if let hPtr = headerOutPtr {
        let len = min(Int(headerCapacity) * MemoryLayout<GaussianHeader>.stride, orderedH.length)
        memcpy(hPtr, orderedH.contents(), len)
    }
    
    // Indices
    if let tiPtr = tileIndicesOutPtr {
        let len = min(Int(tileIndicesCapacity) * 4, ti.length)
        memcpy(tiPtr, ti.contents(), len)
    }
    
    // IDs
    if let tidsPtr = tileIdsOutPtr {
        let len = min(Int(tileIdsCapacity) * 4, tids.length)
        memcpy(tidsPtr, tids.contents(), len)
    }
    
    // Total Assignments
    let header = globalH.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1).pointee
    if let out = totalAssignmentsOut {
        out.pointee = Int32(header.totalAssignments)
    }
    
    return 0
}

@_cdecl("gaussian_renderer_dump_tile_bounds")
public func gaussian_renderer_dump_tile_bounds(
    boundsOutPtr: UnsafeMutablePointer<Int32>?,
    boundsCapacity: Int32,
    coverageOutPtr: UnsafeMutablePointer<UInt32>?,
    coverageCapacity: Int32,
    offsetsOutPtr: UnsafeMutablePointer<UInt32>?,
    offsetsCapacity: Int32
) -> Int32 {
    // Acquire a frame for debug dumps
    guard let (frame, slotIndex) = Renderer.shared.acquireFrame(width: 1, height: 1) else { return -5 }
    defer { Renderer.shared.releaseFrame(index: slotIndex) }

    guard let r = frame.boundsBuffer,
          let c = frame.coverageBuffer,
          let o = frame.offsetsBuffer else {
        return -2
    }
    
    guard let cmd = Renderer.shared.queue.makeCommandBuffer(),
          let blit = cmd.makeBlitCommandEncoder(),
          let bShared = Renderer.shared.device.makeBuffer(length: r.length, options: .storageModeShared),
          let cShared = Renderer.shared.device.makeBuffer(length: c.length, options: .storageModeShared),
          let oShared = Renderer.shared.device.makeBuffer(length: o.length, options: .storageModeShared)
    else { return -3 }
    
    blit.copy(from: r, sourceOffset: 0, to: bShared, destinationOffset: 0, size: r.length)
    blit.copy(from: c, sourceOffset: 0, to: cShared, destinationOffset: 0, size: c.length)
    blit.copy(from: o, sourceOffset: 0, to: oShared, destinationOffset: 0, size: o.length)
    blit.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    
    let count = r.length / MemoryLayout<SIMD4<Int32>>.stride
    
    if let bPtr = boundsOutPtr {
        memcpy(bPtr, bShared.contents(), min(Int(boundsCapacity) * 4, r.length))
    }
    
    if let cPtr = coverageOutPtr {
        memcpy(cPtr, cShared.contents(), min(Int(coverageCapacity) * 4, c.length))
    }
    
    if let oPtr = offsetsOutPtr {
        memcpy(oPtr, oShared.contents(), min(Int(offsetsCapacity) * 4, o.length))
    }
    
    return Int32(count)
}

@_cdecl("gaussian_renderer_tile_pack_debug")
public func gaussian_renderer_tile_pack_debug(
    gaussianCount: Int32,
    meansPtr: UnsafePointer<Float>?,
    scalesPtr: UnsafePointer<Float>?,
    rotationsPtr: UnsafePointer<Float>?,
    harmonicsPtr: UnsafePointer<Float>?,
    opacityPtr: UnsafePointer<Float>?,
    shComponents: Int32,
    viewMatrixPtr: UnsafePointer<Float>?,
    projectionMatrixPtr: UnsafePointer<Float>?,
    cameraCenterPtr: UnsafePointer<Float>?,
    pixelFactor: Float,
    focalX: Float,
    focalY: Float,
    nearPlane: Float,
    farPlane: Float,
    width: Int32,
    height: Int32,
    tileWidth: Int32,
    tileHeight: Int32,
    maxPerTile: Int32,
    whiteBackground: Int32,
    headerOutPtr: UnsafeMutableRawPointer?,
    headerCapacity: Int32,
    tileIndicesOutPtr: UnsafeMutablePointer<Int32>?,
    tileIndicesCapacity: Int32,
    tileIdsOutPtr: UnsafeMutablePointer<Int32>?,
    tileIdsCapacity: Int32,
    meansOutPtr: UnsafeMutablePointer<Float>?,
    meansCapacity: Int32,
    conicsOutPtr: UnsafeMutablePointer<Float>?,
    conicsCapacity: Int32,
    colorsOutPtr: UnsafeMutablePointer<Float>?,
    colorsCapacity: Int32,
    opacitiesOutPtr: UnsafeMutablePointer<Float>?,
    opacitiesCapacity: Int32,
    depthsOutPtr: UnsafeMutablePointer<Float>?,
    depthsCapacity: Int32,
    tileCountOutPtr: UnsafeMutablePointer<Int32>?,
    maxAssignmentsOutPtr: UnsafeMutablePointer<Int32>?,
    totalAssignmentsOutPtr: UnsafeMutablePointer<Int32>?
) -> Int32 {
    return -1
}

@_cdecl("gaussian_renderer_debug_sort_assignments")
public func gaussian_renderer_debug_sort_assignments(
    tileIdsPtr: UnsafePointer<Int32>?,
    tileIndicesPtr: UnsafePointer<Int32>?,
    depthsPtr: UnsafePointer<Float>?,
    count: Int32,
    outputPtr: UnsafeMutablePointer<Int32>?
) -> Int32 {
    return -1
}

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
        
    private let sortAlgorithm: SortAlgorithm = .bitonic
            
    private enum SortAlgorithm {
        case bitonic
        case radix
    }

    init() {
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
            
            self.tileBoundsEncoder = try TileBoundsEncoder(device: device, library: library)
            self.coverageEncoder = try CoverageEncoder(device: device, library: library)
            self.scatterEncoder = try ScatterEncoder(device: device, library: library)
            self.sortKeyGenEncoder = try SortKeyGenEncoder(device: device, library: library)
            self.bitonicSortEncoder = try BitonicSortEncoder(device: device, library: library)
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
            activeTileCount: activeTileCount
        )
        
        guard let commandBuffer = self.queue.makeCommandBuffer() else { return -1 }
        
        return self.encodeAndRunRender(
            commandBuffer: commandBuffer,
            ordered: ordered,
            colorOutPtr: colorOutPtr,
            depthOutPtr: depthOutPtr,
            alphaOutPtr: alphaOutPtr,
            params: params
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
        
        guard let assignment = self.buildTileAssignmentsGPU(commandBuffer: commandBuffer, gaussianCount: count, gaussianBuffers: inputs, params: params, frame: frame) else {
            self.releaseFrame(index: slotIndex)
            return -3
        }
        
        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            assignment: assignment,
            gaussianBuffers: inputs,
            params: params,
            frame: frame
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
            params: params
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
            frame: frame
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
            params: params
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
            frame: frame
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
            slotIndex: slotIndex
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
    
    internal func buildTileAssignmentsGPU(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        gaussianBuffers: GaussianInputBuffers,
        params: RenderParams,
        frame: FrameResources
    ) -> TileAssignmentBuffers? {
        let tileCount = Int(params.tilesX * params.tilesY)
        let perTileLimit = (params.maxPerTile == 0) ? UInt32(max(gaussianCount, 1)) : params.maxPerTile
        
        let baseCapacity = max(tileCount * Int(perTileLimit), 1)
        // Conservatively allow each gaussian to touch up to 8 tiles (radius may span multiple tiles).
        let overlapCapacity = gaussianCount * 8
        let forcedCapacity = max(baseCapacity, overlapCapacity)
        guard self.prepareTileBuilderResources(frame: frame, gaussianCount: gaussianCount, tileCount: tileCount, maxPerTile: Int(perTileLimit), forcedCapacity: forcedCapacity) else {
            return nil
        }

        // Clear tile assignment buffers to sentinel values to avoid reading stale data when max capacity exceeds actual assignments.
//        if let blit = commandBuffer.makeBlitCommandEncoder() {
//            blit.fill(buffer: frame.tileIndices!, range: 0 ..< frame.tileIndices!.length, value: 0xFF)
//            blit.fill(buffer: frame.tileIds!, range: 0 ..< frame.tileIds!.length, value: 0xFF)
//            blit.endEncoding()
//        }
        
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
        frame: FrameResources
    ) -> OrderedGaussianBuffers? {
        let maxAssignments = assignment.maxAssignments
        guard self.prepareOrderedBuffers(frame: frame, maxAssignments: maxAssignments, tileCount: assignment.tileCount) else { return nil }
        
        guard let sortKeysBuffer = self.ensureBuffer(&frame.sortKeys, length: frame.tileAssignmentPaddedCapacity * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared, label: "SortKeys"),
              let sortedIndicesBuffer = self.ensureBuffer(&frame.sortedIndices, length: frame.tileAssignmentPaddedCapacity * MemoryLayout<Int32>.stride, options: .storageModeShared, label: "SortedIndices")
        else { return nil }

        let totalAssignments = Int(0) // Not needed for indirect dispatch if we use buffers correctly
        
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
            _ = self.ensureRadixBuffers(frame: frame, paddedCapacity: frame.tileAssignmentPaddedCapacity)
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
                paddedCapacity: frame.tileAssignmentPaddedCapacity
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
            activeTileCount: frame.activeTileCount!
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
            activeTileCount: frame.activeTileCount!
        )
    }
        
    
    private func submitRender(
        commandBuffer: MTLCommandBuffer,
        ordered: OrderedGaussianBuffers,
        params: RenderParams,
        frame: FrameResources,
        slotIndex: Int
    ) -> Bool {
        guard let outputs = frame.outputBuffers, let dispatchArgs = frame.dispatchArgs else { return false }
        
        self.renderEncoder.encode(
            commandBuffer: commandBuffer,
            orderedBuffers: ordered,
            outputBuffers: outputs,
            params: params,
            dispatchArgs: dispatchArgs,
            dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )

        return true
    }
    
    private func encodeAndRunRender(
        commandBuffer: MTLCommandBuffer,
        ordered: OrderedGaussianBuffers,
        colorOutPtr: UnsafeMutablePointer<Float>?,
        depthOutPtr: UnsafeMutablePointer<Float>?,
        alphaOutPtr: UnsafeMutablePointer<Float>?,
        params: RenderParams
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
            params: params
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
        params: RenderParams
    ) -> Int32 {
        let cpuReadback = !(colorOutPtr == nil && depthOutPtr == nil && alphaOutPtr == nil)
        guard self.submitRender(
            commandBuffer: commandBuffer,
            ordered: ordered,
            params: params,
            frame: frame,
            slotIndex: slotIndex
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
            let _ = self.ensureBuffer(&frame.tileIndices, length: tileIndexBytes, options: .storageModeShared, label: "TileIndices"),
            let _ = self.ensureBuffer(&frame.tileIds, length: tileIdBytes, options: .storageModeShared, label: "TileIds")
        else {
            return false
        }
        
        // Initialize header
        let headerPtr = frame.tileAssignmentHeader!.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.maxAssignments = UInt32(totalAssignments)
        headerPtr.pointee.paddedCount = UInt32(frame.tileAssignmentPaddedCapacity)
        headerPtr.pointee.totalAssignments = UInt32(0)
        
        return true
    }
    
    internal func prepareOrderedBuffers(frame: FrameResources, maxAssignments: Int, tileCount: Int) -> Bool {
        let assignmentCount = max(1, maxAssignments)
        let headerCount = max(1, tileCount)
        guard
            self.ensureBuffer(&frame.orderedHeaders, length: headerCount * MemoryLayout<GaussianHeader>.stride, options: .storageModeShared, label: "OrderedHeaders") != nil,
            self.ensureBuffer(&frame.packedMeans, length: assignmentCount * 8, options: .storageModeShared, label: "PackedMeans") != nil,
            self.ensureBuffer(&frame.packedConics, length: assignmentCount * 16, options: .storageModeShared, label: "PackedConics") != nil,
            self.ensureBuffer(&frame.packedColors, length: assignmentCount * 12, options: .storageModeShared, label: "PackedColors") != nil,
            self.ensureBuffer(&frame.packedOpacities, length: assignmentCount * 4, options: .storageModeShared, label: "PackedOpacities") != nil,
            self.ensureBuffer(&frame.packedDepths, length: assignmentCount * 4, options: .storageModeShared, label: "PackedDepths") != nil,
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

@_cdecl("gaussian_renderer_debug_dump_tile_pipeline_buffers")
public func gaussian_renderer_debug_dump_tile_pipeline_buffers(
    gaussianCount: Int32,
    meansPtr: UnsafePointer<Float>?,
    scalesPtr: UnsafePointer<Float>?,
    rotationsPtr: UnsafePointer<Float>?,
    harmonicsPtr: UnsafePointer<Float>?,
    opacityPtr: UnsafePointer<Float>?,
    shComponents: Int32,
    viewMatrixPtr: UnsafePointer<Float>?,
    projectionMatrixPtr: UnsafePointer<Float>?,
    cameraCenterPtr: UnsafePointer<Float>?,
    pixelFactor: Float,
    focalX: Float,
    focalY: Float,
    nearPlane: Float,
    farPlane: Float,
    width: Int32,
    height: Int32,
    tileWidth: Int32,
    tileHeight: Int32,
    tilesX: Int32,
    tilesY: Int32,
    maxPerTile: Int32,
    whiteBackground: Int32,
    _tileAssignmentHeaderOutPtr: UnsafeMutableRawPointer?,
    _tileIndicesOutPtr: UnsafeMutableRawPointer?,
    _tileIdsOutPtr: UnsafeMutableRawPointer?,
    _totalAssignmentsOutPtr: UnsafeMutableRawPointer?,
    _boundsOutPtr: UnsafeMutableRawPointer?,
    _coverageOutPtr: UnsafeMutableRawPointer?,
    _offsetsOutPtr: UnsafeMutableRawPointer?
) -> Int32 {
    let renderer = Renderer.shared

    guard
        let meansPtr,
        let scalesPtr,
        let rotationsPtr,
        let harmonicsPtr,
        let opacityPtr,
        let viewPtr = viewMatrixPtr,
        let projPtr = projectionMatrixPtr,
        let cameraPtr = cameraCenterPtr
    else {
        return -1
    }
    
    let count = Int(gaussianCount)
    
    // Prepare world buffers
    guard let worldBuffers = renderer.prepareWorldBuffers(
        count: count,
        meansPtr: meansPtr,
        scalesPtr: scalesPtr,
        rotationsPtr: rotationsPtr,
        harmonicsPtr: harmonicsPtr,
        opacitiesPtr: opacityPtr,
        shComponents: Int(shComponents)
    ) else {
        return -2
    }
    
    // Prepare camera uniforms
    let cameraUniforms = CameraUniformsSwift(
        viewMatrix: matrixFromPointer(viewPtr),
        projectionMatrix: matrixFromPointer(projPtr),
        cameraCenter: SIMD3<Float>(cameraPtr[0], cameraPtr[1], cameraPtr[2]),
        pixelFactor: pixelFactor,
        focalX: focalX,
        focalY: focalY,
        width: Float(width),
        height: Float(height),
        nearPlane: max(max(nearPlane, 0.001), 0.2),
        farPlane: farPlane,
        shComponents: UInt32(max(shComponents, 0)),
        gaussianCount: UInt32(count),
        padding0: 0,
        padding1: 0
    )
    
    // Prepare render parameters
    let params = RenderParams(
        width: UInt32(width),
        height: UInt32(height),
        tileWidth: UInt32(tileWidth),
        tileHeight: UInt32(tileHeight),
        tilesX: UInt32(tilesX),
        tilesY: UInt32(tilesY),
        maxPerTile: UInt32(maxPerTile),
        whiteBackground: UInt32(whiteBackground),
        activeTileCount: 0,
        gaussianCount: UInt32(count)
    )
    
    // Acquire a frame for rendering
    guard let (frame, slotIndex) = renderer.acquireFrame(width: Int(width), height: Int(height)) else { return -5 }
    defer { renderer.releaseFrame(index: slotIndex) }

    guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
        return -1
    }

    // Use shared gaussian buffers (not per-frame)
    guard let gaussianBuffers = renderer.prepareGaussianBuffers(count: count) else {
        return -2
    }

    // 1. Project Gaussians
    renderer.projectEncoder.encodeForRender(
        commandBuffer: commandBuffer,
        gaussianCount: count,
        worldBuffers: worldBuffers,
        cameraUniforms: cameraUniforms,
        gaussianBuffers: gaussianBuffers
    )
    
    // 2. Build Tile Assignments (Bounds, Coverage, Scatter)
    guard let assignment = renderer.buildTileAssignmentsGPU(commandBuffer: commandBuffer, gaussianCount: count, gaussianBuffers: gaussianBuffers, params: params, frame: frame) else {
        return -3
    }

    // Commit and wait for all commands up to this point to complete
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Copy debug buffers back to CPU pointers
    guard let blitCommand = renderer.queue.makeCommandBuffer(),
          let blitEncoder = blitCommand.makeBlitCommandEncoder() else {
        return -4
    }
    
    // Revised strategy: Create shared buffers, schedule all copies, commit, wait, then memcpy.
    var copyTasks: [(MTLBuffer, UnsafeMutableRawPointer)] = []
    
    func scheduleCopy(sourceBuffer: MTLBuffer?, destinationPtr: UnsafeMutableRawPointer?) {
        guard let src = sourceBuffer, let dstPtr = destinationPtr else { return }
        guard let sharedBuffer = renderer.device.makeBuffer(length: src.length, options: .storageModeShared) else { return }
        blitEncoder.copy(from: src, sourceOffset: 0, to: sharedBuffer, destinationOffset: 0, size: src.length)
        copyTasks.append((sharedBuffer, dstPtr))
    }

    scheduleCopy(sourceBuffer: assignment.header, destinationPtr: _tileAssignmentHeaderOutPtr)
    scheduleCopy(sourceBuffer: assignment.tileIndices, destinationPtr: _tileIndicesOutPtr)
    scheduleCopy(sourceBuffer: assignment.tileIds, destinationPtr: _tileIdsOutPtr)
    scheduleCopy(sourceBuffer: frame.boundsBuffer, destinationPtr: _boundsOutPtr)
    scheduleCopy(sourceBuffer: frame.coverageBuffer, destinationPtr: _coverageOutPtr)
    scheduleCopy(sourceBuffer: frame.offsetsBuffer, destinationPtr: _offsetsOutPtr)
    
    blitEncoder.endEncoding()
    blitCommand.commit()
    blitCommand.waitUntilCompleted()
    
    for (sharedBuf, dstPtr) in copyTasks {
        memcpy(dstPtr, sharedBuf.contents(), sharedBuf.length)
    }

    // Total Assignments (from header)
    if let out = _totalAssignmentsOutPtr {
       let headerBuffer = assignment.header
        // We already copied header to _tileAssignmentHeaderOutPtr if it was provided.
        // If not, we need to read it. Or we can just read from the task if we have it.
        // Let's just do a separate read for simplicity if needed, or reuse.
        
        // Just re-read from the shared buffer that contains the header
        if let task = copyTasks.first(where: { $0.1 == _tileAssignmentHeaderOutPtr }) {
             let header = task.0.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1).pointee
             out.assumingMemoryBound(to: Int32.self).pointee = Int32(header.totalAssignments)
        } else {
             // Fallback if header ptr wasn't asked for but count was
             guard let sharedHeader = renderer.device.makeBuffer(length: headerBuffer.length, options: .storageModeShared) else { return -4 }
             let cmd = renderer.queue.makeCommandBuffer()!
             let blit = cmd.makeBlitCommandEncoder()!
             blit.copy(from: headerBuffer, sourceOffset: 0, to: sharedHeader, destinationOffset: 0, size: headerBuffer.length)
             blit.endEncoding()
             cmd.commit()
             cmd.waitUntilCompleted()
             let header = sharedHeader.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1).pointee
             out.assumingMemoryBound(to: Int32.self).pointee = Int32(header.totalAssignments)
        }
    }

    return 0
}

@_cdecl("gaussian_renderer_debug_dump_raw")
public func gaussian_renderer_debug_dump_raw(
    gaussianCount: Int32,
    meansPtr: UnsafePointer<Float>?,
    conicsPtr: UnsafePointer<Float>?,
    colorsPtr: UnsafePointer<Float>?,
    opacityPtr: UnsafePointer<Float>?,
    depthsPtr: UnsafePointer<Float>?,
    radiiPtr: UnsafePointer<Float>?,
    width: Int32,
    height: Int32,
    tileWidth: Int32,
    tileHeight: Int32,
    tilesX: Int32,
    tilesY: Int32,
    maxPerTile: Int32,
    whiteBackground: Int32,
    _tileAssignmentHeaderOutPtr: UnsafeMutableRawPointer?,
    _tileIndicesOutPtr: UnsafeMutableRawPointer?,
    _tileIdsOutPtr: UnsafeMutableRawPointer?,
    _totalAssignmentsOutPtr: UnsafeMutableRawPointer?,
    _boundsOutPtr: UnsafeMutableRawPointer?,
    _coverageOutPtr: UnsafeMutableRawPointer?,
    _offsetsOutPtr: UnsafeMutableRawPointer?,
    _sortedIndicesOutPtr: UnsafeMutableRawPointer?,
    _packedMeansOutPtr: UnsafeMutableRawPointer?,
    _dispatchArgsOutPtr: UnsafeMutableRawPointer?,
    _sortKeysOutPtr: UnsafeMutableRawPointer?
) -> Int32 {
    let renderer = Renderer.shared
    guard
        let meansPtr,
        let conicsPtr,
        let colorsPtr,
        let opacityPtr,
        let depthsPtr,
        let radiiPtr
    else {
        return -1
    }
    
    let pixelCount = Int(width * height)
    if gaussianCount == 0 || pixelCount == 0 {
        return 0
    }
    
    let count = Int(gaussianCount)
    
    // Acquire a frame for rendering
    guard let (frame, slotIndex) = renderer.acquireFrame(width: Int(width), height: Int(height)) else { return -5 }
    defer { renderer.releaseFrame(index: slotIndex) }

    // Create temporary buffers for input
    guard
        let meansBuffer = renderer.makeBuffer(ptr: meansPtr, count: count * 2),
        let conicsBuffer = renderer.makeBuffer(ptr: conicsPtr, count: count * 4),
        let colorsBuffer = renderer.makeBuffer(ptr: colorsPtr, count: count * 3),
        let opacityBuffer = renderer.makeBuffer(ptr: opacityPtr, count: count),
        let depthsBuffer = renderer.makeBuffer(ptr: depthsPtr, count: count),
        let radiiBuffer = renderer.makeBuffer(ptr: radiiPtr, count: count),
        let maskBuffer = renderer.device.makeBuffer(length: count, options: .storageModeShared)
    else {
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
    
    guard let commandBuffer = renderer.queue.makeCommandBuffer() else { 
        return -1 
    }
    
    let params = RenderParams(
        width: UInt32(width),
        height: UInt32(height),
        tileWidth: UInt32(tileWidth),
        tileHeight: UInt32(tileHeight),
        tilesX: UInt32(tilesX),
        tilesY: UInt32(tilesY),
        maxPerTile: UInt32(maxPerTile),
        whiteBackground: UInt32(whiteBackground),
        activeTileCount: 0,
        gaussianCount: UInt32(count)
    )
    
    guard let assignment = renderer.buildTileAssignmentsGPU(commandBuffer: commandBuffer, gaussianCount: count, gaussianBuffers: inputs, params: params, frame: frame) else {
        return -3
    }
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    guard let commandBuffer2 = renderer.queue.makeCommandBuffer() else { return -1 }

    guard let ordered = renderer.buildOrderedGaussians(
        commandBuffer: commandBuffer2,
        gaussianCount: count,
        assignment: assignment,
        gaussianBuffers: inputs,
        params: params,
        frame: frame
    ) else {
        return -4
    }
    
    commandBuffer2.commit()
    commandBuffer2.waitUntilCompleted()

    if commandBuffer2.status == .error {
        print("CommandBuffer Error: \(String(describing: commandBuffer2.error))")
        return -5
    }

    // Copy debug buffers back to CPU pointers
    guard let blitCommand = renderer.queue.makeCommandBuffer(),
          let blitEncoder = blitCommand.makeBlitCommandEncoder() else {
        return -4
    }
    
    var copyTasks: [(MTLBuffer, UnsafeMutableRawPointer)] = []
    
    func scheduleCopy(sourceBuffer: MTLBuffer?, destinationPtr: UnsafeMutableRawPointer?) {
        guard let src = sourceBuffer, let dstPtr = destinationPtr else { return }
        guard let sharedBuffer = renderer.device.makeBuffer(length: src.length, options: .storageModeShared) else { return }
        blitEncoder.copy(from: src, sourceOffset: 0, to: sharedBuffer, destinationOffset: 0, size: src.length)
        copyTasks.append((sharedBuffer, dstPtr))
    }

    scheduleCopy(sourceBuffer: assignment.header, destinationPtr: _tileAssignmentHeaderOutPtr)
    scheduleCopy(sourceBuffer: assignment.tileIndices, destinationPtr: _tileIndicesOutPtr)
    scheduleCopy(sourceBuffer: assignment.tileIds, destinationPtr: _tileIdsOutPtr)
    scheduleCopy(sourceBuffer: frame.boundsBuffer, destinationPtr: _boundsOutPtr)
    scheduleCopy(sourceBuffer: frame.coverageBuffer, destinationPtr: _coverageOutPtr)
    scheduleCopy(sourceBuffer: frame.offsetsBuffer, destinationPtr: _offsetsOutPtr)
    scheduleCopy(sourceBuffer: frame.sortedIndices, destinationPtr: _sortedIndicesOutPtr)
    scheduleCopy(sourceBuffer: ordered.means, destinationPtr: _packedMeansOutPtr)
    scheduleCopy(sourceBuffer: frame.dispatchArgs, destinationPtr: _dispatchArgsOutPtr)
    scheduleCopy(sourceBuffer: frame.sortKeys, destinationPtr: _sortKeysOutPtr)
    
    blitEncoder.endEncoding()
    blitCommand.commit()
    blitCommand.waitUntilCompleted()
    
    for (sharedBuf, dstPtr) in copyTasks {
        memcpy(dstPtr, sharedBuf.contents(), sharedBuf.length)
    }

    if let out = _totalAssignmentsOutPtr {
       let headerBuffer = assignment.header
        if let task = copyTasks.first(where: { $0.1 == _tileAssignmentHeaderOutPtr }) {
             let header = task.0.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1).pointee
             out.assumingMemoryBound(to: Int32.self).pointee = Int32(header.totalAssignments)
        } else {
             guard let sharedHeader = renderer.device.makeBuffer(length: headerBuffer.length, options: .storageModeShared) else { return -4 }
             let cmd = renderer.queue.makeCommandBuffer()!
             let blit = cmd.makeBlitCommandEncoder()!
             blit.copy(from: headerBuffer, sourceOffset: 0, to: sharedHeader, destinationOffset: 0, size: headerBuffer.length)
             blit.endEncoding()
             cmd.commit()
             cmd.waitUntilCompleted()
             let header = sharedHeader.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1).pointee
             out.assumingMemoryBound(to: Int32.self).pointee = Int32(header.totalAssignments)
        }
    }

    return 0
}
