import Foundation
import Metal
import simd

// MARK: - C Bridge Functions (DISABLED - under refactoring)

// The C API is temporarily disabled while RenderParams is being simplified.
// The Swift API is the recommended interface.

/*
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
         activeTileIndices: frame.activeTileIndices,
         activeTileCount: frame.activeTileCount,
         precision: .float32,
         interleavedGaussians: nil,
         sortedIndices: nil
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
         frame: frame,
         precision: Renderer.shared.precisionSetting
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
         gaussianBuffers: gaussianBuffers,
         precision: .float32
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
         frame: frame,
         precision: Renderer.shared.precisionSetting
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

     let orderedH = frame.orderedHeaders
     let globalH = frame.tileAssignmentHeader
     let ti = frame.tileIndices
     let tids = frame.tileIds

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

     let r = frame.boundsBuffer
     let c = frame.coverageBuffer
     let o = frame.offsetsBuffer

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
         gaussianBuffers: gaussianBuffers,
         precision: .float32
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
         frame: frame,
         precision: renderer.precisionSetting
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
 */
