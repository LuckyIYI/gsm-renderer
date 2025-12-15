import Metal
import RendererTypes

/// Check if a pixel format is depth-renderable
private func isDepthRenderable(_ format: MTLPixelFormat) -> Bool {
    switch format {
    case .depth16Unorm, .depth32Float, .depth32Float_stencil8, .depth24Unorm_stencil8:
        return true
    default:
        return false
    }
}

/// Encoder for instanced Gaussian rendering.
/// Handles center-sort stereo and mono rendering modes.
final class InstancedRenderEncoder {

    private enum Constants {
        static let tileSize = MTLSize(width: 32, height: 32, depth: 1)
    }

    private let initializePipeline: MTLRenderPipelineState
    private let centerSortPipeline: MTLRenderPipelineState
    private let postprocessPipeline: MTLRenderPipelineState
    private let monoPipeline: MTLRenderPipelineState
    private let depthStencilState: MTLDepthStencilState
    private let noDepthStencilState: MTLDepthStencilState
    private let quadIndexBuffer: MTLBuffer
    private let quadIndexType: MTLIndexType = .uint32

    init(
        initializePipeline: MTLRenderPipelineState,
        centerSortPipeline: MTLRenderPipelineState,
        postprocessPipeline: MTLRenderPipelineState,
        monoPipeline: MTLRenderPipelineState,
        depthStencilState: MTLDepthStencilState,
        noDepthStencilState: MTLDepthStencilState,
        quadIndexBuffer: MTLBuffer
    ) {
        self.initializePipeline = initializePipeline
        self.centerSortPipeline = centerSortPipeline
        self.postprocessPipeline = postprocessPipeline
        self.monoPipeline = monoPipeline
        self.depthStencilState = depthStencilState
        self.noDepthStencilState = noDepthStencilState
        self.quadIndexBuffer = quadIndexBuffer
    }

    func encodeStereo(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        rasterizationRateMap: MTLRasterizationRateMap?,
        projectedSorted: MTLBuffer,
        header: MTLBuffer,
        drawArgs: MTLBuffer,
        configuration: StereoConfiguration,
        width: Int,
        height: Int
    ) {
        
        let renderPassDesc = MTLRenderPassDescriptor()
        renderPassDesc.colorAttachments[0].texture = colorTexture
        renderPassDesc.colorAttachments[0].loadAction = .clear
        renderPassDesc.colorAttachments[0].storeAction = .store
        renderPassDesc.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

        let renderTargetArrayLength = colorTexture.arrayLength
        renderPassDesc.renderTargetArrayLength = renderTargetArrayLength
        renderPassDesc.rasterizationRateMap = rasterizationRateMap

        let hasValidDepth = depthTexture != nil && isDepthRenderable(depthTexture!.pixelFormat)
        if hasValidDepth {
            renderPassDesc.depthAttachment.texture = depthTexture
            renderPassDesc.depthAttachment.loadAction = .clear
            renderPassDesc.depthAttachment.storeAction = .store
            renderPassDesc.depthAttachment.clearDepth = 1.0
        }

        renderPassDesc.tileWidth = Constants.tileSize.width
        renderPassDesc.tileHeight = Constants.tileSize.height
        renderPassDesc.imageblockSampleLength = initializePipeline.imageblockSampleLength

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc) else { return }
        encoder.label = "InstancedGaussian_CenterSortFoveated"

        encoder.pushDebugGroup("Initialize Fragment Store")
        encoder.setRenderPipelineState(initializePipeline)
        encoder.dispatchThreadsPerTile(Constants.tileSize)
        encoder.popDebugGroup()

        encoder.pushDebugGroup("Draw Gaussians")
        encoder.setRenderPipelineState(centerSortPipeline)
        encoder.setDepthStencilState(noDepthStencilState)

        let leftViewport = MTLViewport(
            originX: configuration.leftEye.viewport.originX,
            originY: configuration.leftEye.viewport.originY,
            width: configuration.leftEye.viewport.width,
            height: configuration.leftEye.viewport.height,
            znear: 0, zfar: 1
        )
        let rightViewport = MTLViewport(
            originX: configuration.rightEye.viewport.originX,
            originY: configuration.rightEye.viewport.originY,
            width: configuration.rightEye.viewport.width,
            height: configuration.rightEye.viewport.height,
            znear: 0, zfar: 1
        )

        let viewports = [leftViewport, rightViewport]
        encoder.setViewports(viewports)

        var viewMappings = [
            MTLVertexAmplificationViewMapping(
                viewportArrayIndexOffset: 0,
                renderTargetArrayIndexOffset: 0
            ),
            MTLVertexAmplificationViewMapping(
                viewportArrayIndexOffset: renderTargetArrayLength == 1 ? 1 : 0,
                renderTargetArrayIndexOffset: renderTargetArrayLength == 1 ? 0 : 1
            ),
        ]
        encoder.setVertexAmplificationCount(2, viewMappings: &viewMappings)

        encoder.setVertexBuffer(projectedSorted, offset: 0, index: 0)

        var uniforms = CenterSortRenderUniforms(
            width: Float(width),
            height: Float(height),
            farPlane: configuration.leftEye.far,
            _pad0: 0
        )
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<CenterSortRenderUniforms>.stride, index: 1)

        encoder.setVertexBuffer(header, offset: 0, index: 2)

        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexType: quadIndexType,
                                      indexBuffer: quadIndexBuffer,
                                      indexBufferOffset: 0,
                                      indirectBuffer: drawArgs,
                                      indirectBufferOffset: 0)

        encoder.popDebugGroup()

        encoder.pushDebugGroup("Postprocess")
        encoder.setRenderPipelineState(postprocessPipeline)
        encoder.setDepthStencilState(depthStencilState)
        encoder.setCullMode(.none)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.popDebugGroup()

        encoder.endEncoding()
    }

    /// Encode mono render
    func encodeMono(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        projectedSorted: MTLBuffer,
        header: MTLBuffer,
        drawArgs: MTLBuffer,
        width: Int,
        height: Int,
        farPlane: Float
    ) {
        let renderPassDesc = MTLRenderPassDescriptor()
        renderPassDesc.colorAttachments[0].texture = colorTexture
        renderPassDesc.colorAttachments[0].loadAction = .clear
        renderPassDesc.colorAttachments[0].storeAction = .store
        renderPassDesc.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

        let hasValidDepth = depthTexture != nil && isDepthRenderable(depthTexture!.pixelFormat)
        if hasValidDepth {
            renderPassDesc.depthAttachment.texture = depthTexture
            renderPassDesc.depthAttachment.loadAction = .clear
            renderPassDesc.depthAttachment.storeAction = .store
            renderPassDesc.depthAttachment.clearDepth = 1.0
        }

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc) else { return }
        encoder.label = "InstancedGaussian_Mono"

        encoder.setRenderPipelineState(monoPipeline)
        encoder.setDepthStencilState(hasValidDepth ? depthStencilState : noDepthStencilState)

        let viewport = MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height), znear: 0, zfar: 1)
        encoder.setViewport(viewport)

        encoder.setVertexBuffer(projectedSorted, offset: 0, index: 0)

        var uniforms = CenterSortRenderUniforms(width: Float(width), height: Float(height), farPlane: farPlane, _pad0: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<CenterSortRenderUniforms>.stride, index: 1)

        encoder.setVertexBuffer(header, offset: 0, index: 2)

        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexType: quadIndexType,
                                      indexBuffer: quadIndexBuffer,
                                      indexBufferOffset: 0,
                                      indirectBuffer: drawArgs,
                                      indirectBufferOffset: 0)

        encoder.endEncoding()
    }
}
