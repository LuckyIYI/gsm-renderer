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

/// Encoder for mesh shader Gaussian rendering.
/// Handles hardware stereo and mono rendering modes.
final class MeshRenderEncoder {

    static let gaussiansPerObjectTG: Int = 64
    static let gaussiansPerMeshTG: Int = 16
    static let meshThreads: Int = 64
    static let centerGaussiansPerObjectTG: Int = 64
    static let centerGaussiansPerMeshTG: Int = 16
    static let centerMeshThreads: Int = 64

    private enum Constants {
        static let tileSize = MTLSize(width: 32, height: 32, depth: 1)
    }

    private let initializePipeline: MTLRenderPipelineState
    private let stereoPipeline: MTLRenderPipelineState
    private let postprocessPipeline: MTLRenderPipelineState
    private let monoPipeline: MTLRenderPipelineState
    private let depthStencilState: MTLDepthStencilState
    private let noDepthStencilState: MTLDepthStencilState

    init(
        initializePipeline: MTLRenderPipelineState,
        stereoPipeline: MTLRenderPipelineState,
        postprocessPipeline: MTLRenderPipelineState,
        monoPipeline: MTLRenderPipelineState,
        depthStencilState: MTLDepthStencilState,
        noDepthStencilState: MTLDepthStencilState
    ) {
        self.initializePipeline = initializePipeline
        self.stereoPipeline = stereoPipeline
        self.postprocessPipeline = postprocessPipeline
        self.monoPipeline = monoPipeline
        self.depthStencilState = depthStencilState
        self.noDepthStencilState = noDepthStencilState
    }

    func encodeStereo(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        rasterizationRateMap: MTLRasterizationRateMap?,
        projectedSorted: MTLBuffer,
        header: MTLBuffer,
        meshDrawArgs: MTLBuffer,
        configuration: StereoConfiguration,
        width: Int,
        height: Int
    ) {
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = colorTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        
        let renderTargetArrayLength = colorTexture.arrayLength
        renderPassDescriptor.renderTargetArrayLength = renderTargetArrayLength
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap

        let hasValidDepth = depthTexture != nil && isDepthRenderable(depthTexture!.pixelFormat)
        if hasValidDepth {
            renderPassDescriptor.depthAttachment.texture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.storeAction = .store
            renderPassDescriptor.depthAttachment.clearDepth = 1.0
        }

        renderPassDescriptor.tileWidth = Constants.tileSize.width
        renderPassDescriptor.tileHeight = Constants.tileSize.height
        renderPassDescriptor.imageblockSampleLength = initializePipeline.imageblockSampleLength

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }
        encoder.label = "MeshGaussian_Stereo"

        encoder.pushDebugGroup("Initialize Fragment Store")
        encoder.setRenderPipelineState(initializePipeline)
        encoder.dispatchThreadsPerTile(Constants.tileSize)
        encoder.popDebugGroup()

        encoder.pushDebugGroup("Draw Gaussians")
        encoder.setRenderPipelineState(stereoPipeline)
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

        encoder.setObjectBuffer(projectedSorted, offset: 0, index: 0)
        encoder.setObjectBuffer(header, offset: 0, index: 2)

        var uniforms = HardwareRenderUniforms(
            width: Float(width),
            height: Float(height),
            farPlane: configuration.leftEye.far,
            _pad0: 0
        )
        
        encoder.setObjectBytes(&uniforms, length: MemoryLayout<HardwareRenderUniforms>.stride, index: 1)
        encoder.setMeshBytes(&uniforms, length: MemoryLayout<HardwareRenderUniforms>.stride, index: 1)

        encoder.drawMeshThreadgroups(
            indirectBuffer: meshDrawArgs,
            indirectBufferOffset: 0,
            threadsPerObjectThreadgroup: MTLSize(width: Self.centerGaussiansPerObjectTG, height: 1, depth: 1),
            threadsPerMeshThreadgroup: MTLSize(width: Self.meshThreads, height: 1, depth: 1)
        )

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
        meshDrawArgs: MTLBuffer,
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

        renderPassDesc.imageblockSampleLength = 8

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc) else { return }
        encoder.label = "MeshGaussian_Mono"

        encoder.setRenderPipelineState(monoPipeline)
        encoder.setDepthStencilState(hasValidDepth ? depthStencilState : noDepthStencilState)

        let viewport = MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height), znear: 0, zfar: 1)
        encoder.setViewport(viewport)

        encoder.setObjectBuffer(projectedSorted, offset: 0, index: 0)
        encoder.setObjectBuffer(header, offset: 0, index: 2) // visibleCount is at offset 0 in HardwareMonoHeader

        var uniforms = HardwareRenderUniforms(width: Float(width), height: Float(height), farPlane: farPlane, _pad0: 0)
        encoder.setObjectBytes(&uniforms, length: MemoryLayout<HardwareRenderUniforms>.stride, index: 1)
        encoder.setMeshBytes(&uniforms, length: MemoryLayout<HardwareRenderUniforms>.stride, index: 1)

        encoder.drawMeshThreadgroups(
            indirectBuffer: meshDrawArgs,
            indirectBufferOffset: 0,
            threadsPerObjectThreadgroup: MTLSize(width: Self.gaussiansPerObjectTG, height: 1, depth: 1),
            threadsPerMeshThreadgroup: MTLSize(width: Self.meshThreads, height: 1, depth: 1)
        )

        encoder.endEncoding()
    }
}
