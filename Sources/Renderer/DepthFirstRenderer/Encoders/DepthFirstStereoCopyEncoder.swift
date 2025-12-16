import Metal
import RendererTypes

/// Encoder that copies the intermediate 2-slice stereo texture array into the final target,
/// using a render pass so we can attach a rasterization rate map (foveation) and use vertex amplification.
final class DepthFirstStereoCopyEncoder {
    private let renderPipeline: MTLRenderPipelineState

    init(device: MTLDevice, library: MTLLibrary, colorFormat: MTLPixelFormat = .rgba16Float) throws {
        guard let vertexFn = library.makeFunction(name: "stereoCopyVertex"),
              let fragmentFn = library.makeFunction(name: "stereoCopyFragment")
        else {
            throw RendererError.failedToCreatePipeline("Stereo copy shaders not found")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        desc.colorAttachments[0].pixelFormat = colorFormat
        desc.inputPrimitiveTopology = .triangle
        desc.maxVertexAmplificationCount = 2
        self.renderPipeline = try device.makeRenderPipelineState(descriptor: desc)
    }

    func encodeRender(
        commandBuffer: MTLCommandBuffer,
        sourceTexture: MTLTexture,
        destinationTexture: MTLTexture,
        rasterizationRateMap: MTLRasterizationRateMap?,
        configuration: StereoConfiguration
    ) {
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = destinationTexture
        rpd.colorAttachments[0].loadAction = .dontCare
        rpd.colorAttachments[0].storeAction = .store
        rpd.renderTargetArrayLength = destinationTexture.arrayLength
        rpd.rasterizationRateMap = rasterizationRateMap

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) else { return }
        encoder.label = "DepthFirstStereoCopy"

        encoder.setRenderPipelineState(self.renderPipeline)

        let leftViewport = MTLViewport(
            originX: configuration.leftEye.viewport.originX,
            originY: configuration.leftEye.viewport.originY,
            width: configuration.leftEye.viewport.width,
            height: configuration.leftEye.viewport.height,
            znear: 0,
            zfar: 1
        )
        let rightViewport = MTLViewport(
            originX: configuration.rightEye.viewport.originX,
            originY: configuration.rightEye.viewport.originY,
            width: configuration.rightEye.viewport.width,
            height: configuration.rightEye.viewport.height,
            znear: 0,
            zfar: 1
        )
        encoder.setViewports([leftViewport, rightViewport])

        encoder.setFragmentTexture(sourceTexture, index: 0)

        var viewMappings = [
            MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: 0, renderTargetArrayIndexOffset: 0),
            MTLVertexAmplificationViewMapping(
                viewportArrayIndexOffset: 1,
                renderTargetArrayIndexOffset: destinationTexture.arrayLength == 1 ? 0 : 1
            ),
        ]
        encoder.setVertexAmplificationCount(2, viewMappings: &viewMappings)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }
}
