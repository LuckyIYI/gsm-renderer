import Metal
import RendererTypes

/// Uniforms for foveated stereo rendering (matches Metal shader struct)
struct FoveatedStereoUniforms {
    var width: UInt32
    var height: UInt32
    var tileWidth: UInt32
    var tileHeight: UInt32
    var tilesX: UInt32
    var tilesY: UInt32
    var viewIndex: UInt32
    var padding: UInt32

    init(
        width: Int,
        height: Int,
        tileWidth: Int = 16,
        tileHeight: Int = 16
    ) {
        self.width = UInt32(width)
        self.height = UInt32(height)
        self.tileWidth = UInt32(tileWidth)
        self.tileHeight = UInt32(tileHeight)
        self.tilesX = UInt32((width + tileWidth - 1) / tileWidth)
        self.tilesY = UInt32((height + tileHeight - 1) / tileHeight)
        self.viewIndex = 0
        self.padding = 0
    }
}

/// Per-view uniforms for stereo rendering (matches Metal shader struct)
struct StereoViewUniforms {
    var viewportX: Float
    var viewportY: Float
    var viewportWidth: Float
    var viewportHeight: Float
    var renderTargetArrayIndex: UInt32
    var padding0: UInt32
    var padding1: UInt32
    var padding2: UInt32

    init(viewport: MTLViewport, renderTargetArrayIndex: Int) {
        self.viewportX = Float(viewport.originX)
        self.viewportY = Float(viewport.originY)
        self.viewportWidth = Float(viewport.width)
        self.viewportHeight = Float(viewport.height)
        self.renderTargetArrayIndex = UInt32(renderTargetArrayIndex)
        self.padding0 = 0
        self.padding1 = 0
        self.padding2 = 0
    }
}

/// Encoder for foveated stereo rendering using rasterization pipeline
/// Replaces compute-based rendering for Vision Pro Compositor Services compatibility
final class FoveatedStereoEncoder {
    private let device: MTLDevice

    // Render pipelines (created on demand based on pixel formats)
    private var pipelineCache: [PipelineCacheKey: MTLRenderPipelineState] = [:]
    private var pipelineWithDepthCache: [PipelineCacheKey: MTLRenderPipelineState] = [:]

    // Shader functions
    private let vertexFunction: MTLFunction
    private let fragmentFunction: MTLFunction
    private let fragmentWithDepthFunction: MTLFunction

    // Pipeline cache key
    private struct PipelineCacheKey: Hashable {
        let colorFormat: MTLPixelFormat
        let depthFormat: MTLPixelFormat
        let sampleCount: Int
    }

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device

        guard let vertexFn = library.makeFunction(name: "foveatedStereoVertex") else {
            throw RendererError.failedToCreatePipeline("foveatedStereoVertex not found")
        }
        self.vertexFunction = vertexFn

        guard let fragmentFn = library.makeFunction(name: "foveatedStereoFragment") else {
            throw RendererError.failedToCreatePipeline("foveatedStereoFragment not found")
        }
        self.fragmentFunction = fragmentFn

        guard let fragmentWithDepthFn = library.makeFunction(name: "foveatedStereoFragmentWithDepth") else {
            throw RendererError.failedToCreatePipeline("foveatedStereoFragmentWithDepth not found")
        }
        self.fragmentWithDepthFunction = fragmentWithDepthFn
    }

    /// Get or create render pipeline for the given formats
    private func getPipeline(
        colorFormat: MTLPixelFormat,
        depthFormat: MTLPixelFormat,
        sampleCount: Int,
        withDepthOutput: Bool
    ) throws -> MTLRenderPipelineState {
        let key = PipelineCacheKey(
            colorFormat: colorFormat,
            depthFormat: depthFormat,
            sampleCount: sampleCount
        )

        let cache = withDepthOutput ? pipelineWithDepthCache : pipelineCache

        if let existing = cache[key] {
            return existing
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = withDepthOutput ? "FoveatedStereoWithDepth" : "FoveatedStereo"
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = withDepthOutput ? fragmentWithDepthFunction : fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = colorFormat

        // Enable blending for proper alpha compositing
        descriptor.colorAttachments[0].isBlendingEnabled = false  // We do blending in shader

        // Depth configuration
        if depthFormat != .invalid {
            descriptor.depthAttachmentPixelFormat = depthFormat
        }

        // Stereo rendering configuration
        descriptor.maxVertexAmplificationCount = 2
        descriptor.inputPrimitiveTopology = .triangle

        // Multi-sample support
        descriptor.rasterSampleCount = sampleCount

        let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)

        if withDepthOutput {
            pipelineWithDepthCache[key] = pipeline
        } else {
            pipelineCache[key] = pipeline
        }

        return pipeline
    }

    /// Encode foveated stereo render pass
    /// - Parameters:
    ///   - commandBuffer: The command buffer to encode into
    ///   - drawable: The foveated stereo drawable from Compositor Services
    ///   - configuration: Stereo camera configuration
    ///   - tileHeaders: Buffer containing per-tile gaussian headers
    ///   - renderData: Buffer containing gaussian render data
    ///   - sortedGaussianIndices: Buffer containing depth-sorted gaussian indices
    ///   - width: Render width in pixels
    ///   - height: Render height in pixels
    func encode(
        commandBuffer: MTLCommandBuffer,
        drawable: FoveatedStereoDrawable,
        configuration: FoveatedStereoConfiguration,
        tileHeaders: MTLBuffer,
        renderData: MTLBuffer,
        sortedGaussianIndices: MTLBuffer,
        width: Int,
        height: Int
    ) throws {
        // Create render pass descriptor
        let renderPassDescriptor = MTLRenderPassDescriptor()

        // Color attachment
        renderPassDescriptor.colorAttachments[0].texture = drawable.colorTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        // Depth attachment (optional)
        let hasDepth = drawable.depthTexture != nil
        if let depthTexture = drawable.depthTexture {
            renderPassDescriptor.depthAttachment.texture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.storeAction = .store
            renderPassDescriptor.depthAttachment.clearDepth = 1.0
        }

        // Apply rasterization rate map for foveated rendering
        if let rateMap = drawable.rasterizationRateMap {
            renderPassDescriptor.rasterizationRateMap = rateMap
        }

        // For layered textures, set render target array length
        if configuration.layout == .layered {
            renderPassDescriptor.renderTargetArrayLength = 2
        }

        // Get or create pipeline
        let pipeline = try getPipeline(
            colorFormat: drawable.colorPixelFormat,
            depthFormat: hasDepth ? drawable.depthPixelFormat : .invalid,
            sampleCount: 1,
            withDepthOutput: hasDepth
        )

        // Create render encoder
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            throw RendererError.encoderCreationFailed(stage: "FoveatedStereoRender")
        }
        encoder.label = "FoveatedStereoRender"

        encoder.setRenderPipelineState(pipeline)

        // Set up stereo view mappings for vertex amplification
        var viewMappings: [MTLVertexAmplificationViewMapping] = []

        let leftMapping = MTLVertexAmplificationViewMapping(
            viewportArrayIndexOffset: 0,
            renderTargetArrayIndexOffset: UInt32(configuration.leftEye.renderTargetArrayIndex)
        )
        let rightMapping = MTLVertexAmplificationViewMapping(
            viewportArrayIndexOffset: 1,
            renderTargetArrayIndexOffset: UInt32(configuration.rightEye.renderTargetArrayIndex)
        )
        viewMappings = [leftMapping, rightMapping]

        encoder.setVertexAmplificationCount(2, viewMappings: &viewMappings)

        // Set viewports for both eyes
        let viewports = [
            configuration.leftEye.viewport,
            configuration.rightEye.viewport
        ]
        encoder.setViewports(viewports)

        // Prepare per-view uniforms
        var viewUniforms: [StereoViewUniforms] = [
            StereoViewUniforms(
                viewport: configuration.leftEye.viewport,
                renderTargetArrayIndex: configuration.leftEye.renderTargetArrayIndex
            ),
            StereoViewUniforms(
                viewport: configuration.rightEye.viewport,
                renderTargetArrayIndex: configuration.rightEye.renderTargetArrayIndex
            )
        ]

        // Set vertex stage buffers
        encoder.setVertexBytes(
            &viewUniforms,
            length: MemoryLayout<StereoViewUniforms>.stride * 2,
            index: 0
        )

        // Prepare fragment uniforms
        var uniforms = FoveatedStereoUniforms(width: width, height: height)

        // Set fragment stage buffers
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<FoveatedStereoUniforms>.stride, index: 0)
        encoder.setFragmentBuffer(tileHeaders, offset: 0, index: 1)
        encoder.setFragmentBuffer(renderData, offset: 0, index: 2)
        encoder.setFragmentBuffer(sortedGaussianIndices, offset: 0, index: 3)
        encoder.setFragmentBytes(
            &viewUniforms,
            length: MemoryLayout<StereoViewUniforms>.stride * 2,
            index: 4
        )

        // Draw fullscreen triangle (3 vertices, amplified to both eyes)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        encoder.endEncoding()
    }

    /// Encode single-eye render pass (for non-stereo or dedicated layout)
    func encodeSingleEye(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        rasterizationRateMap: MTLRasterizationRateMap?,
        viewport: MTLViewport,
        tileHeaders: MTLBuffer,
        renderData: MTLBuffer,
        sortedGaussianIndices: MTLBuffer,
        width: Int,
        height: Int,
        colorPixelFormat: MTLPixelFormat,
        depthPixelFormat: MTLPixelFormat
    ) throws {
        // Create render pass descriptor
        let renderPassDescriptor = MTLRenderPassDescriptor()

        // Color attachment
        renderPassDescriptor.colorAttachments[0].texture = colorTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        // Depth attachment
        let hasDepth = depthTexture != nil
        if let depthTex = depthTexture {
            renderPassDescriptor.depthAttachment.texture = depthTex
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.storeAction = .store
            renderPassDescriptor.depthAttachment.clearDepth = 1.0
        }

        // Apply rasterization rate map
        if let rateMap = rasterizationRateMap {
            renderPassDescriptor.rasterizationRateMap = rateMap
        }

        // Get pipeline (single eye, no amplification needed)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "FoveatedSingleEye"
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = hasDepth ? fragmentWithDepthFunction : fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = colorPixelFormat

        if hasDepth {
            descriptor.depthAttachmentPixelFormat = depthPixelFormat
        }

        descriptor.maxVertexAmplificationCount = 1
        descriptor.inputPrimitiveTopology = .triangle

        let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            throw RendererError.encoderCreationFailed(stage: "FoveatedSingleEyeRender")
        }
        encoder.label = "FoveatedSingleEyeRender"

        encoder.setRenderPipelineState(pipeline)
        encoder.setViewport(viewport)

        // Prepare view uniform for single eye
        var viewUniform = StereoViewUniforms(viewport: viewport, renderTargetArrayIndex: 0)

        encoder.setVertexBytes(&viewUniform, length: MemoryLayout<StereoViewUniforms>.stride, index: 0)

        // Fragment uniforms
        var uniforms = FoveatedStereoUniforms(width: width, height: height)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<FoveatedStereoUniforms>.stride, index: 0)
        encoder.setFragmentBuffer(tileHeaders, offset: 0, index: 1)
        encoder.setFragmentBuffer(renderData, offset: 0, index: 2)
        encoder.setFragmentBuffer(sortedGaussianIndices, offset: 0, index: 3)
        encoder.setFragmentBytes(&viewUniform, length: MemoryLayout<StereoViewUniforms>.stride, index: 4)

        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }
}
