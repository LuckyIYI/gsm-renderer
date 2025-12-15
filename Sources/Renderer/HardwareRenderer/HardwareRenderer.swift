import Foundation
@preconcurrency import Metal
import RendererTypes
import simd

// MARK: - Resources

/// Resources for center-sort stereo rendering (single shared sort).
/// Includes buffers for both instanced and mesh-shader draw backends.
struct HardwareCenterSortResources {
    let projected: MTLBuffer
    let projectedSorted: MTLBuffer
    let depthKeys: MTLBuffer
    let sortedIndices: MTLBuffer
    let scratchKeys: MTLBuffer
    let scratchIndices: MTLBuffer
    let visibleCount: MTLBuffer
    let visibilityMarks: MTLBuffer
    let compactionScratch: MTLBuffer

    let histogram: MTLBuffer
    let blockSums: MTLBuffer
    let scannedHistogram: MTLBuffer

    let header: MTLBuffer
    let sortHeader: MTLBuffer

    let sortDispatch: MTLBuffer
    let scanDispatch: MTLBuffer
    let reorderDispatch: MTLBuffer

    /// Instanced backend draw args (`DrawIndexedIndirectArgs`).
    let drawArgs: MTLBuffer
    /// Mesh backend draw args (`threadgroupsPerGrid` = 3x UInt32).
    let meshDrawArgs: MTLBuffer

    let paddedMax: Int
    let numBlocks: Int
}

/// Resources for mono rendering (single eye).
/// Includes buffers for both instanced and mesh-shader draw backends.
struct HardwareMonoResources {
    let projected: MTLBuffer
    let projectedSorted: MTLBuffer
    let depthKeys: MTLBuffer
    let sortedIndices: MTLBuffer
    let scratchKeys: MTLBuffer
    let scratchIndices: MTLBuffer
    let counter: MTLBuffer
    let visibilityMarks: MTLBuffer
    let compactionScratch: MTLBuffer

    let histogram: MTLBuffer
    let blockSums: MTLBuffer
    let scannedHistogram: MTLBuffer

    let header: MTLBuffer
    let sortHeader: MTLBuffer

    let sortDispatch: MTLBuffer
    let scanDispatch: MTLBuffer
    let reorderDispatch: MTLBuffer

    /// `prepareMonoSortKernel` writes `DrawIndexedIndirectArgs` (5x UInt32 = 20 bytes).
    /// Used by the instanced backend; also written for mesh backend to satisfy kernel validation.
    let drawArgs: MTLBuffer
    /// Mesh backend draw args (`threadgroupsPerGrid` = 3x UInt32).
    let meshDrawArgs: MTLBuffer

    let paddedMax: Int
    let numBlocks: Int
}

/// Manages resource allocation for HardwareRenderer.
final class HardwareResourceManager {
    private let device: MTLDevice
    private let radixBlockSize: Int
    private let radixGrainSize: Int

    private var centerSortResources: HardwareCenterSortResources?
    private var monoResources: HardwareMonoResources?
    private var maxCenterSortGaussians: Int = 0
    private var maxMonoGaussians: Int = 0

    init(device: MTLDevice, radixBlockSize: Int = 256, radixGrainSize: Int = 4) {
        self.device = device
        self.radixBlockSize = radixBlockSize
        self.radixGrainSize = radixGrainSize
    }

    func ensureCenterSortResources(gaussianCount: Int) throws -> HardwareCenterSortResources {
        if let existing = centerSortResources, maxCenterSortGaussians >= gaussianCount {
            return existing
        }

        let newMax = max(gaussianCount, maxCenterSortGaussians * 2, 100_000)
        let radixAlignment = radixBlockSize * radixGrainSize
        let paddedMax = ((newMax + radixAlignment - 1) / radixAlignment) * radixAlignment
        let numBlocks = paddedMax / radixAlignment + 4
        let scanBlockSize = 256
        let maxScanBlocks = (paddedMax + scanBlockSize - 1) / scanBlockSize
        let maxScanLevel2Blocks = (maxScanBlocks + scanBlockSize - 1) / scanBlockSize
        let maxScanLevel3Blocks = (maxScanLevel2Blocks + scanBlockSize - 1) / scanBlockSize
        let compactionScratchCount = maxScanBlocks + 1 + maxScanLevel2Blocks + 1 + maxScanLevel3Blocks + 1

        let dataSize = MemoryLayout<CenterSortGaussianData>.stride
        let dispatchArgsSize = MemoryLayout<DispatchIndirectArgs>.stride
        let drawArgsSize = MemoryLayout<DrawIndexedIndirectArgs>.stride

        guard let projected = device.makeBuffer(length: paddedMax * dataSize, options: .storageModePrivate),
              let projectedSorted = device.makeBuffer(length: paddedMax * dataSize, options: .storageModePrivate),
              let depthKeys = device.makeBuffer(length: paddedMax * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let sortedIndices = device.makeBuffer(length: paddedMax * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let scratchKeys = device.makeBuffer(length: paddedMax * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let scratchIndices = device.makeBuffer(length: paddedMax * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let visibleCount = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let visibilityMarks = device.makeBuffer(length: paddedMax * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let compactionScratch = device.makeBuffer(length: compactionScratchCount * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let histogram = device.makeBuffer(length: 256 * numBlocks * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let blockSums = device.makeBuffer(length: numBlocks * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let scannedHistogram = device.makeBuffer(length: 256 * numBlocks * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let header = device.makeBuffer(length: MemoryLayout<CenterSortHeader>.stride, options: .storageModeShared),
              let sortHeader = device.makeBuffer(length: MemoryLayout<DepthFirstHeader>.stride, options: .storageModeShared),
              let sortDispatch = device.makeBuffer(length: dispatchArgsSize, options: .storageModeShared),
              let scanDispatch = device.makeBuffer(length: dispatchArgsSize, options: .storageModeShared),
              let reorderDispatch = device.makeBuffer(length: dispatchArgsSize, options: .storageModeShared),
              let drawArgs = device.makeBuffer(length: drawArgsSize, options: .storageModeShared),
              let meshDrawArgs = device.makeBuffer(length: 12, options: .storageModePrivate)
        else {
            throw RendererError.failedToAllocateBuffer(label: "HardwareCenterSortResources", size: paddedMax * dataSize)
        }

        let resources = HardwareCenterSortResources(
            projected: projected,
            projectedSorted: projectedSorted,
            depthKeys: depthKeys,
            sortedIndices: sortedIndices,
            scratchKeys: scratchKeys,
            scratchIndices: scratchIndices,
            visibleCount: visibleCount,
            visibilityMarks: visibilityMarks,
            compactionScratch: compactionScratch,
            histogram: histogram,
            blockSums: blockSums,
            scannedHistogram: scannedHistogram,
            header: header,
            sortHeader: sortHeader,
            sortDispatch: sortDispatch,
            scanDispatch: scanDispatch,
            reorderDispatch: reorderDispatch,
            drawArgs: drawArgs,
            meshDrawArgs: meshDrawArgs,
            paddedMax: paddedMax,
            numBlocks: numBlocks
        )

        centerSortResources = resources
        maxCenterSortGaussians = newMax
        return resources
    }

    func ensureMonoResources(gaussianCount: Int) throws -> HardwareMonoResources {
        if let existing = monoResources, maxMonoGaussians >= gaussianCount {
            return existing
        }

        let newMax = max(gaussianCount, maxMonoGaussians * 2, 100_000)
        let radixAlignment = radixBlockSize * radixGrainSize
        let paddedMax = ((newMax + radixAlignment - 1) / radixAlignment) * radixAlignment
        let numBlocks = paddedMax / radixAlignment + 4
        let scanBlockSize = 256
        let maxScanBlocks = (paddedMax + scanBlockSize - 1) / scanBlockSize
        let maxScanLevel2Blocks = (maxScanBlocks + scanBlockSize - 1) / scanBlockSize
        let maxScanLevel3Blocks = (maxScanLevel2Blocks + scanBlockSize - 1) / scanBlockSize
        let compactionScratchCount = maxScanBlocks + 1 + maxScanLevel2Blocks + 1 + maxScanLevel3Blocks + 1

        let dataSize = MemoryLayout<InstancedGaussianData>.stride
        let dispatchArgsSize = MemoryLayout<DispatchIndirectArgs>.stride
        let drawArgsSize = MemoryLayout<DrawIndexedIndirectArgs>.stride

        guard let projected = device.makeBuffer(length: paddedMax * dataSize, options: .storageModePrivate),
              let projectedSorted = device.makeBuffer(length: paddedMax * dataSize, options: .storageModePrivate),
              let depthKeys = device.makeBuffer(length: paddedMax * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let sortedIndices = device.makeBuffer(length: paddedMax * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let scratchKeys = device.makeBuffer(length: paddedMax * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let scratchIndices = device.makeBuffer(length: paddedMax * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let counter = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let visibilityMarks = device.makeBuffer(length: paddedMax * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let compactionScratch = device.makeBuffer(length: compactionScratchCount * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let histogram = device.makeBuffer(length: 256 * numBlocks * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let blockSums = device.makeBuffer(length: numBlocks * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let scannedHistogram = device.makeBuffer(length: 256 * numBlocks * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let header = device.makeBuffer(length: MemoryLayout<MonoRenderHeader>.stride, options: .storageModeShared),
              let sortHeader = device.makeBuffer(length: MemoryLayout<DepthFirstHeader>.stride, options: .storageModeShared),
              let sortDispatch = device.makeBuffer(length: dispatchArgsSize, options: .storageModeShared),
              let scanDispatch = device.makeBuffer(length: dispatchArgsSize, options: .storageModeShared),
              let reorderDispatch = device.makeBuffer(length: dispatchArgsSize, options: .storageModeShared),
              let drawArgs = device.makeBuffer(length: drawArgsSize, options: .storageModeShared),
              let meshDrawArgs = device.makeBuffer(length: 12, options: .storageModePrivate)
        else {
            throw RendererError.failedToAllocateBuffer(label: "HardwareMonoResources", size: paddedMax * dataSize)
        }

        let resources = HardwareMonoResources(
            projected: projected,
            projectedSorted: projectedSorted,
            depthKeys: depthKeys,
            sortedIndices: sortedIndices,
            scratchKeys: scratchKeys,
            scratchIndices: scratchIndices,
            counter: counter,
            visibilityMarks: visibilityMarks,
            compactionScratch: compactionScratch,
            histogram: histogram,
            blockSums: blockSums,
            scannedHistogram: scannedHistogram,
            header: header,
            sortHeader: sortHeader,
            sortDispatch: sortDispatch,
            scanDispatch: scanDispatch,
            reorderDispatch: reorderDispatch,
            drawArgs: drawArgs,
            meshDrawArgs: meshDrawArgs,
            paddedMax: paddedMax,
            numBlocks: numBlocks
        )

        monoResources = resources
        maxMonoGaussians = newMax
        return resources
    }
}

public final class HardwareRenderer: GaussianRenderer, @unchecked Sendable {
    public enum DrawBackend: Sendable {
        case meshShaders
        case instanced
    }

    public let device: MTLDevice
    public let precision: RenderPrecision
    public let backToFront: Bool
    public let backend: DrawBackend
    public private(set) var lastGPUTime: Double?

    private let inputIsSRGB: Bool

    // Shared encoders
    private let centerProjectionEncoder: CenterProjectionEncoder
    private let monoProjectionEncoder: MonoProjectionEncoder
    private let prepareDispatchEncoder: PrepareDispatchEncoder
    private let initIndicesEncoder: InitIndicesEncoder
    private let depthRadixSortEncoder: DepthRadixSortEncoder
    private let reorderDataEncoder: ReorderDataEncoder
    private let visibilityCompactionEncoder: VisibilityCompactionEncoder

    // Backends
    private enum BackendImpl {
        case mesh(MeshRenderEncoder)
        case instanced(InstancedRenderEncoder)
    }
    private let backendImpl: BackendImpl

    // Resources
    private let resourceManager: HardwareResourceManager

    public init(
        device: MTLDevice? = nil,
        config: RendererConfig = RendererConfig(),
        backend: DrawBackend = .meshShaders,
        depthSortKeyPrecision: RadixSortKeyPrecision = .bits32
    ) throws {
        let device = device ?? MTLCreateSystemDefaultDevice()
        guard let device else { throw RendererError.deviceNotAvailable }

        self.device = device
        self.precision = config.precision
        self.backToFront = config.backToFront
        self.backend = backend
        self.inputIsSRGB = config.gaussianColorSpace == .srgb

        guard let hardwareLibURL = Bundle.module.url(forResource: "HardwareGaussianShaders", withExtension: "metallib") else {
            throw RendererError.failedToCreatePipeline("HardwareGaussianShaders.metallib not found")
        }
        let hardwareLib = try device.makeLibrary(URL: hardwareLibURL)

        guard let depthFirstLibURL = Bundle.module.url(forResource: "DepthFirstShaders", withExtension: "metallib") else {
            throw RendererError.failedToCreatePipeline("DepthFirstShaders.metallib not found")
        }
        let depthFirstLib = try device.makeLibrary(URL: depthFirstLibURL)

        self.centerProjectionEncoder = try CenterProjectionEncoder(device: device, library: hardwareLib)
        self.monoProjectionEncoder = try MonoProjectionEncoder(device: device, library: hardwareLib)
        self.prepareDispatchEncoder = try PrepareDispatchEncoder(device: device, library: hardwareLib)
        self.initIndicesEncoder = try InitIndicesEncoder(device: device, library: hardwareLib)
        self.depthRadixSortEncoder = try DepthRadixSortEncoder(device: device, library: depthFirstLib, precision: depthSortKeyPrecision)
        self.reorderDataEncoder = try ReorderDataEncoder(device: device, library: hardwareLib)
        self.visibilityCompactionEncoder = try VisibilityCompactionEncoder(
            device: device,
            library: depthFirstLib,
            depthKeyPrecision: depthSortKeyPrecision
        )

        let depthStencilState: MTLDepthStencilState = {
            let d = MTLDepthStencilDescriptor()
            d.isDepthWriteEnabled = true
            d.depthCompareFunction = .always
            return device.makeDepthStencilState(descriptor: d)!
        }()
        let noDepthStencilState: MTLDepthStencilState = {
            let d = MTLDepthStencilDescriptor()
            d.isDepthWriteEnabled = false
            d.depthCompareFunction = .always
            return device.makeDepthStencilState(descriptor: d)!
        }()

        switch backend {
        case .meshShaders:
            let pipelines = try Self.createMeshPipelines(
                device: device,
                library: hardwareLib,
                colorFormat: config.colorFormat,
                backToFront: config.backToFront
            )
            let encoder = MeshRenderEncoder(
                initializePipeline: pipelines.initialize,
                centerSortPipeline: pipelines.centerSortDraw,
                postprocessPipeline: pipelines.postprocess,
                monoPipeline: pipelines.mono,
                depthStencilState: depthStencilState,
                noDepthStencilState: noDepthStencilState
            )
            self.backendImpl = .mesh(encoder)

        case .instanced:
            let pipelines = try Self.createInstancedPipelines(
                device: device,
                library: hardwareLib,
                colorFormat: config.colorFormat,
                backToFront: config.backToFront,
                depthStencilState: depthStencilState,
                noDepthStencilState: noDepthStencilState
            )
            let quadIndexBuffer = Self.makeQuadIndexBuffer(device: device)
            let encoder = InstancedRenderEncoder(
                initializePipeline: pipelines.initialize,
                centerSortPipeline: pipelines.centerSortDraw,
                postprocessPipeline: pipelines.postprocess,
                monoPipeline: pipelines.mono,
                depthStencilState: pipelines.depthStencilState,
                noDepthStencilState: pipelines.noDepthStencilState,
                quadIndexBuffer: quadIndexBuffer
            )
            self.backendImpl = .instanced(encoder)
        }

        self.resourceManager = HardwareResourceManager(device: device)
    }

    // MARK: - GaussianRenderer

    public func render(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int
    ) {
        guard input.gaussianCount > 0 else { return }

        do {
            let resources = try resourceManager.ensureMonoResources(gaussianCount: input.gaussianCount)
            renderMonoInternal(
                commandBuffer: commandBuffer,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                input: input,
                camera: camera,
                resources: resources,
                width: width,
                height: height
            )
        } catch {
            print("Failed to allocate mono resources: \(error)")
        }
    }

    public func renderStereo(
        commandBuffer: MTLCommandBuffer,
        target: StereoRenderTarget,
        input: GaussianInput,
        camera: StereoCameraParams,
        width: Int,
        height: Int
    ) {
        switch target {
        case .sideBySide(let colorTexture, let depthTexture):
            renderSideBySideCenterSort(
                commandBuffer: commandBuffer,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                input: input,
                configuration: .init(
                    from: camera,
                    width: width,
                    height: height,
                    rightViewOrigin: .init(Float(width), 0)
                ),
                eyeWidth: width,
                eyeHeight: height
            )

        case .foveated(let drawable, let configuration):
            renderFoveatedStereoCenterSort(
                commandBuffer: commandBuffer,
                drawable: drawable,
                input: input,
                configuration: configuration,
                width: width,
                height: height
            )
        }
    }

    // MARK: - Mono

    private func renderMonoInternal(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        input: GaussianInput,
        camera: CameraParams,
        resources: HardwareMonoResources,
        width: Int,
        height: Int
    ) {
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.label = "ResetMonoCounter"
            blitEncoder.fill(buffer: resources.counter, range: 0..<4, value: 0)
            blitEncoder.endEncoding()
        }

        monoProjectionEncoder.encode(
            commandBuffer: commandBuffer,
            input: input,
            projected: resources.projected,
            preDepthKeys: resources.scratchKeys,
            visibilityMarks: resources.visibilityMarks,
            camera: camera,
            width: width,
            height: height,
            inputIsSRGB: inputIsSRGB,
            precision: precision
        )

        visibilityCompactionEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            nTouchedTiles: resources.visibilityMarks,
            preDepthKeys: resources.scratchKeys,
            prefixSumsScratch: resources.compactionScratch,
            prefixOffsetsOut: resources.scratchIndices,
            depthKeysOut: resources.depthKeys,
            primitiveIndicesOut: resources.sortedIndices,
            visibleCountOut: resources.counter,
            maxOutCount: resources.paddedMax,
            maxCountCapacity: resources.paddedMax
        )

        let sortConfig = PrepareDispatchConfig(numBlocks: resources.numBlocks)
        prepareDispatchEncoder.encodeMono(
            commandBuffer: commandBuffer,
            counter: resources.counter,
            header: resources.header,
            sortHeader: resources.sortHeader,
            sortDispatch: resources.sortDispatch,
            scanDispatch: resources.scanDispatch,
            reorderDispatch: resources.reorderDispatch,
            drawArgs: resources.drawArgs,
            config: sortConfig
        )

        let sortBuffers = RadixSortBuffers(
            histogram: resources.histogram,
            blockSums: resources.blockSums,
            scannedHistogram: resources.scannedHistogram,
            scratchKeys: resources.scratchKeys,
            scratchIndices: resources.scratchIndices
        )
        depthRadixSortEncoder.encode(
            commandBuffer: commandBuffer,
            depthKeys: resources.depthKeys,
            sortedIndices: resources.sortedIndices,
            sortBuffers: sortBuffers,
            sortHeader: resources.sortHeader,
            sortDispatch: resources.sortDispatch,
            scanDispatch: resources.scanDispatch,
            label: "Mono"
        )

        reorderDataEncoder.encodeMono(
            commandBuffer: commandBuffer,
            projected: resources.projected,
            sortedIndices: resources.sortedIndices,
            projectedSorted: resources.projectedSorted,
            header: resources.header,
            reorderDispatch: resources.reorderDispatch,
            backToFront: backToFront
        )

        switch backendImpl {
        case .instanced(let encoder):
            encoder.encodeMono(
                commandBuffer: commandBuffer,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                projectedSorted: resources.projectedSorted,
                header: resources.header,
                drawArgs: resources.drawArgs,
                width: width,
                height: height,
                farPlane: camera.far
            )

        case .mesh(let encoder):
            prepareDispatchEncoder.encodeMeshDrawArgs(
                commandBuffer: commandBuffer,
                header: resources.header,
                meshDrawArgs: resources.meshDrawArgs
            )
            encoder.encodeMono(
                commandBuffer: commandBuffer,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                projectedSorted: resources.projectedSorted,
                header: resources.header,
                meshDrawArgs: resources.meshDrawArgs,
                width: width,
                height: height,
                farPlane: camera.far
            )
        }
    }

    // MARK: - Stereo (Center Sort)

    private func renderSideBySideCenterSort(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        input: GaussianInput,
        configuration: StereoConfiguration,
        eyeWidth: Int,
        eyeHeight: Int
    ) {
        guard input.gaussianCount > 0 else { return }

        do {
            let resources = try resourceManager.ensureCenterSortResources(gaussianCount: input.gaussianCount)
            renderCenterSortInternal(
                commandBuffer: commandBuffer,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                rasterizationRateMap: nil,
                input: input,
                configuration: configuration,
                resources: resources,
                width: eyeWidth,
                height: eyeHeight
            )
        } catch {
            print("Failed to allocate center-sort resources: \(error)")
        }
    }

    private func renderFoveatedStereoCenterSort(
        commandBuffer: MTLCommandBuffer,
        drawable: FoveatedStereoDrawable,
        input: GaussianInput,
        configuration: StereoConfiguration,
        width: Int,
        height: Int
    ) {
        guard input.gaussianCount > 0 else { return }

        do {
            let resources = try resourceManager.ensureCenterSortResources(gaussianCount: input.gaussianCount)
            renderCenterSortInternal(
                commandBuffer: commandBuffer,
                colorTexture: drawable.colorTexture,
                depthTexture: drawable.depthTexture,
                rasterizationRateMap: drawable.rasterizationRateMap,
                input: input,
                configuration: configuration,
                resources: resources,
                width: width,
                height: height
            )
        } catch {
            print("Failed to allocate center-sort resources: \(error)")
        }
    }

    private func renderCenterSortInternal(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        rasterizationRateMap: MTLRasterizationRateMap?,
        input: GaussianInput,
        configuration: StereoConfiguration,
        resources: HardwareCenterSortResources,
        width: Int,
        height: Int
    ) {
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.label = "ResetCenterSortCounter"
            blitEncoder.fill(buffer: resources.visibleCount, range: 0..<4, value: 0)
            blitEncoder.endEncoding()
        }

        centerProjectionEncoder.encode(
            commandBuffer: commandBuffer,
            input: input,
            projected: resources.projected,
            preDepthKeys: resources.scratchKeys,
            visibilityMarks: resources.visibilityMarks,
            configuration: configuration,
            eyeWidth: width,
            eyeHeight: height,
            inputIsSRGB: inputIsSRGB,
            precision: precision
        )

        visibilityCompactionEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            nTouchedTiles: resources.visibilityMarks,
            preDepthKeys: resources.scratchKeys,
            prefixSumsScratch: resources.compactionScratch,
            prefixOffsetsOut: resources.scratchIndices,
            depthKeysOut: resources.depthKeys,
            primitiveIndicesOut: resources.sortedIndices,
            visibleCountOut: resources.visibleCount,
            maxOutCount: resources.paddedMax,
            maxCountCapacity: resources.paddedMax
        )

        let centerConfig = PrepareDispatchConfig(numBlocks: resources.numBlocks)
        prepareDispatchEncoder.encodeCenterSort(
            commandBuffer: commandBuffer,
            visibleCount: resources.visibleCount,
            header: resources.header,
            sortHeader: resources.sortHeader,
            sortDispatch: resources.sortDispatch,
            scanDispatch: resources.scanDispatch,
            reorderDispatch: resources.reorderDispatch,
            config: centerConfig
        )

        let sortBuffers = RadixSortBuffers(
            histogram: resources.histogram,
            blockSums: resources.blockSums,
            scannedHistogram: resources.scannedHistogram,
            scratchKeys: resources.scratchKeys,
            scratchIndices: resources.scratchIndices
        )
        depthRadixSortEncoder.encode(
            commandBuffer: commandBuffer,
            depthKeys: resources.depthKeys,
            sortedIndices: resources.sortedIndices,
            sortBuffers: sortBuffers,
            sortHeader: resources.sortHeader,
            sortDispatch: resources.sortDispatch,
            scanDispatch: resources.scanDispatch,
            label: "CenterSort"
        )

        reorderDataEncoder.encodeCenterSort(
            commandBuffer: commandBuffer,
            projected: resources.projected,
            sortedIndices: resources.sortedIndices,
            projectedSorted: resources.projectedSorted,
            header: resources.header,
            reorderDispatch: resources.reorderDispatch,
            backToFront: backToFront
        )

        switch backendImpl {
        case .instanced(let encoder):
            prepareDispatchEncoder.encodeCenterSortDrawArgs(
                commandBuffer: commandBuffer,
                header: resources.header,
                drawArgs: resources.drawArgs
            )
            encoder.encodeStereo(
                commandBuffer: commandBuffer,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                rasterizationRateMap: rasterizationRateMap,
                projectedSorted: resources.projectedSorted,
                header: resources.header,
                drawArgs: resources.drawArgs,
                configuration: configuration,
                width: width,
                height: height
            )

        case .mesh(let encoder):
            prepareDispatchEncoder.encodeMeshDrawArgs(
                commandBuffer: commandBuffer,
                header: resources.header,
                meshDrawArgs: resources.meshDrawArgs
            )
            encoder.encodeStereo(
                commandBuffer: commandBuffer,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                rasterizationRateMap: rasterizationRateMap,
                projectedSorted: resources.projectedSorted,
                header: resources.header,
                meshDrawArgs: resources.meshDrawArgs,
                configuration: configuration,
                width: width,
                height: height
            )
        }
    }
}

// MARK: - Pipelines

private extension HardwareRenderer {
    struct InstancedPipelines {
        let initialize: MTLRenderPipelineState
        let centerSortDraw: MTLRenderPipelineState
        let postprocess: MTLRenderPipelineState
        let mono: MTLRenderPipelineState
        let depthStencilState: MTLDepthStencilState
        let noDepthStencilState: MTLDepthStencilState
    }

    static func createInstancedPipelines(
        device: MTLDevice,
        library: MTLLibrary,
        colorFormat: MTLPixelFormat,
        backToFront: Bool,
        depthStencilState: MTLDepthStencilState,
        noDepthStencilState: MTLDepthStencilState
    ) throws -> InstancedPipelines {
        guard let centerVertexFn = library.makeFunction(name: "centerSortInstancedVertex"),
              let initFn = library.makeFunction(name: "initializeFragmentStore"),
              let postVertexFn = library.makeFunction(name: "postprocessVertexShader"),
              let postFragmentFn = library.makeFunction(name: "postprocessFragmentShader"),
              let centerFragmentFn = library.makeFunction(name: backToFront ? "centerSortInstancedFragmentBackToFront" : "centerSortInstancedFragment") else {
            throw RendererError.failedToCreatePipeline("Render shader functions not found")
        }

        let tileDesc = MTLTileRenderPipelineDescriptor()
        tileDesc.tileFunction = initFn
        tileDesc.threadgroupSizeMatchesTileSize = true
        tileDesc.colorAttachments[0].pixelFormat = colorFormat
        let initializePipeline = try device.makeRenderPipelineState(tileDescriptor: tileDesc, options: [], reflection: nil)

        let drawDesc = MTLRenderPipelineDescriptor()
        drawDesc.vertexFunction = centerVertexFn
        drawDesc.fragmentFunction = centerFragmentFn
        drawDesc.colorAttachments[0].pixelFormat = colorFormat
        drawDesc.depthAttachmentPixelFormat = .depth32Float
        drawDesc.inputPrimitiveTopology = .triangle
        drawDesc.maxVertexAmplificationCount = 2
        let centerSortDrawPipeline = try device.makeRenderPipelineState(descriptor: drawDesc)

        let postDesc = MTLRenderPipelineDescriptor()
        postDesc.vertexFunction = postVertexFn
        postDesc.fragmentFunction = postFragmentFn
        postDesc.colorAttachments[0].pixelFormat = colorFormat
        postDesc.depthAttachmentPixelFormat = .depth32Float
        postDesc.maxVertexAmplificationCount = 2
        let postprocessPipeline = try device.makeRenderPipelineState(descriptor: postDesc)

        guard let monoVertexFn = library.makeFunction(name: "monoGaussianVertex"),
              let monoFragmentFn = library.makeFunction(name: "monoGaussianFragment") else {
            throw RendererError.failedToCreatePipeline("Mono render functions not found")
        }

        let monoRenderDesc = MTLRenderPipelineDescriptor()
        monoRenderDesc.vertexFunction = monoVertexFn
        monoRenderDesc.fragmentFunction = monoFragmentFn
        monoRenderDesc.colorAttachments[0].pixelFormat = colorFormat
        monoRenderDesc.colorAttachments[0].isBlendingEnabled = true
        if backToFront {
            monoRenderDesc.colorAttachments[0].sourceRGBBlendFactor = .one
            monoRenderDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            monoRenderDesc.colorAttachments[0].sourceAlphaBlendFactor = .one
            monoRenderDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        } else {
            monoRenderDesc.colorAttachments[0].sourceRGBBlendFactor = .oneMinusDestinationAlpha
            monoRenderDesc.colorAttachments[0].destinationRGBBlendFactor = .one
            monoRenderDesc.colorAttachments[0].sourceAlphaBlendFactor = .oneMinusDestinationAlpha
            monoRenderDesc.colorAttachments[0].destinationAlphaBlendFactor = .one
        }
        monoRenderDesc.depthAttachmentPixelFormat = .depth32Float
        monoRenderDesc.inputPrimitiveTopology = .triangle

        let monoPipeline = try device.makeRenderPipelineState(descriptor: monoRenderDesc)

        return InstancedPipelines(
            initialize: initializePipeline,
            centerSortDraw: centerSortDrawPipeline,
            postprocess: postprocessPipeline,
            mono: monoPipeline,
            depthStencilState: depthStencilState,
            noDepthStencilState: noDepthStencilState
        )
    }

    struct MeshPipelines {
        let initialize: MTLRenderPipelineState
        let centerSortDraw: MTLRenderPipelineState
        let postprocess: MTLRenderPipelineState
        let mono: MTLRenderPipelineState
    }

    static func createMeshPipelines(
        device: MTLDevice,
        library: MTLLibrary,
        colorFormat: MTLPixelFormat,
        backToFront: Bool
    ) throws -> MeshPipelines {
        let gaussiansPerObjectTG = 64
        let gaussiansPerMeshTG = 16
        let meshThreads = 64
        let centerGaussiansPerObjectTG = 64
        let centerGaussiansPerMeshTG = 16

        guard let centerObjectFn = library.makeFunction(name: "centerSortObjectShader"),
              let centerMeshFn = library.makeFunction(name: "centerSortMeshShader") else {
            throw RendererError.failedToCreatePipeline("Center-sort mesh shader functions not found")
        }

        guard let initFn = library.makeFunction(name: "initializeFragmentStore"),
              let postVertexFn = library.makeFunction(name: "postprocessVertexShader"),
              let postFragmentFn = library.makeFunction(name: "postprocessFragmentShader"),
              let centerFragmentFn = library.makeFunction(name: backToFront ? "gaussianFragmentShaderBackToFront" : "gaussianFragmentShader") else {
            throw RendererError.failedToCreatePipeline("Imageblock shader functions not found")
        }

        let tileDesc = MTLTileRenderPipelineDescriptor()
        tileDesc.tileFunction = initFn
        tileDesc.threadgroupSizeMatchesTileSize = true
        tileDesc.colorAttachments[0].pixelFormat = colorFormat
        let initializePipeline = try device.makeRenderPipelineState(tileDescriptor: tileDesc, options: [], reflection: nil)

        let centerDrawDesc = MTLMeshRenderPipelineDescriptor()
        centerDrawDesc.objectFunction = centerObjectFn
        centerDrawDesc.meshFunction = centerMeshFn
        centerDrawDesc.fragmentFunction = centerFragmentFn
        centerDrawDesc.colorAttachments[0].pixelFormat = colorFormat
        centerDrawDesc.depthAttachmentPixelFormat = .depth32Float
        centerDrawDesc.payloadMemoryLength = MemoryLayout<CenterSortGaussianData>.stride * centerGaussiansPerObjectTG + 16
        centerDrawDesc.maxTotalThreadsPerObjectThreadgroup = centerGaussiansPerObjectTG
        centerDrawDesc.maxTotalThreadsPerMeshThreadgroup = meshThreads
        centerDrawDesc.maxTotalThreadgroupsPerMeshGrid = (centerGaussiansPerObjectTG + centerGaussiansPerMeshTG - 1) / centerGaussiansPerMeshTG
        centerDrawDesc.maxVertexAmplificationCount = 2
        let centerSortDrawPipeline = try device.makeRenderPipelineState(descriptor: centerDrawDesc, options: []).0

        let postDesc = MTLRenderPipelineDescriptor()
        postDesc.vertexFunction = postVertexFn
        postDesc.fragmentFunction = postFragmentFn
        postDesc.colorAttachments[0].pixelFormat = colorFormat
        postDesc.depthAttachmentPixelFormat = .depth32Float
        postDesc.maxVertexAmplificationCount = 2
        let postprocessPipeline = try device.makeRenderPipelineState(descriptor: postDesc)

        guard let monoObjectFn = library.makeFunction(name: "monoObjectShader"),
              let monoMeshFn = library.makeFunction(name: "monoMeshShader"),
              let monoFragmentFn = library.makeFunction(name: "monoFragmentShader") else {
            throw RendererError.failedToCreatePipeline("Mono mesh shader functions not found")
        }

        let monoMeshDesc = MTLMeshRenderPipelineDescriptor()
        monoMeshDesc.objectFunction = monoObjectFn
        monoMeshDesc.meshFunction = monoMeshFn
        monoMeshDesc.fragmentFunction = monoFragmentFn
        monoMeshDesc.colorAttachments[0].pixelFormat = colorFormat
        monoMeshDesc.colorAttachments[0].isBlendingEnabled = true
        monoMeshDesc.colorAttachments[0].alphaBlendOperation = .add
        if backToFront {
            monoMeshDesc.colorAttachments[0].sourceRGBBlendFactor = .one
            monoMeshDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            monoMeshDesc.colorAttachments[0].sourceAlphaBlendFactor = .one
            monoMeshDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        } else {
            monoMeshDesc.colorAttachments[0].sourceRGBBlendFactor = .oneMinusDestinationAlpha
            monoMeshDesc.colorAttachments[0].destinationRGBBlendFactor = .one
            monoMeshDesc.colorAttachments[0].sourceAlphaBlendFactor = .oneMinusDestinationAlpha
            monoMeshDesc.colorAttachments[0].destinationAlphaBlendFactor = .one
        }
        monoMeshDesc.depthAttachmentPixelFormat = .depth32Float
        monoMeshDesc.payloadMemoryLength = MemoryLayout<InstancedGaussianData>.stride * gaussiansPerObjectTG + 16
        monoMeshDesc.maxTotalThreadsPerObjectThreadgroup = gaussiansPerObjectTG
        monoMeshDesc.maxTotalThreadsPerMeshThreadgroup = meshThreads
        monoMeshDesc.maxTotalThreadgroupsPerMeshGrid = (gaussiansPerObjectTG + gaussiansPerMeshTG - 1) / gaussiansPerMeshTG

        let monoPipeline = try device.makeRenderPipelineState(descriptor: monoMeshDesc, options: []).0

        return MeshPipelines(
            initialize: initializePipeline,
            centerSortDraw: centerSortDrawPipeline,
            postprocess: postprocessPipeline,
            mono: monoPipeline
        )
    }

    static func makeQuadIndexBuffer(device: MTLDevice) -> MTLBuffer {
        let maxIndexedSplatCount = 1024
        let indexCount = maxIndexedSplatCount * 6
        let bufferLength = indexCount * MemoryLayout<UInt32>.stride

        guard let buffer = device.makeBuffer(length: bufferLength, options: .storageModeShared) else {
            fatalError("Failed to allocate quad index buffer")
        }
        buffer.label = "InstancedGaussian.QuadIndexBuffer"

        let indices = buffer.contents().bindMemory(to: UInt32.self, capacity: indexCount)
        for i in 0..<maxIndexedSplatCount {
            let baseV = UInt32(i * 4)
            let baseI = i * 6
            indices[baseI + 0] = baseV + 0
            indices[baseI + 1] = baseV + 1
            indices[baseI + 2] = baseV + 2
            indices[baseI + 3] = baseV + 1
            indices[baseI + 4] = baseV + 2
            indices[baseI + 5] = baseV + 3
        }
        return buffer
    }
}
