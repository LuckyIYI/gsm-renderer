import Metal

public extension MTLBuffer {
    func copy(
        to other: MTLBuffer,
        offset: Int = 0
    ) {
        memcpy(
            other.contents() + offset,
            contents(),
            length
        )
    }

    func pointer<T>(of type: T.Type) -> UnsafeMutablePointer<T>? {
        #if DEBUG
        guard length >= MemoryLayout<T>.stride
        else { fatalError("Buffer length check failed") }
        #endif

        let bindedPointer = contents()
            .assumingMemoryBound(to: type)
        return bindedPointer
    }

    func bufferPointer<T>(
        of type: T.Type,
        count: Int
    ) -> UnsafeBufferPointer<T>? {
        guard let startPointer = pointer(of: type)
        else { return nil }
        let bufferPointer = UnsafeBufferPointer(
            start: startPointer,
            count: count
        )
        return bufferPointer
    }

    func array<T>(
        of type: T.Type,
        count: Int
    ) -> [T]? {
        guard let bufferPointer = bufferPointer(
            of: type,
            count: count
        )
        else { return nil }
        let valueArray = Array(bufferPointer)
        return valueArray
    }

    /// Put a value in `MTLBuffer` at desired offset.
    /// - Parameters:
    ///   - value: value to put in the buffer.
    ///   - offset: offset in bytes.
    func put<T>(
        _ value: T,
        at offset: Int = 0
    ) throws {
        guard length - offset >= MemoryLayout<T>.stride
        else { throw MetalError.MTLBufferError.incompatibleData }
        (contents() + offset).assumingMemoryBound(to: T.self)
            .pointee = value
    }

    /// Put values in `MTLBuffer` at desired offset.
    /// - Parameters:
    ///   - values: values to put in the buffer.
    ///   - offset: offset in bytes.
    func put<T>(
        _ values: [T],
        at offset: Int = 0
    ) throws {
        let dataLength = MemoryLayout<T>.stride * values.count
        guard length - offset >= dataLength
        else { throw MetalError.MTLBufferError.incompatibleData }

        _ = try values.withUnsafeBytes {
            if let p = $0.baseAddress {
                memcpy(
                    contents() + offset,
                    p,
                    dataLength
                )
            } else {
                throw MetalError.MTLBufferError.incompatibleData
            }
        }
    }
}

public enum MetalError {
    public enum MTLContextError: Error {
        case textureCacheCreationFailed
    }

    public enum MTLDeviceError: Error {
        case argumentEncoderCreationFailed
        case bufferCreationFailed
        case commandQueueCreationFailed
        case depthStencilStateCreationFailed
        case eventCreationFailed
        case fenceCreationFailed
        case heapCreationFailed
        case indirectCommandBufferCreationFailed
        case libraryCreationFailed
        case rasterizationRateMapCreationFailed
        case samplerStateCreationFailed
        case textureCreationFailed
        case textureViewCreationFailed
    }

    public enum MTLHeapError: Error {
        case bufferCreationFailed
        case textureCreationFailed
    }

    public enum MTLCommandQueueError: Error {
        case commandBufferCreationFailed
    }

    public enum MTLLibraryError: Error {
        case functionCreationFailed
    }

    public enum MTLTextureSerializationError: Error {
        case allocationFailed
        case dataAccessFailure
        case unsupportedPixelFormat
    }

    public enum MTLTextureError: Error {
        case imageCreationFailed
        case imageIncompatiblePixelFormat
    }

    public enum MTLResourceError: Error {
        case resourceUnavailable
    }

    public enum MTLBufferError: Error {
        case incompatibleData
    }

    public enum MTLPixelFormatError: Error {
        case incompatibleCVPixelFormat
    }
}
