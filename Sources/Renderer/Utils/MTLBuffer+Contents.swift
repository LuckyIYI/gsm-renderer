import Metal

// MARK: - Buffer Allocation Error

struct BufferAllocationError: Error, LocalizedError {
    let label: String
    var errorDescription: String? { "Failed to allocate buffer: \(label)" }
}

// MARK: - MTLDevice Buffer Creation

extension MTLDevice {
    /// Create a buffer with typed count and automatic size calculation.
    func makeBuffer<T>(
        count: Int,
        type _: T.Type,
        options: MTLResourceOptions = .storageModePrivate,
        label: String
    ) throws -> MTLBuffer {
        let length = max(1, count) * MemoryLayout<T>.stride
        guard let buffer = makeBuffer(length: length, options: options) else {
            throw BufferAllocationError(label: label)
        }
        buffer.label = label
        return buffer
    }
}

// MARK: - MTLBuffer Utilities

extension MTLBuffer {
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
        guard length >= MemoryLayout<T>.stride else {
            return nil
        }

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

    func put<T>(
        _ value: T,
        at offset: Int = 0
    ) throws {
        guard length - offset >= MemoryLayout<T>.stride
        else { throw MetalError.MTLBufferError.incompatibleData }
        (contents() + offset).assumingMemoryBound(to: T.self)
            .pointee = value
    }

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

enum MetalError {
    enum MTLBufferError: Error {
        case incompatibleData
    }
}
