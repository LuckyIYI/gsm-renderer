import Foundation
import Metal

enum GaussianRendererError: Error {
    case deviceUnavailable
}

@_cdecl("gaussian_renderer_version")
public func gaussian_renderer_version() -> Int32 {
    1
}

@_cdecl("gaussian_renderer_warmup")
public func gaussian_renderer_warmup() -> Int32 {
    do {
        _ = try makeDefaultDevice()
        return 0
    } catch {
        return -1
    }
}

private func makeDefaultDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw GaussianRendererError.deviceUnavailable
    }
    return device
}
