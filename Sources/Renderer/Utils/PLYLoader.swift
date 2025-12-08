import Foundation
import simd

// MARK: - PLYHeader

public struct PLYHeader: Equatable {
    enum Keyword: String {
        case ply
        case format
        case comment
        case element
        case property
        case endHeader = "end_header"
        case obj_info
    }

    public enum Format: String, Equatable, Sendable {
        case ascii
        case binaryLittleEndian = "binary_little_endian"
        case binaryBigEndian = "binary_big_endian"
    }

    public struct Element: Equatable {
        public var name: String
        public var count: UInt32
        public var properties: [Property]

        public init(name: String, count: UInt32, properties: [Property]) {
            self.name = name
            self.count = count
            self.properties = properties
        }
    }

    public enum PropertyType: Equatable {
        case primitive(PrimitivePropertyType)
        case list(countType: PrimitivePropertyType, valueType: PrimitivePropertyType)

        var byteWidth: Int {
            switch self {
            case let .primitive(t): t.byteWidth
            case .list: 0
            }
        }
    }

    public enum PrimitivePropertyType: Equatable {
        case int8
        case uint8
        case int16
        case uint16
        case int32
        case uint32
        case float32
        case float64

        var byteWidth: Int {
            switch self {
            case .int8, .uint8: 1
            case .int16, .uint16: 2
            case .int32, .uint32, .float32: 4
            case .float64: 8
            }
        }
    }

    public struct Property: Equatable {
        public var name: String
        public var type: PropertyType

        public init(name: String, type: PropertyType) {
            self.name = name
            self.type = type
        }
    }

    public var format: Format
    public var version: String
    public var elements: [Element]

    public init(format: Format, version: String, elements: [Element]) {
        self.format = format
        self.version = version
        self.elements = elements
    }
}

// MARK: - PLYHeader Parsing

extension PLYHeader {
    public enum DecodeError: LocalizedError {
        case headerFormatMissing
        case headerInvalidCharacters
        case headerUnknownKeyword(String)
        case headerUnexpectedKeyword(String)
        case headerInvalidLine(String)
        case headerInvalidFileFormatType(String)
        case headerUnknownPropertyType(String)
        case headerInvalidListCountType(String)

        public var errorDescription: String? {
            switch self {
            case .headerFormatMissing: "Header format missing"
            case .headerInvalidCharacters: "Invalid characters in header"
            case let .headerUnknownKeyword(keyword): "Unknown keyword: \"\(keyword)\""
            case let .headerUnexpectedKeyword(keyword): "Unexpected keyword: \"\(keyword)\""
            case let .headerInvalidLine(line): "Invalid line: \"\(line)\""
            case let .headerInvalidFileFormatType(type): "Invalid format type: \(type)"
            case let .headerUnknownPropertyType(type): "Unknown property type: \(type)"
            case let .headerInvalidListCountType(type): "Invalid list count type: \(type)"
            }
        }
    }

    static func decodeASCII(from headerData: Data) throws -> PLYHeader {
        guard let headerString = String(data: headerData, encoding: .utf8) else {
            throw DecodeError.headerInvalidCharacters
        }
        var parseError: Swift.Error?
        var header: PLYHeader?
        headerString.enumerateLines { (headerLine, stop: inout Bool) in
            do {
                guard let keywordString = headerLine.components(separatedBy: .whitespaces).filter({ !$0.isEmpty }).first else {
                    return
                }
                guard let keyword = PLYHeader.Keyword(rawValue: keywordString) else {
                    throw DecodeError.headerUnknownKeyword(keywordString)
                }
                switch keyword {
                case .ply, .comment, .obj_info:
                    return
                case .format:
                    guard header == nil else {
                        throw DecodeError.headerUnexpectedKeyword(keyword.rawValue)
                    }
                    let regex = #/\s*format\s+(?<format>\w+?)\s+(?<version>\S+?)/#
                    guard let match = try regex.wholeMatch(in: headerLine) else {
                        throw DecodeError.headerInvalidLine(headerLine)
                    }
                    guard let format = PLYHeader.Format(rawValue: String(match.format)) else {
                        throw DecodeError.headerInvalidFileFormatType(String(match.format))
                    }
                    header = PLYHeader(format: format, version: String(match.version), elements: [])
                case .element:
                    guard header != nil else {
                        throw DecodeError.headerUnexpectedKeyword(keyword.rawValue)
                    }
                    let regex = #/\s*element\s+(?<name>\S+?)\s+(?<count>\d+?)/#
                    guard let match = try regex.wholeMatch(in: headerLine) else {
                        throw DecodeError.headerInvalidLine(headerLine)
                    }
                    header?.elements.append(PLYHeader.Element(
                        name: String(match.name),
                        count: UInt32(match.count)!,
                        properties: []
                    ))
                case .property:
                    guard header != nil, header?.elements.isEmpty == false else {
                        throw DecodeError.headerUnexpectedKeyword(keyword.rawValue)
                    }
                    let listRegex = #/\s*property\s+list\s+(?<countType>\w+?)\s+(?<valueType>\w+?)\s+(?<name>\S+)/#
                    let nonListRegex = #/\s*property\s+(?<valueType>\w+?)\s+(?<name>\S+)/#
                    if let match = try listRegex.wholeMatch(in: headerLine) {
                        guard let countType = PLYHeader.PrimitivePropertyType.fromString(String(match.countType)) else {
                            throw DecodeError.headerUnknownPropertyType(String(match.countType))
                        }
                        guard let valueType = PLYHeader.PrimitivePropertyType.fromString(String(match.valueType)) else {
                            throw DecodeError.headerUnknownPropertyType(String(match.valueType))
                        }
                        let property = PLYHeader.Property(
                            name: String(match.name),
                            type: .list(countType: countType, valueType: valueType)
                        )
                        header!.elements[header!.elements.count - 1].properties.append(property)
                    } else if let match = try nonListRegex.wholeMatch(in: headerLine) {
                        guard let valueType = PLYHeader.PrimitivePropertyType.fromString(String(match.valueType)) else {
                            throw DecodeError.headerUnknownPropertyType(String(match.valueType))
                        }
                        let property = PLYHeader.Property(
                            name: String(match.name),
                            type: .primitive(valueType)
                        )
                        header!.elements[header!.elements.count - 1].properties.append(property)
                    } else {
                        throw DecodeError.headerInvalidLine(headerLine)
                    }
                case .endHeader:
                    stop = true
                }
            } catch {
                parseError = error
                stop = true
            }
        }

        if let parseError { throw parseError }
        guard let header else { throw DecodeError.headerFormatMissing }
        return header
    }
}

extension PLYHeader.PrimitivePropertyType {
    static func fromString(_ string: String) -> PLYHeader.PrimitivePropertyType? {
        switch string {
        case "int8", "char": .int8
        case "uint8", "uchar": .uint8
        case "int16", "short": .int16
        case "uint16", "ushort": .uint16
        case "int32", "int": .int32
        case "uint32", "uint": .uint32
        case "float32", "float": .float32
        case "float64", "double": .float64
        default: nil
        }
    }
}

// MARK: - PLYLoader Errors

public enum PLYLoaderError: Error, LocalizedError {
    case invalidHeader
    case unsupportedFormat(PLYHeader.Format)
    case missingVertexElement
    case missingRequiredProperties([String])
    case listPropertiesNotSupported
    case insufficientData
    case missingChunkElement

    public var errorDescription: String? {
        switch self {
        case .invalidHeader:
            "Invalid PLY header"
        case let .unsupportedFormat(format):
            "Unsupported PLY format: \(format.rawValue). Only binary_little_endian is supported."
        case .missingVertexElement:
            "No 'vertex' element found in PLY"
        case let .missingRequiredProperties(props):
            "Missing required properties: \(props.joined(separator: ", "))"
        case .listPropertiesNotSupported:
            "List properties in vertex element are not supported"
        case .insufficientData:
            "PLY file has insufficient data for declared vertex count"
        case .missingChunkElement:
            "Compressed PLY requires 'chunk' element with bounding box data"
        }
    }
}

// MARK: - PLYLoader

public enum PLYLoader {
    /// Load a Gaussian PLY file using memory-mapped binary decoding.
    /// Supports both standard and compressed (splat-transform) PLY formats.
    public static func load(url: URL) throws -> GaussianDataset {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])

        // Find end_header and parse header
        let endLF = Data("end_header\n".utf8)
        let endCRLF = Data("end_header\r\n".utf8)
        guard let headerEnd = data.range(of: endLF) ?? data.range(of: endCRLF) else {
            throw PLYLoaderError.invalidHeader
        }

        let headerData = data.subdata(in: data.startIndex ..< headerEnd.upperBound)
        let header = try PLYHeader.decodeASCII(from: headerData)

        guard header.format == .binaryLittleEndian else {
            throw PLYLoaderError.unsupportedFormat(header.format)
        }

        guard let vertex = header.elements.first(where: { $0.name == "vertex" }) else {
            throw PLYLoaderError.missingVertexElement
        }

        // Detect compressed format (splat-transform)
        let isCompressed = header.elements.contains(where: { $0.name == "chunk" }) &&
            vertex.properties.contains(where: { $0.name == "packed_position" }) &&
            vertex.properties.contains(where: { $0.name == "packed_rotation" }) &&
            vertex.properties.contains(where: { $0.name == "packed_scale" }) &&
            vertex.properties.contains(where: { $0.name == "packed_color" })

        if isCompressed {
            return try loadCompressed(data: data, header: header, bodyStart: headerEnd.upperBound)
        } else {
            return try loadStandard(data: data, header: header, vertex: vertex, bodyStart: headerEnd.upperBound)
        }
    }

    // MARK: - Compressed Format (splat-transform)

    private static func loadCompressed(data: Data, header: PLYHeader, bodyStart: Int) throws -> GaussianDataset {
        guard let chunk = header.elements.first(where: { $0.name == "chunk" }),
              let vertex = header.elements.first(where: { $0.name == "vertex" })
        else {
            throw PLYLoaderError.missingChunkElement
        }

        let chunkCount = Int(chunk.count)
        let vertexCount = Int(vertex.count)

        // Calculate element offsets and strides
        var chunkStride = 0
        var chunkOffsets: [String: Int] = [:]
        for prop in chunk.properties {
            chunkOffsets[prop.name] = chunkStride
            chunkStride += prop.type.byteWidth
        }

        var vertexStride = 0
        var vertexOffsets: [String: Int] = [:]
        for prop in vertex.properties {
            vertexOffsets[prop.name] = vertexStride
            vertexStride += prop.type.byteWidth
        }

        // SH element (optional)
        let shElement = header.elements.first(where: { $0.name == "sh" })
        var shStride = 0
        if let sh = shElement {
            for prop in sh.properties {
                shStride += prop.type.byteWidth
            }
        }

        // Calculate data offsets
        let chunkDataStart = bodyStart
        let vertexDataStart = chunkDataStart + chunkStride * chunkCount
        let shDataStart = vertexDataStart + vertexStride * vertexCount

        guard data.count >= shDataStart + shStride * vertexCount else {
            throw PLYLoaderError.insufficientData
        }

        var records: [GaussianRecord] = []
        records.reserveCapacity(vertexCount)

        data.withUnsafeBytes { rawBuffer in
            guard let base = rawBuffer.baseAddress else { return }

            // Helper to read chunk float property
            func getChunkFloat(_ chunkIdx: Int, _ name: String) -> Float {
                guard let offset = chunkOffsets[name] else { return 0 }
                let ptr = base + chunkDataStart + chunkIdx * chunkStride + offset
                return ptr.loadUnaligned(as: Float.self)
            }

            // Helper to read vertex uint32 property
            func getVertexUInt32(_ vertexIdx: Int, _ name: String) -> UInt32 {
                guard let offset = vertexOffsets[name] else { return 0 }
                let ptr = base + vertexDataStart + vertexIdx * vertexStride + offset
                return ptr.loadUnaligned(as: UInt32.self)
            }

            // Unpack helpers
            func unpackUnorm(_ value: UInt32, _ bits: Int) -> Float {
                let mask = UInt32((1 << bits) - 1)
                return Float(value & mask) / Float(mask)
            }

            func unpack111011(_ value: UInt32) -> SIMD3<Float> {
                let x = unpackUnorm(value >> 21, 11)
                let y = unpackUnorm(value >> 11, 10)
                let z = unpackUnorm(value, 11)
                return SIMD3<Float>(x, y, z)
            }

            func unpack8888(_ value: UInt32) -> SIMD4<Float> {
                let x = unpackUnorm(value >> 24, 8)
                let y = unpackUnorm(value >> 16, 8)
                let z = unpackUnorm(value >> 8, 8)
                let w = unpackUnorm(value, 8)
                return SIMD4<Float>(x, y, z, w)
            }

            func unpackRotation(_ value: UInt32) -> simd_quatf {
                let norm: Float = 1.0 / (sqrt(2.0) * 0.5)
                // PlayCanvas unpackRot extracts: a from bits 20-29, b from bits 10-19, c from bits 0-9
                let a = (unpackUnorm(value >> 20, 10) - 0.5) * norm
                let b = (unpackUnorm(value >> 10, 10) - 0.5) * norm
                let c = (unpackUnorm(value, 10) - 0.5) * norm
                let m = sqrt(max(0, 1.0 - (a * a + b * b + c * c)))

                // PlayCanvas stores quaternion as (x,y,z,w) in rot_0..rot_3, but their matrix uses:
                //   x(rot_0)=qw, y(rot_1)=qx, z(rot_2)=qy, w(rot_3)=qz
                // So rot_0=qw, rot_1=qx, rot_2=qy, rot_3=qz
                // My simd_quatf(ix,iy,iz,r) = (qx,qy,qz,qw)
                // case 0: set(m,a,b,c) -> rot=(m,a,b,c) -> (qw,qx,qy,qz)=(m,a,b,c) -> simd(ix:a,iy:b,iz:c,r:m)
                // case 1: set(a,m,b,c) -> rot=(a,m,b,c) -> (qw,qx,qy,qz)=(a,m,b,c) -> simd(ix:m,iy:b,iz:c,r:a)
                // case 2: set(a,b,m,c) -> rot=(a,b,m,c) -> (qw,qx,qy,qz)=(a,b,m,c) -> simd(ix:b,iy:m,iz:c,r:a)
                // case 3: set(a,b,c,m) -> rot=(a,b,c,m) -> (qw,qx,qy,qz)=(a,b,c,m) -> simd(ix:b,iy:c,iz:m,r:a)
                switch value >> 30 {
                case 0: return simd_quatf(ix: a, iy: b, iz: c, r: m)
                case 1: return simd_quatf(ix: m, iy: b, iz: c, r: a)
                case 2: return simd_quatf(ix: b, iy: m, iz: c, r: a)
                case 3: return simd_quatf(ix: b, iy: c, iz: m, r: a)
                default: return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
                }
            }

            func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float {
                a * (1 - t) + b * t
            }

            // Process all vertices
            for v in 0 ..< vertexCount {
                let chunkIdx = v / 256

                // Get chunk bounds
                let minX = getChunkFloat(chunkIdx, "min_x")
                let minY = getChunkFloat(chunkIdx, "min_y")
                let minZ = getChunkFloat(chunkIdx, "min_z")
                let maxX = getChunkFloat(chunkIdx, "max_x")
                let maxY = getChunkFloat(chunkIdx, "max_y")
                let maxZ = getChunkFloat(chunkIdx, "max_z")

                let minScaleX = getChunkFloat(chunkIdx, "min_scale_x")
                let minScaleY = getChunkFloat(chunkIdx, "min_scale_y")
                let minScaleZ = getChunkFloat(chunkIdx, "min_scale_z")
                let maxScaleX = getChunkFloat(chunkIdx, "max_scale_x")
                let maxScaleY = getChunkFloat(chunkIdx, "max_scale_y")
                let maxScaleZ = getChunkFloat(chunkIdx, "max_scale_z")

                // Get color bounds for this chunk
                let minR = getChunkFloat(chunkIdx, "min_r")
                let minG = getChunkFloat(chunkIdx, "min_g")
                let minB = getChunkFloat(chunkIdx, "min_b")
                let maxR = getChunkFloat(chunkIdx, "max_r")
                let maxG = getChunkFloat(chunkIdx, "max_g")
                let maxB = getChunkFloat(chunkIdx, "max_b")

                // Get packed vertex data
                let packedPos = getVertexUInt32(v, "packed_position")
                let packedRot = getVertexUInt32(v, "packed_rotation")
                let packedScale = getVertexUInt32(v, "packed_scale")
                let packedColor = getVertexUInt32(v, "packed_color")

                // Unpack position
                let p = unpack111011(packedPos)
                let position = SIMD3<Float>(
                    lerp(minX, maxX, p.x),
                    lerp(minY, maxY, p.y),
                    lerp(minZ, maxZ, p.z)
                )

                // Unpack rotation
                let rotation = unpackRotation(packedRot)

                // Unpack scale (already in log space from the chunk bounds)
                let s = unpack111011(packedScale)
                let scale = SIMD3<Float>(
                    exp(lerp(minScaleX, maxScaleX, s.x)),
                    exp(lerp(minScaleY, maxScaleY, s.y)),
                    exp(lerp(minScaleZ, maxScaleZ, s.z))
                )

                // Unpack color -> opacity and DC color (interpolated using chunk bounds)
                let c = unpack8888(packedColor)
                let opacity = c.w // Already in [0,1] range

                // Interpolate RGB using chunk color bounds to get normalized color [0,1]
                // Then convert to SH DC coefficients using: (color - 0.5) / SH_C0
                let SH_C0: Float = 0.28209479177387814
                let colorR = lerp(minR, maxR, c.x)
                let colorG = lerp(minG, maxG, c.y)
                let colorB = lerp(minB, maxB, c.z)
                let sh0 = (colorR - 0.5) / SH_C0
                let sh1 = (colorG - 0.5) / SH_C0
                let sh2 = (colorB - 0.5) / SH_C0

                records.append(GaussianRecord(
                    position: position,
                    scale: scale,
                    rotation: rotation,
                    opacity: opacity,
                    sh: [sh0, sh1, sh2]
                ))
            }
        }

        // Build harmonics array in shader format
        // Layout per vertex: [R0, G0, B0] for DC only (shComponents = 1)
        // Standard format uses: [R0..R15, G0..G15, B0..B15] per vertex
        let shComponentsDC = 1 // DC only
        let shStrideDC = shComponentsDC * 3 // 3 for DC only
        var harmonics = [Float](repeating: 0, count: records.count * shStrideDC)

        for i in 0 ..< records.count {
            let base = i * shStrideDC
            harmonics[base + 0] = records[i].sh[0] // R channel (f_dc_0)
            harmonics[base + 1] = records[i].sh[1] // G channel (f_dc_1)
            harmonics[base + 2] = records[i].sh[2] // B channel (f_dc_2)
        }

        // Note: PlayCanvas compressed format does NOT apply the (-1,-1,1) transform
        // that standard PLY uses. The data is already in the correct coordinate system.

        // Recenter positions
        let bounds = GaussianSceneBuilder.bounds(of: records)
        let center = bounds.center
        if simd_length(center) > 1e-6 {
            for i in 0 ..< records.count {
                records[i].position -= center
            }
        }

        return GaussianDataset(
            records: records,
            harmonics: harmonics,
            shComponents: 1, // DC only (1 SH band = f_dc_0, f_dc_1, f_dc_2)
            cameraPoses: [],
            cameraNames: [],
            imageSize: nil
        )
    }

    // MARK: - Standard Format

    private static func loadStandard(data: Data, header _: PLYHeader, vertex: PLYHeader.Element, bodyStart: Int) throws -> GaussianDataset {
        // Ensure all properties are primitive (no lists)
        for prop in vertex.properties {
            if case .list = prop.type {
                throw PLYLoaderError.listPropertiesNotSupported
            }
        }

        let vertexCount = Int(vertex.count)

        // Calculate property offsets within each vertex
        var offsets: [Int] = []
        offsets.reserveCapacity(vertex.properties.count)
        var stride = 0
        for prop in vertex.properties {
            offsets.append(stride)
            stride += prop.type.byteWidth
        }

        guard data.count - bodyStart >= stride * vertexCount else {
            throw PLYLoaderError.insufficientData
        }

        // Map property indices
        var idx_x = -1, idx_y = -1, idx_z = -1
        var idx_s0 = -1, idx_s1 = -1, idx_s2 = -1
        var idx_r0 = -1, idx_r1 = -1, idx_r2 = -1, idx_r3 = -1
        var idx_op = -1
        var shMap: [(name: String, index: Int)] = []

        for (i, p) in vertex.properties.enumerated() {
            let name = p.name.lowercased()
            switch name {
            case "x", "px", "pos_x", "position_x": idx_x = i
            case "y", "py", "pos_y", "position_y": idx_y = i
            case "z", "pz", "pos_z", "position_z": idx_z = i
            case "scale_0", "scale0", "sx", "scale_x": idx_s0 = i
            case "scale_1", "scale1", "sy", "scale_y": idx_s1 = i
            case "scale_2", "scale2", "sz", "scale_z": idx_s2 = i
            case "rot_0", "rot0", "qw", "rotation_w": idx_r0 = i
            case "rot_1", "rot1", "qx", "rotation_x": idx_r1 = i
            case "rot_2", "rot2", "qy", "rotation_y": idx_r2 = i
            case "rot_3", "rot3", "qz", "rotation_z": idx_r3 = i
            case "opacity", "alpha": idx_op = i
            default:
                if name.starts(with: "f_dc_") || name.starts(with: "f_rest_") ||
                    name.starts(with: "sh_") || name.starts(with: "spherical_harmonics_")
                {
                    shMap.append((name, i))
                }
            }
        }

        guard idx_x >= 0, idx_y >= 0, idx_z >= 0 else {
            throw PLYLoaderError.missingRequiredProperties(["x", "y", "z"])
        }

        // Sort SH properties: f_dc first, then f_rest in order
        func shSortKey(_ name: String) -> Int {
            if name.starts(with: "f_dc_") { return Int(name.dropFirst(5)) ?? 0 }
            if name.starts(with: "f_rest_") { return 3 + (Int(name.dropFirst(7)) ?? 0) }
            if name.starts(with: "sh_") { return Int(name.dropFirst(3)) ?? 0 }
            return Int.max
        }
        shMap.sort { shSortKey($0.name) < shSortKey($1.name) }
        let idx_sh = shMap.map(\.index)
        let shStride = idx_sh.count

        // Format detection and vertex processing
        var records: [GaussianRecord] = []
        records.reserveCapacity(vertexCount)
        var shCoeffs: [Float] = []
        shCoeffs.reserveCapacity(vertexCount * shStride)

        var scaleIsLogSpace = true
        var opacityIsLogit = true

        data.withUnsafeBytes { rawBuffer in
            guard let base = rawBuffer.baseAddress else { return }

            func getFloat(v: Int, idx: Int) -> Float {
                guard idx >= 0 else { return 0 }
                let vOffset = bodyStart + v * stride
                switch vertex.properties[idx].type {
                case let .primitive(t):
                    switch t {
                    case .float32: return (base + vOffset + offsets[idx]).loadUnaligned(as: Float.self)
                    case .float64: return Float((base + vOffset + offsets[idx]).loadUnaligned(as: Double.self))
                    case .uint8: return Float((base + vOffset + offsets[idx]).loadUnaligned(as: UInt8.self)) / 255.0
                    case .int8: return Float((base + vOffset + offsets[idx]).loadUnaligned(as: Int8.self))
                    case .int16: return Float((base + vOffset + offsets[idx]).loadUnaligned(as: Int16.self))
                    case .uint16: return Float((base + vOffset + offsets[idx]).loadUnaligned(as: UInt16.self))
                    case .int32: return Float((base + vOffset + offsets[idx]).loadUnaligned(as: Int32.self))
                    case .uint32: return Float((base + vOffset + offsets[idx]).loadUnaligned(as: UInt32.self))
                    }
                case .list:
                    return 0
                }
            }

            // Sample first vertices for format detection
            let sampleCount = min(100, vertexCount)
            var sampleScales: [Float] = []
            var sampleOpacities: [Float] = []

            for v in 0 ..< sampleCount {
                if idx_s0 >= 0 { sampleScales.append(getFloat(v: v, idx: idx_s0)) }
                if idx_op >= 0 { sampleOpacities.append(getFloat(v: v, idx: idx_op)) }
            }

            // Detect scale format
            if !sampleScales.isEmpty {
                let hasNegScale = sampleScales.contains { $0 < 0 }
                let hasLargeScale = sampleScales.contains { $0 > 1.0 }
                let avgScale = sampleScales.reduce(0, +) / Float(sampleScales.count)

                if hasNegScale {
                    scaleIsLogSpace = true
                } else if !hasLargeScale, avgScale > 0, avgScale < 0.5 {
                    scaleIsLogSpace = false
                }
            }

            // Detect opacity format
            if !sampleOpacities.isEmpty {
                let minOp = sampleOpacities.min() ?? 0
                let maxOp = sampleOpacities.max() ?? 1
                opacityIsLogit = minOp < 0 || maxOp > 1.0
            }

            // Process all vertices
            for v in 0 ..< vertexCount {
                let s0 = getFloat(v: v, idx: idx_s0)
                let s1 = getFloat(v: v, idx: idx_s1)
                let s2 = getFloat(v: v, idx: idx_s2)
                let opacityRaw = getFloat(v: v, idx: idx_op)

                // Skip placeholder vertices
                let isPlaceholder = s0 == 2.0 && s1 == 2.0 && s2 == 2.0 && abs(opacityRaw - 4.8402) < 0.001
                if isPlaceholder { continue }

                // Position
                let pos = SIMD3<Float>(getFloat(v: v, idx: idx_x), getFloat(v: v, idx: idx_y), getFloat(v: v, idx: idx_z))

                // Scale
                let scale = scaleIsLogSpace
                    ? SIMD3<Float>(exp(s0), exp(s1), exp(s2))
                    : SIMD3<Float>(s0, s1, s2)

                // Rotation
                let rot = simd_normalize(simd_quatf(
                    ix: getFloat(v: v, idx: idx_r1),
                    iy: getFloat(v: v, idx: idx_r2),
                    iz: getFloat(v: v, idx: idx_r3),
                    r: getFloat(v: v, idx: idx_r0)
                ))

                // Opacity
                let opacity = opacityIsLogit ? 1.0 / (1.0 + exp(-opacityRaw)) : opacityRaw

                records.append(GaussianRecord(position: pos, scale: scale, rotation: rot, opacity: opacity, sh: []))

                // SH coefficients
                for idx in idx_sh {
                    shCoeffs.append(getFloat(v: v, idx: idx))
                }
            }
        }

        // Convert SH from PLY layout to shader planar layout
        let actualVertexCount = records.count
        let shComponents = shStride == 0 ? 0 : shStride / 3
        var shPlanar: [Float] = []

        if shComponents > 0, shCoeffs.count == actualVertexCount * shStride {
            let higherOrderCount = shComponents - 1
            shPlanar = [Float](repeating: 0, count: actualVertexCount * shStride)

            for i in 0 ..< actualVertexCount {
                let srcBase = i * shStride
                let dstBase = i * shStride

                // PLY: [DC_R, DC_G, DC_B, R1..R15, G1..G15, B1..B15]
                // Shader: [R0..R15, G0..G15, B0..B15]

                // R channel
                shPlanar[dstBase + 0] = shCoeffs[srcBase + 0]
                for c in 0 ..< higherOrderCount {
                    shPlanar[dstBase + 1 + c] = shCoeffs[srcBase + 3 + c]
                }

                // G channel
                shPlanar[dstBase + shComponents + 0] = shCoeffs[srcBase + 1]
                for c in 0 ..< higherOrderCount {
                    shPlanar[dstBase + shComponents + 1 + c] = shCoeffs[srcBase + 3 + higherOrderCount + c]
                }

                // B channel
                shPlanar[dstBase + shComponents * 2 + 0] = shCoeffs[srcBase + 2]
                for c in 0 ..< higherOrderCount {
                    shPlanar[dstBase + shComponents * 2 + 1 + c] = shCoeffs[srcBase + 3 + 2 * higherOrderCount + c]
                }
            }
        }

        // Recenter positions
        let bounds = GaussianSceneBuilder.bounds(of: records)
        let center = bounds.center
        var recentered = records
        if simd_length(center) > 1e-6 {
            for i in 0 ..< recentered.count {
                recentered[i].position -= center
            }
        }

        return GaussianDataset(
            records: recentered,
            harmonics: shPlanar,
            shComponents: shComponents,
            cameraPoses: [],
            cameraNames: [],
            imageSize: nil
        )
    }
}
