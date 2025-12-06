import Foundation
import simd

// MARK: - PLYHeader

public struct PLYHeader: Equatable {
    enum Keyword: String {
        case ply = "ply"
        case format = "format"
        case comment = "comment"
        case element = "element"
        case property = "property"
        case endHeader = "end_header"
        case obj_info = "obj_info"
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
            case .primitive(let t): return t.byteWidth
            case .list: return 0
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
            case .int8, .uint8: return 1
            case .int16, .uint16: return 2
            case .int32, .uint32, .float32: return 4
            case .float64: return 8
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
            case .headerUnknownKeyword(let keyword): "Unknown keyword: \"\(keyword)\""
            case .headerUnexpectedKeyword(let keyword): "Unexpected keyword: \"\(keyword)\""
            case .headerInvalidLine(let line): "Invalid line: \"\(line)\""
            case .headerInvalidFileFormatType(let type): "Invalid format type: \(type)"
            case .headerUnknownPropertyType(let type): "Unknown property type: \(type)"
            case .headerInvalidListCountType(let type): "Invalid list count type: \(type)"
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

enum PLYLoaderError: Error, LocalizedError {
    case invalidHeader
    case unsupportedFormat(PLYHeader.Format)
    case missingVertexElement
    case missingRequiredProperties([String])
    case listPropertiesNotSupported
    case insufficientData

    var errorDescription: String? {
        switch self {
        case .invalidHeader:
            return "Invalid PLY header"
        case .unsupportedFormat(let format):
            return "Unsupported PLY format: \(format.rawValue). Only binary_little_endian is supported."
        case .missingVertexElement:
            return "No 'vertex' element found in PLY"
        case .missingRequiredProperties(let props):
            return "Missing required properties: \(props.joined(separator: ", "))"
        case .listPropertiesNotSupported:
            return "List properties in vertex element are not supported"
        case .insufficientData:
            return "PLY file has insufficient data for declared vertex count"
        }
    }
}

// MARK: - PLYLoader

enum PLYLoader {
    /// Load a Gaussian PLY file using memory-mapped binary decoding.
    /// Only supports binary_little_endian format with primitive (non-list) vertex properties.
    static func load(url: URL, flipY: Bool = false) throws -> GaussianDataset {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])

        // Find end_header and parse header
        let endLF = Data("end_header\n".utf8)
        let endCRLF = Data("end_header\r\n".utf8)
        guard let headerEnd = data.range(of: endLF) ?? data.range(of: endCRLF) else {
            throw PLYLoaderError.invalidHeader
        }

        let headerData = data.subdata(in: data.startIndex..<headerEnd.upperBound)
        let header = try PLYHeader.decodeASCII(from: headerData)

        guard header.format == .binaryLittleEndian else {
            throw PLYLoaderError.unsupportedFormat(header.format)
        }

        guard let vertex = header.elements.first(where: { $0.name == "vertex" }) else {
            throw PLYLoaderError.missingVertexElement
        }

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

        let bodyStart = headerEnd.upperBound
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
                   name.starts(with: "sh_") || name.starts(with: "spherical_harmonics_") {
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
        let idx_sh = shMap.map { $0.index }
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
                case .primitive(let t):
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

            for v in 0..<sampleCount {
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
                } else if !hasLargeScale && avgScale > 0 && avgScale < 0.5 {
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
            for v in 0..<vertexCount {
                let s0 = getFloat(v: v, idx: idx_s0)
                let s1 = getFloat(v: v, idx: idx_s1)
                let s2 = getFloat(v: v, idx: idx_s2)
                let opacityRaw = getFloat(v: v, idx: idx_op)

                // Skip placeholder vertices
                let isPlaceholder = s0 == 2.0 && s1 == 2.0 && s2 == 2.0 && abs(opacityRaw - 4.8402) < 0.001
                if isPlaceholder { continue }

                // Position
                let rawY = getFloat(v: v, idx: idx_y)
                let pos = SIMD3<Float>(getFloat(v: v, idx: idx_x), flipY ? -rawY : rawY, getFloat(v: v, idx: idx_z))

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
                let finalRot: simd_quatf
                if flipY {
                    let S = simd_float3x3(diagonal: SIMD3<Float>(1, -1, 1))
                    let R = simd_float3x3(rot)
                    finalRot = simd_quatf(S * R * S)
                } else {
                    finalRot = rot
                }

                // Opacity
                let opacity = opacityIsLogit ? 1.0 / (1.0 + exp(-opacityRaw)) : opacityRaw

                records.append(GaussianRecord(position: pos, scale: scale, rotation: finalRot, opacity: opacity, sh: []))

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

        if shComponents > 0 && shCoeffs.count == actualVertexCount * shStride {
            let higherOrderCount = shComponents - 1
            shPlanar = [Float](repeating: 0, count: actualVertexCount * shStride)

            for i in 0..<actualVertexCount {
                let srcBase = i * shStride
                let dstBase = i * shStride

                // PLY: [DC_R, DC_G, DC_B, R1..R15, G1..G15, B1..B15]
                // Shader: [R0..R15, G0..G15, B0..B15]

                // R channel
                shPlanar[dstBase + 0] = shCoeffs[srcBase + 0]
                for c in 0..<higherOrderCount {
                    shPlanar[dstBase + 1 + c] = shCoeffs[srcBase + 3 + c]
                }

                // G channel
                shPlanar[dstBase + shComponents + 0] = shCoeffs[srcBase + 1]
                for c in 0..<higherOrderCount {
                    shPlanar[dstBase + shComponents + 1 + c] = shCoeffs[srcBase + 3 + higherOrderCount + c]
                }

                // B channel
                shPlanar[dstBase + shComponents * 2 + 0] = shCoeffs[srcBase + 2]
                for c in 0..<higherOrderCount {
                    shPlanar[dstBase + shComponents * 2 + 1 + c] = shCoeffs[srcBase + 3 + 2 * higherOrderCount + c]
                }
            }

            // Apply SH sign corrections when Y is flipped
            if flipY {
                let flipIndices = [1, 4, 5, 9, 10, 11]
                for i in 0..<actualVertexCount {
                    let base = i * shStride
                    for flipIdx in flipIndices where flipIdx < shComponents {
                        shPlanar[base + flipIdx] = -shPlanar[base + flipIdx]
                        shPlanar[base + shComponents + flipIdx] = -shPlanar[base + shComponents + flipIdx]
                        shPlanar[base + shComponents * 2 + flipIdx] = -shPlanar[base + shComponents * 2 + flipIdx]
                    }
                }
            }
        }

        // Recenter positions
        let bounds = GaussianSceneBuilder.bounds(of: records)
        let center = bounds.center
        var recentered = records
        if simd_length(center) > 1e-6 {
            for i in 0..<recentered.count {
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
