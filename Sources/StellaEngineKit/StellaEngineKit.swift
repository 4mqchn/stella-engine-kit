import Foundation
import onnxruntime

/// Stella Cheki のエンジン実証用ラッパー。
/// ONNX Runtime の C API を Swift から直接叩き、組み込みの加算モデルで推論する。
public enum StellaEngineKit {
    /// 成功時は "ONNX Runtime: OK — Y = [2.0, 3.0, 4.0]"、失敗時は理由を返す。
    public static func runOnnxProof() -> String {
        do {
            return "ONNX Runtime: OK — \(try inference())"
        } catch {
            return "ONNX Runtime: 失敗 — \(error)"
        }
    }

    private static func inference() throws -> String {
        let apiBase: UnsafePointer<OrtApiBase>? = OrtGetApiBase()
        guard let apiBase, let apiPointer = apiBase.pointee.GetApi(UInt32(ORT_API_VERSION)) else {
            throw ProofError("ONNX Runtime C API v\(ORT_API_VERSION) を取得できません")
        }
        let api = apiPointer.pointee

        func checkStatus(_ status: OpaquePointer?, _ operation: String) throws {
            guard let status else { return }
            defer { api.ReleaseStatus(status) }
            let message = String(cString: api.GetErrorMessage(status))
            throw ProofError("\(operation) — \(message)")
        }

        var env: OpaquePointer?
        defer { api.ReleaseEnv(env) }
        try checkStatus(api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "StellaEngineKit", &env), "CreateEnv")

        var options: OpaquePointer?
        defer { api.ReleaseSessionOptions(options) }
        try checkStatus(api.CreateSessionOptions(&options), "CreateSessionOptions")

        var session: OpaquePointer?
        defer { api.ReleaseSession(session) }
        try model.withUnsafeBytes { bytes in
            try checkStatus(
                api.CreateSessionFromArray(env, bytes.baseAddress, bytes.count, options, &session),
                "CreateSessionFromArray"
            )
        }

        var memoryInfo: OpaquePointer?
        defer { api.ReleaseMemoryInfo(memoryInfo) }
        try checkStatus(
            api.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo),
            "CreateCpuMemoryInfo"
        )

        var values: [Float] = [1, 2, 3]
        let shape: [Int64] = [3]
        let actual: [Float] = try values.withUnsafeMutableBytes { bytes in
            var input: OpaquePointer?
            defer { api.ReleaseValue(input) }
            try shape.withUnsafeBufferPointer { dimensions in
                try checkStatus(
                    api.CreateTensorWithDataAsOrtValue(
                        memoryInfo, bytes.baseAddress, bytes.count,
                        dimensions.baseAddress, dimensions.count,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input
                    ),
                    "CreateTensorWithDataAsOrtValue"
                )
            }

            var output: OpaquePointer?
            defer { api.ReleaseValue(output) }
            try "X".withCString { inputName in
                try "Y".withCString { outputName in
                    var inputNamePointer: UnsafePointer<CChar>? = inputName
                    var outputNamePointer: UnsafePointer<CChar>? = outputName
                    try checkStatus(
                        api.Run(session, nil, &inputNamePointer, &input, 1, &outputNamePointer, 1, &output),
                        "Run"
                    )
                }
            }

            var outputInfo: OpaquePointer?
            defer { api.ReleaseTensorTypeAndShapeInfo(outputInfo) }
            try checkStatus(api.GetTensorTypeAndShape(output, &outputInfo), "GetTensorTypeAndShape")

            var elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
            try checkStatus(api.GetTensorElementType(outputInfo, &elementType), "GetTensorElementType")
            var count = 0
            try checkStatus(api.GetTensorShapeElementCount(outputInfo, &count), "GetTensorShapeElementCount")
            guard elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, count == 3 else {
                throw ProofError("出力 Y の要素型または要素数が不一致 — 型 \(elementType.rawValue)、\(count) 要素")
            }

            var outputData: UnsafeMutableRawPointer?
            try checkStatus(api.GetTensorMutableData(output, &outputData), "GetTensorMutableData")
            guard let outputData else {
                throw ProofError("出力 Y のデータがありません")
            }
            return Array(UnsafeBufferPointer(
                start: outputData.assumingMemoryBound(to: Float.self), count: count
            ))
        }

        guard actual == [2, 3, 4] else {
            throw ProofError("出力 Y が不一致 — \(actual)")
        }
        return "Y = \(actual)"
    }

    private struct ProofError: Error, CustomStringConvertible {
        let description: String

        init(_ description: String) {
            self.description = description
        }
    }

    private static let model: [UInt8] = [
        0x08, 0x08, 0x3a, 0x50, 0x0a, 0x10, 0x0a, 0x01, 0x58, 0x0a, 0x03, 0x6f,
        0x6e, 0x65, 0x12, 0x01, 0x59, 0x22, 0x03, 0x41, 0x64, 0x64, 0x12, 0x09,
        0x53, 0x74, 0x65, 0x6c, 0x6c, 0x61, 0x41, 0x64, 0x64, 0x2a, 0x0f, 0x08,
        0x01, 0x10, 0x01, 0x42, 0x03, 0x6f, 0x6e, 0x65, 0x4a, 0x04, 0x00, 0x00,
        0x80, 0x3f, 0x5a, 0x0f, 0x0a, 0x01, 0x58, 0x12, 0x0a, 0x0a, 0x08, 0x08,
        0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x03, 0x62, 0x0f, 0x0a, 0x01, 0x59,
        0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x03,
        0x42, 0x02, 0x10, 0x0d,
    ]
}
