// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "StellaEngineKit",
    platforms: [.iOS(.v16)],
    products: [
        .library(name: "StellaEngineKit", targets: ["StellaEngineKit"])
    ],
    targets: [
        .binaryTarget(
            name: "onnxruntime",
            url: "https://github.com/4mqchn/stella-engine-kit/releases/download/0.1.0/onnxruntime.xcframework.zip",
            checksum: "24cb57479ba43ab5028cdfa366e7f3c4a0f1a1ca01293308df8b7c457e3dcc1f"
        ),
        .binaryTarget(
            name: "opencv2",
            url: "https://github.com/4mqchn/stella-engine-kit/releases/download/0.1.0/opencv2.xcframework.zip",
            checksum: "367e47834dd5eda8b741640013b83fa9c469f02a64e1f066bc95f768b0ac4fa0"
        ),
        .target(
            name: "StellaEngineKit",
            dependencies: ["onnxruntime", "opencv2"]
        )
    ]
)
