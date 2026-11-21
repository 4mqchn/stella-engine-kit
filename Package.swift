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
            path: "Frameworks/onnxruntime.xcframework"
        ),
        .target(
            name: "StellaEngineKit",
            dependencies: ["onnxruntime"]
        )
    ]
)
