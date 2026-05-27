<div align="center">
<h1>avanalyze</h1>
</div>
<div align="center">

Long-running Apple Vision.framework worker that analyses keyframes and emits
[`mediaschema`](https://github.com/Findit-AI/mediaschema)-shaped detections.

[<img alt="github" src="https://img.shields.io/badge/github-Findit--AI/avanalyze-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-avanalyze-66c2a5?style=for-the-badge&labelColor=555555" height="20">][doc-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge" height="22">

</div>

## What it does

`avanalyze` wraps Apple's Vision.framework with a synchronous Rust API. A
single [`VisionAnalyzer`] owns one of every supported request kind
(face / body-pose / body-pose-3D / hand-pose / classification /
saliency / aesthetics / barcode / text / horizon / animal / animal-body-pose
/ person-segmentation / person-instance-mask / document-segmentation)
at fixed, pinned revisions, and `analyze_keyframe(...)` runs them all
against a single JPEG and packages the results into one
[`mediaschema::domain::Keyframe`].

The output is the **validated domain shape** — `Keyframe<Uuid7>` with
`try_new`-style detection value objects (`BoundingBox`, `Confidence`,
`NormCoord`, …). Serialisation to the wire / sqlx / mongodb backends
happens inside `mediaschema`, not at the engine boundary.

Note: `feature_print` detections previously emitted by Apple's
`VNGenerateImageFeaturePrintRequest` are no longer part of the
keyframe payload — feature embeddings live in LanceDB keyed by the
keyframe id under the locked schema, so they are produced by a
separate downstream stage rather than at the Vision-engine boundary.

## Requirements

- macOS (Vision.framework is Apple-only).
- A working `objc2` toolchain (Xcode command-line tools).
- Rust **1.95** or newer (edition 2024).

On non-macOS targets the `cfg(target_os = "macos")` gates make the
platform deps drop out entirely; the crate still compiles as a no-op so
downstream workspaces can keep `avanalyze` in their dep tree
unconditionally.

## Status

Pre-release (`0.0.0`). The data plane — `VisionAnalyzer::analyze_keyframe`
— is functional. Service-framework integration (`ThreadService`,
`Request` / `Reply`, `handle_message`) is **commented out** pending the
external `findit-service` / `findit-pipeline` crates landing in the
workspace; once those exist the block at the top of `src/lib.rs` will
be re-enabled.

## Layout

- `src/lib.rs` — `VisionAnalyzer`, the request set, and the
  per-request extractors that translate `VNObservation`s into
  `mediaschema` detections.
- `src/options.rs` — per-request configuration knobs
  (`AppleVisionClassificationOptions`, `…BodyPoseOptions`, …) and the
  top-level `ServiceOptions`.
- `src/wire_ext.rs` — local extension traits that give the
  `mediaschema` wire types ergonomic `::new(…)` / `.with_*(…)` builder
  surfaces (the proto-generated structs ship as
  `#[derive(Default)]` records with public fields and no
  constructors).

## License

`avanalyze` is licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT license](LICENSE-MIT)

at your option.

[Github-url]: https://github.com/Findit-AI/avanalyze
[doc-url]: https://docs.rs/avanalyze
