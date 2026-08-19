<div align="center">
<h1>avanalyze</h1>
</div>
<div align="center">

Apple Vision.framework keyframe analysis that builds detections into an output
vocabulary you name.

[<img alt="github" src="https://img.shields.io/badge/github-Findit--AI/avanalyze-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-avanalyze-66c2a5?style=for-the-badge&labelColor=555555" height="20">][doc-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge" height="22">

</div>

## What it does

`avanalyze` wraps Apple's Vision.framework in a synchronous Rust API. One
`VisionAnalyzer` owns one of every supported request kind — face,
face-landmark, body-pose, body-pose-3D, hand-pose, classification, saliency,
aesthetics, barcode, text, horizon, animal, animal-body-pose,
person-segmentation, person-instance-mask, document-segmentation — at fixed,
pinned revisions. `analyze_keyframe` runs them all against a single JPEG and
returns an `Analysis`.

## What it does not do

It does not know what a keyframe *is*. The engine mints no identifiers, carries
no timestamp, reads no frame dimensions, and assembles no aggregate: an
`Analysis` is eighteen flat slots of detections and nothing else. Composing
those into whatever record you store is your job, because you are the only one
who knows the frame's identity.

It also has no opinion about what a detection is made of. Every output type is
yours:

```rust,ignore
use avanalyze::{AnalyzeOptions, Detections, VisionAnalyzer};

// A bundle marker naming your types; each one implements the matching
// per-part trait (`BoundingBox`, `FaceDetection`, `DocumentSegment`, …).
struct MyVocabulary;
impl Detections for MyVocabulary { /* … */ }

let options = AnalyzeOptions::new();
let analyzer = VisionAnalyzer::new(&options);
let analysis = analyzer.analyze_keyframe::<MyVocabulary>(&jpeg, &options)?;

for face in analysis.faces() { /* your type */ }
```

Every seat on every constructor is a primitive — `f32`, `u32`, `usize`, `&str`,
`&[u8]`, tuples of those — or one of two engine-owned enums. No schema crate,
no wire format, no foreign type crosses the boundary, and `avanalyze` depends on
none of them.

### Proving your vocabulary fits

The traits carry signatures; the conventions they cannot express — the unit
square, the "nothing detected" sentinels, the winding order of document corners,
the argument order of the horizon — live in `avanalyze::conformance` as
assertions you run:

```rust,ignore
#[test]
fn my_vocabulary_fits_the_engine() {
  avanalyze::conformance::assert_contract::<MyVocabulary>();
  // …and, if your types validate their input:
  avanalyze::conformance::assert_refuses_invalid::<MyVocabulary>();
}
```

## Degradation

Failure is per detector, not per frame. A Vision request that raises an
Objective-C exception contributes an empty slot while every other detector's
results still land. Individual detections are filtered before construction —
non-finite geometry, out-of-range confidences, degenerate boxes — and a refused
detection is silently absent; there is no "dropped" counter. An `Err` means no
analysis happened at all, which is a surface of exactly two kinds.

## Requirements

- macOS (Vision.framework is Apple-only).
- A working `objc2` toolchain (Xcode command-line tools).
- Rust **1.95** or newer (edition 2024).

On every other target the `cfg(target_vendor = "apple")` gates drop the platform
deps entirely and `analyze_keyframe` reports `AnalyzeErrorKind::Unsupported`, so
downstream workspaces can keep `avanalyze` in their dependency tree
unconditionally.

## Layout

- `src/contract.rs` — the `Detections` bundle and the nineteen per-part traits.
- `src/analysis.rs` — `Analysis<D>`, the eighteen-slot output carton.
- `src/conformance.rs` — runnable assertions for an implementor's vocabulary.
- `src/lib.rs` — `VisionAnalyzer`, the pinned request set, and the per-request
  extractors that translate `VNObservation`s into your types.
- `src/options.rs` — per-request configuration knobs
  (`AppleVisionClassificationOptions`, `…BodyPoseOptions`, …) and the top-level
  `AnalyzeOptions`.
- `src/error.rs` — `AnalyzeError` and its two kinds.

## License

`avanalyze` is licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT license](LICENSE-MIT)

at your option.

[Github-url]: https://github.com/findit-studio/avanalyze
[doc-url]: https://docs.rs/avanalyze
