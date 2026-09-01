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

`avanalyze` wraps Apple's Vision.framework in a synchronous Rust API as **nine
entry points**, each owning exactly the Vision requests its own capability
needs, at fixed, pinned revisions:

| Entry point | Requests | Produces |
|---|---|---|
| `VisionAnalyzer` | 8 | `Analysis` — classifications, human and animal subjects, both saliency passes, document segments, horizon, aesthetics |
| `TextRecognizer` | 1 | recognised text runs |
| `BarcodeDetector` | 1 | decoded barcodes |
| `FaceDetector` | 3 | one record per face: box, confidence, capture quality, roll/yaw/pitch, five keypoints |
| `FaceLandmarker` | 1 | all thirteen named landmark regions |
| `BodyPoser` | 2 | human 2-D and 3-D poses |
| `HandPoser` | 1 | hand poses |
| `AnimalPoser` | 1 | animal 2-D poses |
| `PersonMasker` | 2 | per-instance and whole-frame person masks |

Construct only what you use. A consumer that wants text builds a
`TextRecognizer`, names one output type, and loads exactly one Vision model —
no face, pose, or mask model is constructed, let alone run.

## What it does not do

It does not know what a keyframe *is*. The engine mints no identifiers, carries
no timestamp, reads no frame dimensions, and assembles no aggregate: an
`Analysis` is eight flat slots of detections and nothing else, and every other
entry point hands back a plain `Vec`. Composing those into whatever record you
store is your job, because you are the only one who knows the frame's identity.

It also has no opinion about what a detection is made of. Every output type is
yours:

```rust,ignore
use avanalyze::{AnalyzeOptions, Detections, VisionAnalyzer};

// A bundle marker naming the seven types the one-pass analyzer builds;
// each implements the matching trait (`BoundingBox`, `SubjectDetection`,
// `DocumentSegment`, …).
struct MyVocabulary;
impl Detections for MyVocabulary { /* … */ }

let options = AnalyzeOptions::new();
let analyzer = VisionAnalyzer::new(&options);
let analysis = analyzer.analyze_keyframe::<MyVocabulary>(&jpeg, &options)?;

for subject in analysis.human_subjects() { /* your type */ }
```

Every other entry point takes **one** type parameter — the type it builds — so
a single-capability consumer never names a bundle:

```rust,ignore
use avanalyze::{AppleVisionTextOptions, TextDetection, TextRecognizer};

struct MyTextRun { /* … */ }
impl TextDetection for MyTextRun { /* … */ }

let options = AppleVisionTextOptions::new();
let runs = TextRecognizer::new(&options).recognize::<MyTextRun>(&jpeg, &options)?;
```

Every seat on every constructor is a primitive — `f32`, `u32`, `usize`, `&str`,
`&[u8]`, tuples of those — or an engine-owned type (`Chirality`,
`HeightEstimation`, `FaceKeypoints`). No schema crate, no wire format, no
foreign type crosses the boundary, and `avanalyze` depends on none of them.

Faces come with the 76→5-point reduction — both eye centres, the nose tip, and
both mouth corners, in image-normalized coordinates and canonical alignment
order. What you crop, align, or embed with them is your business: the engine
produces geometry and never an identity.

### Proving your vocabulary fits

The traits carry signatures; the conventions they cannot express — the unit
square, the "nothing detected" sentinels, the winding order of document corners,
the argument order of the horizon — live in `avanalyze::conformance` as
assertions you run. They are per entry point, so you assert only what you use:

```rust,ignore
#[test]
fn my_vocabulary_fits_the_engine() {
  avanalyze::conformance::assert_contract::<MyVocabulary>();
  avanalyze::conformance::assert_text_accepts::<MyTextRun>();
  // …and, if your types validate their input:
  avanalyze::conformance::assert_refuses_invalid::<MyVocabulary>();
  avanalyze::conformance::assert_text_refusals::<MyTextRun>();
}
```

## Degradation

Failure is per detector, not per frame. A Vision request that raises an
Objective-C exception contributes an empty result while every other detector's
results still land — an empty slot inside `Analysis`, an empty `Vec` from an
entry point of its own. Individual detections are filtered before construction —
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

One module per entry point, each holding its own requests, its own extractors,
and the traits it builds:

- `src/analyzer.rs` — `VisionAnalyzer`, the eight batched detections.
- `src/analysis.rs` — `Analysis<D>`, the eight-slot output carton.
- `src/contract.rs` — the core `Detections` bundle and the seven traits it names.
- `src/text.rs`, `src/barcode.rs` — the two readers and their traits.
- `src/face.rs` — `FaceDetector`, `FaceDetection`, `FaceKeypoints`.
- `src/face_landmarks.rs` — `FaceLandmarker` and the region traits.
- `src/body_pose.rs`, `src/hand_pose.rs`, `src/animal_pose.rs` — the pose entry
  points, their traits, `Chirality`, `HeightEstimation`.
- `src/person_mask.rs` — `PersonMasker`, both mask traits, the pixel-buffer copy.
- `src/ffi.rs` — the shared Vision boundary: coordinate conversion, resource
  ceilings, the Objective-C exception barrier.
- `src/conformance.rs` — runnable assertions, per entry point.
- `src/options.rs` — per-entry configuration knobs
  (`AppleVisionTextOptions`, `AppleVisionFaceOptions`, …) and `AnalyzeOptions`.
- `src/error.rs` — `AnalyzeError` and its two kinds.

## License

`avanalyze` is licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT license](LICENSE-MIT)

at your option.

[Github-url]: https://github.com/findit-studio/avanalyze
[doc-url]: https://docs.rs/avanalyze
