#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

//! Apple Vision.framework, wrapped so it only ever *produces*.
//!
//! One [`VisionAnalyzer`] per worker thread runs every configured
//! Vision request against a JPEG and hands back an [`Analysis`] — a
//! flat set of detections in whatever output vocabulary the caller
//! named through the [`Detections`] bundle. The engine mints no
//! identifiers, assembles no aggregate, and depends on no schema
//! crate; Vision.framework is stateless per-request, so workers run in
//! parallel.

#[cfg(target_vendor = "apple")]
use std::{
  borrow::Cow,
  panic::{AssertUnwindSafe, catch_unwind},
};

#[cfg(target_vendor = "apple")]
use objc2::{
  encode::{Encode, Encoding},
  exception::catch as catch_objc_exception,
  rc::Retained,
};
#[cfg(target_vendor = "apple")]
use objc2_core_foundation::{CGPoint, CGRect};
#[cfg(target_vendor = "apple")]
use objc2_core_video::{
  CVPixelBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
  CVPixelBufferGetDataSize, CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType,
  CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
  CVPixelBufferUnlockBaseAddress, kCVPixelFormatType_OneComponent8,
  kCVPixelFormatType_OneComponent32Float, kCVReturnSuccess,
};
#[cfg(target_vendor = "apple")]
use objc2_foundation::{NSArray, NSData, NSIndexSet, NSNotFound};
#[cfg(target_vendor = "apple")]
use objc2_vision::*;
#[cfg(target_vendor = "apple")]
use smol_str::{SmolStr, StrExt, ToSmolStr};

pub use analysis::*;
pub use contract::*;
pub use error::*;
pub use options::*;

mod analysis;
pub mod conformance;
mod contract;
mod error;
mod options;

#[cfg(test)]
mod tests;

#[cfg(target_vendor = "apple")]
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
struct SimdFloat4([f32; 4]);

#[cfg(target_vendor = "apple")]
unsafe impl Encode for SimdFloat4 {
  // `simd_float4` is an `__attribute__((__ext_vector_type__))` type;
  // Clang intentionally emits NO `@encode` for ext-vector elements, so
  // the matching Rust-side encoding is [`Encoding::None`] (formats as
  // empty string), NOT [`Encoding::Unknown`] (formats as `?`).
  //
  // objc2-encode's `Encoding::None` docstring explicitly calls this
  // out as the SIMD-vector case. The previous `Encoding::Unknown`
  // made the wrapping struct render as `{?=[4?]}`, while Vision's
  // `-[VNHumanBodyRecognizedPoint3D position]` returns `{?=[4]}`
  // (Clang refuses to emit an inner element character) — every
  // msg_send for that selector failed verification on macOS 26.x,
  // and the surrounding `catch_unwind` silently swallowed the
  // panic so 3-D pose detections were always dropped to zero.
  const ENCODING: Encoding = Encoding::None;
}

#[cfg(target_vendor = "apple")]
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
struct SimdFloat4x4 {
  columns: [SimdFloat4; 4],
}

#[cfg(target_vendor = "apple")]
unsafe impl Encode for SimdFloat4x4 {
  // Apple's `simd_float4x4` is a struct-of-vectors. Clang reports
  // `@encode(simd_float4x4)` as `{?=[4]}` — outer struct with no name
  // wrapping an array of 4 whose element type Clang refuses to encode
  // (the element is itself an ext-vector, see [`SimdFloat4`] above).
  // The matching Rust encoding therefore uses `Array(4, &None)` so
  // the inner array element formats to an empty string, producing the
  // literal `[4]` Clang emits.
  const ENCODING: Encoding = Encoding::Struct("?", &[Encoding::Array(4, &Encoding::None)]);
}

// ----- Vision → contract coordinate conversion ------------------------------

/// Clamp a finite `f32` into `[0.0, 1.0]`. Callers MUST filter
/// non-finite inputs before invoking this helper — passing `NaN` /
/// `±Inf` is a regression (collapsing them to `0.0` here previously
/// fabricated edge-aligned coordinates that downstream validators
/// accepted as real detections). The `debug_assert!` catches the
/// regression in debug builds without changing release behaviour
/// (`f32::clamp(0.0, 1.0)` on `NaN` returns `NaN`, and on `±Inf`
/// returns the appropriate edge — both of which the domain
/// `NormCoord::try_new` will reject downstream, so we still
/// degrade safely rather than panicking).
#[cfg(target_vendor = "apple")]
#[inline]
fn clamp01(value: f32) -> f32 {
  debug_assert!(
    value.is_finite(),
    "clamp01 expects finite input; got {value}"
  );
  value.clamp(0.0, 1.0)
}

/// Convert a Vision-framework normalized bounding box (lower-left
/// origin, y grows up) into the contract's convention (top-left
/// origin, y grows down) and intersect it with the unit square
/// `[0, 1] × [0, 1]`.
///
/// [`BoundingBox`] documents floats in `[0.0, 1.0]` with a top-left
/// origin, while `VNObservation::boundingBox` is documented as a
/// normalized rect in image coordinates where `(0,0)` is the
/// lower-left corner. Vision is empirically loose about staying inside
/// `[0, 1]` — partially off-screen detections can produce
/// `origin.x < 0`, `origin.x + width > 1`, etc., which a validating
/// [`BoundingBox::try_new`] would reject. We clamp every component and
/// return `None` if the resulting rectangle is degenerate
/// (zero-width or zero-height); the detection is then dropped at the
/// engine layer instead of poisoning downstream storage.
///
/// `standardize()` is assumed to have already been called on `rect`;
/// the input `size` is non-negative.
#[cfg(target_vendor = "apple")]
fn vision_rect_to_bbox<B: BoundingBox>(rect: CGRect) -> Option<B> {
  // Vision lower-left → schema top-left: the top edge in schema space
  // is `1.0 - (origin.y + size.height)`.
  let raw_x = rect.origin.x as f32;
  let raw_y = (1.0 - (rect.origin.y + rect.size.height)) as f32;
  let raw_width = rect.size.width as f32;
  let raw_height = rect.size.height as f32;

  // Front-load the non-finite check: any `NaN` / `±Inf` in the raw
  // rectangle means the box is geometrically meaningless. Drop it
  // instead of letting `clamp01` (which used to collapse non-finite
  // to `0.0`) fabricate an edge-aligned rectangle that downstream
  // validation would accept.
  if !(raw_x.is_finite() && raw_y.is_finite() && raw_width.is_finite() && raw_height.is_finite()) {
    return None;
  }

  // Intersect with the unit square. Compute right/bottom in raw space,
  // then clamp the four edges so we never end up with `x + width > 1`.
  let left = clamp01(raw_x);
  let top = clamp01(raw_y);
  let right = clamp01(raw_x + raw_width);
  let bottom = clamp01(raw_y + raw_height);
  let width = (right - left).max(0.0);
  let height = (bottom - top).max(0.0);
  if width <= 0.0 || height <= 0.0 {
    return None;
  }
  // The clamp logic above guarantees the invariants a validating
  // vocabulary checks (finite, `[0, 1]`, positive extent,
  // `left + width <= 1.0`); a failure here would be a regression in
  // the upstream guards, not a real Vision input.
  B::try_new(left, top, width, height).ok()
}

/// Flip a Vision normalized point's y axis to match the contract's
/// top-left origin and clamp both components into `[0.0, 1.0]`.
/// [`BoundingBox`], [`BodyPoseJoint`] (2-D), face-landmark points, and
/// [`DocumentSegment`] corners all share the top-left convention.
/// 3-D joints ([`BodyPose3DJoint`]) are model-space metres and are NOT
/// flipped or clamped.
///
/// Returns `None` when either input coordinate is non-finite. A `NaN`
/// or `±Inf` from a glitched Vision observation is geometrically
/// meaningless and previously sanitised to `0.0` via `clamp01`, which
/// fabricated edge-aligned coordinates indistinguishable from real
/// detections. The caller decides whether a single bad point drops
/// the entire detection (e.g. a document quad without all four
/// corners) or just the offending point (e.g. one bad joint among
/// many).
#[cfg(target_vendor = "apple")]
#[inline]
fn vision_point_to_normalized(x: f64, y: f64) -> Option<(f32, f32)> {
  let x32 = x as f32;
  let flipped_y = (1.0 - y) as f32;
  if !x32.is_finite() || !flipped_y.is_finite() {
    return None;
  }
  Some((clamp01(x32), clamp01(flipped_y)))
}

/// Reject non-finite Vision-derived scalars. `NaN` / `±Inf` from
/// glitched Vision observations would otherwise enter the wire as
/// valid-looking detections and later trip downstream validation or
/// silently fail-open through `<` / `>` comparisons (since every
/// comparison against `NaN` is `false`). Callers convert `None` into
/// either a structured "drop the containing detection" decision or a
/// concrete default (typically `0.0`) — the choice depends on whether
/// the scalar is required geometry/score (drop) or an optional pose
/// angle (default).
#[cfg(target_vendor = "apple")]
#[inline]
fn finite_f32(v: f32) -> Option<f32> {
  if v.is_finite() { Some(v) } else { None }
}

/// Upper bound on a single mask payload (post-packing, 8 bits per
/// pixel) before we refuse to allocate. 64 MiB covers any sane image
/// resolution Apple Vision returns today (8K = ~33 MiB at 8 bits per
/// pixel) and prevents a runaway / corrupted `width * height` from
/// driving the worker process into the allocator's abort path.
#[cfg(target_vendor = "apple")]
const MAX_MASK_BYTES: usize = 64 * 1024 * 1024;

/// Upper bound on the number of landmark points per face-landmark
/// region. Vision's `allPoints` is ~76 points; per-feature regions
/// are smaller. 1024 leaves headroom against future API expansion
/// while still capping a corrupted/adversarial `pointCount`.
#[cfg(target_vendor = "apple")]
const MAX_LANDMARK_POINTS: usize = 1024;

/// Upper bound on the number of detection results from a single
/// Vision request before we refuse to pre-allocate OR iterate the
/// FFI-reported `results` array. Apple's per-frame extractor
/// outputs cap out in the low hundreds at most (text recognition,
/// face capture, etc.); 4096 is a generous defence-in-depth
/// ceiling against a corrupted / adversarial `NSArray` length that
/// would otherwise drive either the initial `Vec::with_capacity`
/// or the in-loop `Vec::push` reallocation into the allocator's
/// abort path. Every `for x in results.iter()` is bounded with
/// `.iter().take(MAX_VISION_RESULTS_PER_FRAME)` so the emitted
/// count cannot exceed the cap, independently of whatever
/// configured `max_results` / `max_segments` / … the call site
/// uses inside the loop.
#[cfg(target_vendor = "apple")]
const MAX_VISION_RESULTS_PER_FRAME: usize = 4096;

/// Upper bound on the number of joints / recognised points per
/// pose observation before we refuse to materialise the joint
/// dictionary via `NSDictionary::to_vecs()`. Apple's body-pose /
/// hand-pose / animal-pose joint counts are fixed by the SDK
/// (~17 body, ~21 hand, ~25 animal); 256 leaves headroom against
/// future API expansion while still capping a corrupted /
/// adversarial `points_by_joint.len()` that would otherwise drive
/// `to_vecs()`'s internal allocations into the abort path before
/// the per-extractor logic can drop the pose.
#[cfg(target_vendor = "apple")]
const MAX_POSE_JOINTS: usize = 256;

/// Hard ceiling on instances per segmentation-mask observation.
#[cfg(target_vendor = "apple")]
const MAX_NESTED_INSTANCES_PER_OBSERVATION: usize = 64;

/// Hard ceiling on labels per recognised-animal observation.
#[cfg(target_vendor = "apple")]
const MAX_NESTED_LABELS_PER_OBSERVATION: usize = 32;

/// Hard ceiling on candidate strings per text-recognition
/// observation. Apple's
/// `VNRecognizedTextObservation::topCandidates(_:)` documents an
/// upper limit of 10 — requesting more violates the Objective-C
/// API contract and can surface as a framework exception or
/// undefined behaviour across OS versions, so we clamp to 10 here
/// even though realistic workloads ask for 1-3 candidates
/// (codex R17 F3).
#[cfg(target_vendor = "apple")]
const MAX_TEXT_CANDIDATES_PER_OBSERVATION: usize = 10;

/// Hard ceiling on saliency regions per frame.
#[cfg(target_vendor = "apple")]
const MAX_SALIENCY_REGIONS_PER_FRAME: usize = 64;

/// Hard ceiling on the total mask count emitted per frame across
/// the inner observation × instance loop. Without this, an outer
/// cap of 4096 observations × an inner cap of 64 instances would
/// permit 256K mask emissions per frame even though each individual
/// mask is capped at [`MAX_MASK_BYTES`]. 256 is a generous total
/// matching realistic Vision output for a single frame.
#[cfg(target_vendor = "apple")]
const MAX_TOTAL_MASKS_PER_FRAME: usize = 256;

/// Hard ceiling on the cumulative mask payload bytes emitted per
/// frame. Even at the per-mask cap, a worst-case 256 masks × 64 MiB
/// = 16 GiB would crush the worker. 256 MiB total is generous for
/// realistic Vision output while bounding the cumulative budget.
#[cfg(target_vendor = "apple")]
const MAX_TOTAL_MASK_BYTES_PER_FRAME: usize = 256 * 1024 * 1024;

/// Hard ceiling on the cumulative ATTEMPTED mask generations per
/// frame, summed across both mask extractors. The emission-only
/// counters (count, bytes) only increment after a successful copy
/// and push; a corrupted Vision result whose generate-copy-u32-fit
/// gates all fail would otherwise drive unbounded
/// `generateMaskForInstances_error` calls (each forcing Vision to
/// compute/allocate a mask buffer) while the emission counters
/// stay below their caps. The attempt budget bounds the
/// failure-path work itself (codex R15 + R16).
///
/// Sized as `4 * MAX_TOTAL_MASKS_PER_FRAME` (= 1024): each emitted
/// mask gets up to 3 failed sibling attempts before the budget
/// trips, which leaves ample headroom for transient Vision
/// failures while keeping the attempt cap tied to the emission
/// cap rather than the much larger general results-array cap.
/// Apple's mask APIs return a small number of masks per frame
/// (low units for segmentation, typically <20 for instance masks),
/// so 1024 attempts is conservative against realistic workloads
/// while still defending against the corrupted-array case.
#[cfg(target_vendor = "apple")]
const MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME: usize = 4 * MAX_TOTAL_MASKS_PER_FRAME;

/// Hard ceiling on the cumulative ATTEMPTED face-landmark points
/// visited per frame across all detections × all 13 named regions.
/// Symmetric with `MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME`: a corrupted
/// observation set where every region's points fail
/// `vision_point_to_normalized`'s finite check (or where the parent
/// detection later fails the bbox / min_region_count gates) would
/// otherwise let the helper walk up to
/// `4096 * 13 * MAX_LANDMARK_POINTS` raw points without the
/// per-frame emission budget ever decreasing. Sized as
/// `4 * MAX_FACE_LANDMARK_POINTS_PER_FRAME` so a successful frame
/// can tolerate non-finite/dropped points before the attempt cap
/// trips (codex R17 F1).
#[cfg(target_vendor = "apple")]
const MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME: usize = 4 * MAX_FACE_LANDMARK_POINTS_PER_FRAME;

/// Hard ceiling on the cumulative face-landmark points emitted per
/// frame across all detections × all 13 named regions × all points.
/// Apple's typical output is at most a few faces × ~76 points each,
/// so 16384 is generous defence-in-depth against the worst-case
/// nested-emission product (4096 × 13 × MAX_LANDMARK_POINTS).
#[cfg(target_vendor = "apple")]
const MAX_FACE_LANDMARK_POINTS_PER_FRAME: usize = 16384;

/// Hard ceiling on the total animal-subject rows emitted per frame.
/// Apple's animal recogniser returns a few species per frame at most;
/// 256 caps the adversarial 4096 × MAX_NESTED_LABELS_PER_OBSERVATION
/// product without restricting real workloads.
#[cfg(target_vendor = "apple")]
const MAX_TOTAL_ANIMAL_SUBJECTS_PER_FRAME: usize = 256;

/// Hard ceiling on the total text detections emitted per frame.
/// 256 caps the adversarial 4096 × MAX_TEXT_CANDIDATES_PER_OBSERVATION
/// product without restricting real text-rich-document workloads.
#[cfg(target_vendor = "apple")]
const MAX_TOTAL_TEXT_DETECTIONS_PER_FRAME: usize = 256;

/// Upper bound on the input image byte length accepted by
/// [`VisionAnalyzer::analyze_keyframe`]. Pre-validates the keyframe
/// payload BEFORE Foundation copies it into an `NSData`, so an
/// oversized or hostile input cannot double the worker's peak
/// memory and drive the allocator into the abort path. 64 MiB
/// covers an extremely generous keyframe (Apple's typical
/// keyframe encoded JPEG is well under 1 MiB); inputs above that
/// surface as a structured [`AnalyzeError`] instead of an alloc-side
/// crash (codex R17 F1).
#[cfg(target_vendor = "apple")]
const MAX_INPUT_IMAGE_BYTES: usize = 64 * 1024 * 1024;

/// Apple's documented maximum for
/// `VNDetectHumanHandPoseRequest::setMaximumHandCount(_:)` at
/// revision 1 — the request becomes invalid above this. The pinned
/// request revision in this crate is revision 1; configurations
/// requesting more must clamp at extractor build time to avoid an
/// Objective-C exception crossing the FFI boundary (codex R17 F2).
#[cfg(target_vendor = "apple")]
const MAX_HAND_POSE_MAXIMUM_HAND_COUNT: usize = 6;

/// Upper bound on the byte length of an FFI-sourced `NSString`
/// before we refuse to convert it to a Rust `SmolStr` / `String`.
/// Apple's Vision-emitted strings (OCR text, barcode payloads,
/// classification identifiers, joint names) cap out in the low
/// hundreds of bytes for realistic content; 4096 is a generous
/// defence-in-depth ceiling against a corrupted / adversarial
/// `NSString` whose reported length would drive Rust's infallible
/// string allocation into the abort path. Strings exceeding the
/// cap are dropped; callers skip the offending field rather than
/// truncating mid-grapheme.
#[cfg(target_vendor = "apple")]
const MAX_FFI_STRING_BYTES: usize = 4096;

/// Convert an FFI-sourced `NSString` to a Rust `SmolStr` after
/// verifying its UTF-8 byte length is within
/// [`MAX_FFI_STRING_BYTES`]. Returns `None` if the `NSString`'s
/// reported byte length exceeds the bound; callers drop the
/// offending field (text candidate / barcode payload /
/// classification label / joint name) rather than driving the
/// allocator into the abort path. The length query is FFI but
/// allocation-free.
#[cfg(target_vendor = "apple")]
fn ffi_nsstring_to_smolstr(ns_str: &objc2_foundation::NSString) -> Option<SmolStr> {
  // `NSStringEncoding` is a `usize` type alias (objc2_foundation
  // re-exports it from `objc2::ffi::NSUInteger = usize`).
  // `NSUTF8StringEncoding` is the documented value 4.
  const NS_UTF8_STRING_ENCODING: objc2_foundation::NSStringEncoding = 4;
  // `lengthOfBytesUsingEncoding` is exposed as safe by
  // objc2-foundation 0.3.2 — no `unsafe` wrapper required.
  let utf8_len: usize = ns_str.lengthOfBytesUsingEncoding(NS_UTF8_STRING_ENCODING);
  if utf8_len > MAX_FFI_STRING_BYTES {
    return None;
  }
  Some(ns_str.to_smolstr())
}

/// Run `f` under an Objective-C exception barrier, returning `fallback`
/// (the empty/degraded result for that detector) if Apple's Vision
/// framework raises an `NSException` that unwinds across the FFI
/// boundary.
///
/// Rust's [`std::panic::catch_unwind`] (used elsewhere in this file for
/// a *Rust*-panic quirk) explicitly **cannot** catch a foreign
/// Objective-C exception — one that escaped would abort the entire
/// process with `fatal runtime error: Rust cannot catch foreign
/// exceptions`. [`objc2::exception::catch`] is the only sanctioned
/// barrier: it converts the unwinding `NSException` into a `Result`, so
/// one misbehaving detector degrades to an empty result for that
/// detector instead of taking the whole worker (and pipeline) down. A
/// caught exception's `name`/`reason` is logged via the exception's
/// (safe) `Display` impl.
///
/// `detector` names the Vision request whose perform/result-extraction
/// raised, so the warning pins the culprit.
///
/// The closure is wrapped in [`AssertUnwindSafe`] internally: it
/// captures `&VisionAnalyzer` (whose retained `VNRequest` handles are
/// not `RefUnwindSafe`) and, for the mask extractors, `&mut` budget
/// counters. Asserting unwind-safety is sound here — on a caught
/// exception the closure's borrows are dropped and `fallback` is
/// returned, so no half-updated state is ever observed. A partially
/// advanced mask counter only ever over-counts (the conservative
/// direction for a resource cap), never under-counts.
#[cfg(target_vendor = "apple")]
fn guard_vision_ffi<R>(detector: &'static str, fallback: R, f: impl FnOnce() -> R) -> R {
  match catch_objc_exception(AssertUnwindSafe(f)) {
    Ok(value) => value,
    Err(exception) => {
      #[cfg(feature = "tracing")]
      match &exception {
        // The `Exception` `Display` impl is safe: it checks
        // `isKindOfClass: NSException` internally and renders the
        // reason only when present, so logging it cannot itself raise.
        Some(exc) => tracing::warn!(
          detector,
          exception = %exc,
          "Apple Vision raised an Objective-C exception; skipping this detector and returning a \
           partial keyframe",
        ),
        None => tracing::warn!(
          detector,
          "Apple Vision raised a nil Objective-C exception; skipping this detector and returning \
           a partial keyframe",
        ),
      }
      #[cfg(not(feature = "tracing"))]
      let _ = (detector, exception);
      fallback
    }
  }
}

/// Compute the effective per-extractor cap as
/// `min(user_configured_max, MAX_VISION_RESULTS_PER_FRAME)`. Use
/// this for ALL of: `Vec::with_capacity(cap)`, `.iter().take(cap)`,
/// and the in-loop `if emitted.len() >= cap { break; }` guard.
/// Composing the three around the SAME `cap` value bounds both
/// capacity and emission to the hard ceiling, regardless of what
/// the caller configured.
#[cfg(target_vendor = "apple")]
#[inline]
fn effective_results_cap(user_max: usize) -> usize {
  user_max.min(MAX_VISION_RESULTS_PER_FRAME)
}

/// Validate a byte-length payload pre-`from_raw_parts`. Two
/// preconditions:
///
/// 1. `byte_len <= max_bytes` (caller-provided ceiling against
///    corrupted/adversarial sizes that would drive the bounded
///    allocator into refusal).
/// 2. `byte_len <= isize::MAX as usize` (the
///    [`std::slice::from_raw_parts`] contract for `T: u8`).
///
/// Returns `None` on either violation; the caller propagates that
/// `None` so the detection is dropped rather than triggering UB or
/// the allocator's abort path.
///
/// Currently exercised only by the unit-test suite — the last
/// in-engine caller (the FeaturePrint copy path) was removed when
/// the `feature_print` field migrated to LanceDB. The helper is
/// retained because future FFI byte-slice surfaces will need the
/// same precondition gate.
#[cfg(target_vendor = "apple")]
#[allow(dead_code)]
#[inline]
fn validate_raw_slice_bytes(byte_len: usize, max_bytes: usize) -> Option<()> {
  if byte_len > max_bytes {
    return None;
  }
  if byte_len > isize::MAX as usize {
    return None;
  }
  Some(())
}

/// Validate an element-count payload pre-`from_raw_parts<T>`. Same
/// shape as [`validate_raw_slice_bytes`] but computes
/// `byte_len = elem_count * size_of::<T>()` with overflow checking
/// before the `isize::MAX` comparison, so the helper is safe for
/// element types larger than `u8`.
#[cfg(target_vendor = "apple")]
#[inline]
fn validate_raw_slice_elems<T>(elem_count: usize, max_elems: usize) -> Option<()> {
  if elem_count > max_elems {
    return None;
  }
  let byte_len = elem_count.checked_mul(core::mem::size_of::<T>())?;
  if byte_len > isize::MAX as usize {
    return None;
  }
  Some(())
}

/// Allocate a zero-initialised packed mask buffer with bounded
/// `try_reserve_exact`. Returns `None` on either bound violation or
/// allocator failure — both surface to the caller as a dropped mask
/// detection rather than aborting the process.
#[cfg(target_vendor = "apple")]
fn try_alloc_packed_mask(packed_len: usize) -> Option<Vec<u8>> {
  if packed_len > MAX_MASK_BYTES {
    return None;
  }
  let mut packed: Vec<u8> = Vec::new();
  packed.try_reserve_exact(packed_len).ok()?;
  packed.resize(packed_len, 0u8);
  Some(packed)
}

/// Sanitise a raw face captureQuality reading from Vision.
///
/// `None` at every stage means the same thing to a consumer: no usable
/// quality reading for this observation. Vision omitting the
/// `NSNumber?` entirely (`raw = None`) and Vision reporting a
/// non-finite value (`NaN` / `±Inf`) both collapse to `None` here, and
/// neither is defaulted to `0.0` — that default is exactly the
/// collapse this function exists to stop making (see
/// `contract::FaceDetection`). `Some(v)` is a real, finite
/// measurement, including a genuine `Some(0.0)` — Vision measured this
/// capture and found it terrible, which is not the same claim as
/// never having measured it.
#[cfg(target_vendor = "apple")]
#[inline]
fn sanitize_capture_quality(raw: Option<f32>) -> Option<f32> {
  raw.and_then(finite_f32)
}

/// Minimum intersection-over-union for a face-rectangle detection and a
/// capture-quality observation to be treated as the SAME face when merging the
/// two Vision face passes (issue #48). Vision runs both passes on the identical
/// frame at the same revision, so a genuine match has near-1.0 IoU; 0.5 is a
/// forgiving floor that still rejects two distinct faces.
#[cfg(target_vendor = "apple")]
const FACE_MATCH_MIN_IOU: f32 = 0.5;

/// Intersection-over-union of two normalized bounding boxes. `0.0` when they do
/// not overlap, or when either box is degenerate (zero area).
#[cfg(target_vendor = "apple")]
fn bbox_iou<B: BoundingBox>(a: &B, b: &B) -> f32 {
  let ix = (a.x() + a.width()).min(b.x() + b.width()) - a.x().max(b.x());
  let iy = (a.y() + a.height()).min(b.y() + b.height()) - a.y().max(b.y());
  if ix <= 0.0 || iy <= 0.0 {
    return 0.0;
  }
  let inter = ix * iy;
  let union = a.width() * a.height() + b.width() * b.height() - inter;
  if union <= 0.0 { 0.0 } else { inter / union }
}

/// The capture quality to annotate a detected face `bbox` with: `Some` of
/// the best-overlapping (IoU ≥ [`FACE_MATCH_MIN_IOU`]) capture-quality
/// observation's quality, or `None` when no observation overlaps — the
/// face the capture-quality pass never covered ("unmatched"), not a face
/// measured at `0.0`. `scored` is the `(bbox, quality)` list from the
/// capture-quality pass.
#[cfg(target_vendor = "apple")]
fn matched_capture_quality<B: BoundingBox>(bbox: &B, scored: &[(B, f32)]) -> Option<f32> {
  let mut best_iou = FACE_MATCH_MIN_IOU;
  let mut best_quality = None;
  for (candidate, quality) in scored {
    let iou = bbox_iou(bbox, candidate);
    if iou >= best_iou {
      best_iou = iou;
      best_quality = Some(*quality);
    }
  }
  best_quality
}

/// Sanitise a raw 3-D body-pose height + height-estimation pair.
///
/// Vision's `bodyHeight()` is metres in model space. When the
/// reading is non-finite, both `body_height` AND `height_estimation`
/// must be neutralised together — substituting `0.0` for the height
/// while preserving a `Measured` or `Reference` enum would tell
/// consumers there is a known 0-metre subject. The pair
/// `(0.0, UNKNOWN)` is the truthful encoding of "no estimate
/// available" and the only consistent fallback.
#[cfg(target_vendor = "apple")]
#[inline]
fn sanitize_body_height_pair(
  raw_height: f32,
  measured_or_reference: HeightEstimation,
) -> (f32, HeightEstimation) {
  match finite_f32(raw_height) {
    Some(finite) => (finite, measured_or_reference),
    None => (0.0, HeightEstimation::Unknown),
  }
}

/// Validate mask dimensions BEFORE constructing the raw-parts slice
/// over a `CVPixelBuffer`'s base address. Two preconditions are
/// checked here so the unsafe `std::slice::from_raw_parts` call
/// downstream is sound even against a corrupted or adversarial
/// `CVPixelBuffer`:
///
/// 1. `width * height` (the output payload size after packing to
///    `OneComponent8`) must not exceed [`MAX_MASK_BYTES`].
/// 2. `total_src_len = bytes_per_row * height` (the raw slice
///    length) must fit in `isize::MAX`, which is the
///    [`std::slice::from_raw_parts`] contract.
///
/// Returns `None` on either violation; the caller propagates the
/// `None` so the mask detection is dropped rather than triggering
/// UB.
#[cfg(target_vendor = "apple")]
#[inline]
fn validate_mask_dims_for_slice(width: usize, height: usize, total_src_len: usize) -> Option<()> {
  let output_payload = width.checked_mul(height)?;
  if output_payload > MAX_MASK_BYTES {
    return None;
  }
  if total_src_len > isize::MAX as usize {
    return None;
  }
  Some(())
}

/// Project a face-bbox-relative landmark point into the image's
/// normalized coordinate space (Vision lower-left) using Apple's
/// documented convention: landmark points are normalized within the
/// face's normalized bounding box, NOT directly within the image.
/// `VNImagePointForFaceLandmarkPoint(p, faceBBox, w, h)` performs
/// `imageX = faceBBox.x + p.x * faceBBox.width;
/// imageY = faceBBox.y + p.y * faceBBox.height` (lower-left). Callers
/// then route through [`vision_point_to_normalized`] for the
/// top-left flip + `[0, 1]` clamp + finite check.
#[cfg(target_vendor = "apple")]
#[inline]
fn project_landmark_to_image(point: CGPoint, face_bbox_vision: CGRect) -> CGPoint {
  CGPoint {
    x: face_bbox_vision.origin.x + point.x * face_bbox_vision.size.width,
    y: face_bbox_vision.origin.y + point.y * face_bbox_vision.size.height,
  }
}

/// Derive an axis-aligned bounding box from the min/max of a pose's
/// surviving joint coordinates. Returns `None` when the extent in
/// either axis is zero — a single joint, or joints that are perfectly
/// colinear horizontally/vertically, would otherwise produce a wire
/// box that a validating [`BoundingBox::try_new`] rejects.
/// Callers should skip the pose detection on `None`; the joints alone
/// do not carry enough geometry to construct a valid box.
#[cfg(target_vendor = "apple")]
fn pose_bbox_from_joint_bounds<B: BoundingBox>(
  min_x: f32,
  min_y: f32,
  max_x: f32,
  max_y: f32,
) -> Option<B> {
  if !(min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()) {
    return None;
  }
  let width = max_x - min_x;
  let height = max_y - min_y;
  if width <= 0.0 || height <= 0.0 {
    return None;
  }
  // Joints are sanitised individually upstream (each goes through
  // `vision_point_to_normalized` which clamps to `[0, 1]`), so the
  // derived bbox should satisfy the contract invariants. Drop on the
  // off-chance the vocabulary rejects.
  B::try_new(min_x, min_y, width, height).ok()
}

/// Validate a raw Vision `confidence` value against the configured
/// per-request minimum and the wire/domain `Confidence` invariant
/// (finite, in `[0.0, 1.0]`). Returns `None` if the value is
/// non-finite, outside `[0, 1]`, or below `min` — the caller drops
/// the detection in that case. A simple `value < min` threshold
/// previously let `NaN` through (since every NaN comparison is
/// false) and accepted `>1.0` values, both of which a validating
/// vocabulary rejects.
#[cfg(target_vendor = "apple")]
#[inline]
fn sanitize_confidence(value: f32, min: f32) -> Option<f32> {
  if value.is_finite() && (0.0..=1.0).contains(&value) && value >= min {
    Some(value)
  } else {
    None
  }
}

// ----- CVPixelBuffer RAII lock ----------------------------------------------

/// RAII guard that holds a `CVPixelBufferLockBaseAddress` lock for the
/// lifetime of the guard. `Drop` unlocks even on panic-unwind so the
/// buffer cannot be left in a locked state by a panicking slice index.
#[cfg(target_vendor = "apple")]
struct CVPixelBufferLockGuard<'a> {
  buffer: &'a CVPixelBuffer,
  flags: CVPixelBufferLockFlags,
}

#[cfg(target_vendor = "apple")]
impl<'a> CVPixelBufferLockGuard<'a> {
  /// Acquire a lock on `buffer` with `flags`. Returns `None` if Core
  /// Video refused the lock; on success the guard's `Drop` is
  /// responsible for releasing it.
  #[inline]
  fn lock(buffer: &'a CVPixelBuffer, flags: CVPixelBufferLockFlags) -> Option<Self> {
    // SAFETY: `buffer` is a valid `CVPixelBuffer`; `flags` is a valid
    // `CVPixelBufferLockFlags`. The function is documented as safe to
    // call from any thread.
    let rc = unsafe { CVPixelBufferLockBaseAddress(buffer, flags) };
    if rc == kCVReturnSuccess {
      Some(Self { buffer, flags })
    } else {
      None
    }
  }

  /// Borrow the locked buffer.
  #[inline]
  fn buffer(&self) -> &CVPixelBuffer {
    self.buffer
  }
}

#[cfg(target_vendor = "apple")]
impl Drop for CVPixelBufferLockGuard<'_> {
  fn drop(&mut self) {
    // SAFETY: the corresponding lock was acquired successfully in
    // `lock`; calling unlock with matching flags is required by Core
    // Video. We ignore the return code — even if unlock fails, the
    // buffer is going away with us and there's nothing the caller can
    // do about it.
    let _ = unsafe { CVPixelBufferUnlockBaseAddress(self.buffer, self.flags) };
  }
}

/// Apple Vision analyzer — one per worker thread.
///
/// Construct one [`VisionAnalyzer`] per worker thread via
/// [`VisionAnalyzer::new`]. The analyzer owns retained `VNRequest`
/// Objective-C objects that carry per-call state across
/// `performRequests` / `results()`, so they are *not* safe to share
/// across threads or clone. Construct one fresh analyzer per worker
/// rather than cloning a single shared instance — `Clone` is
/// intentionally not implemented to make that contract a compile-time
/// error.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct VisionAnalyzer {
  requests: VisionRequests,
}

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
struct VisionRequests {
  classify: Retained<VNClassifyImageRequest>,
  face_rectangles: Retained<VNDetectFaceRectanglesRequest>,
  face_landmarks: Retained<VNDetectFaceLandmarksRequest>,
  face_quality: Retained<VNDetectFaceCaptureQualityRequest>,
  human_rectangles: Retained<VNDetectHumanRectanglesRequest>,
  body_pose: Retained<VNDetectHumanBodyPoseRequest>,
  body_pose_3d: Retained<VNDetectHumanBodyPose3DRequest>,
  hand_pose: Retained<VNDetectHumanHandPoseRequest>,
  animals: Retained<VNRecognizeAnimalsRequest>,
  animal_body_pose: Retained<VNDetectAnimalBodyPoseRequest>,
  person_instance_mask: Retained<VNGeneratePersonInstanceMaskRequest>,
  person_segmentation: Retained<VNGeneratePersonSegmentationRequest>,
  text: Retained<VNRecognizeTextRequest>,
  barcodes: Retained<VNDetectBarcodesRequest>,
  attention_saliency: Retained<VNGenerateAttentionBasedSaliencyImageRequest>,
  objectness_saliency: Retained<VNGenerateObjectnessBasedSaliencyImageRequest>,
  horizon: Retained<VNDetectHorizonRequest>,
  document_segments: Retained<VNDetectDocumentSegmentationRequest>,
  aesthetics: Retained<VNCalculateImageAestheticsScoresRequest>,
}

#[cfg(target_vendor = "apple")]
impl VisionRequests {
  fn new(opts: &AnalyzeOptions) -> Self {
    unsafe {
      let classify = VNClassifyImageRequest::new();
      classify.setRevision(VNClassifyImageRequestRevision2);

      let face_rectangles = VNDetectFaceRectanglesRequest::new();
      face_rectangles.setRevision(VNDetectFaceRectanglesRequestRevision3);

      let face_landmarks = VNDetectFaceLandmarksRequest::new();
      face_landmarks.setRevision(VNDetectFaceLandmarksRequestRevision3);

      let face_quality = VNDetectFaceCaptureQualityRequest::new();
      face_quality.setRevision(VNDetectFaceCaptureQualityRequestRevision3);

      let human_rectangles = VNDetectHumanRectanglesRequest::new();
      human_rectangles.setUpperBodyOnly(false);
      human_rectangles.setRevision(VNDetectHumanRectanglesRequestRevision2);

      let body_pose = VNDetectHumanBodyPoseRequest::new();
      body_pose.setRevision(VNDetectHumanBodyPoseRequestRevision1);

      let body_pose_3d = VNDetectHumanBodyPose3DRequest::new();
      body_pose_3d.setRevision(VNDetectHumanBodyPose3DRequestRevision1);

      let hand_pose = VNDetectHumanHandPoseRequest::new();
      // Clamp the user-configured hand count to Apple's documented
      // revision-1 maximum. Apple's request becomes invalid above 6
      // and would surface as an Objective-C exception crossing the
      // FFI boundary; clamp here so a stale/misconfigured option
      // value still produces a usable Vision request (codex R17 F2).
      let user_hand_count = opts.hand_pose().maximum_hand_count();
      hand_pose.setMaximumHandCount(user_hand_count.min(MAX_HAND_POSE_MAXIMUM_HAND_COUNT));
      hand_pose.setRevision(VNDetectHumanHandPoseRequestRevision1);

      let animals = VNRecognizeAnimalsRequest::new();
      animals.setRevision(VNRecognizeAnimalsRequestRevision2);

      let animal_body_pose = VNDetectAnimalBodyPoseRequest::new();
      animal_body_pose.setRevision(VNDetectAnimalBodyPoseRequestRevision1);

      let person_instance_mask = VNGeneratePersonInstanceMaskRequest::new();
      person_instance_mask.setRevision(VNGeneratePersonInstanceMaskRequestRevision1);

      let person_segmentation = VNGeneratePersonSegmentationRequest::new();
      person_segmentation.setRevision(VNGeneratePersonSegmentationRequestRevision1);

      let text = VNRecognizeTextRequest::new();
      text.setRevision(VNRecognizeTextRequestRevision3);

      let barcodes = VNDetectBarcodesRequest::new();
      barcodes.setRevision(VNDetectBarcodesRequestRevision4);

      let attention_saliency = VNGenerateAttentionBasedSaliencyImageRequest::new();
      attention_saliency.setRevision(VNGenerateAttentionBasedSaliencyImageRequestRevision2);

      let objectness_saliency = VNGenerateObjectnessBasedSaliencyImageRequest::new();
      objectness_saliency.setRevision(VNGenerateObjectnessBasedSaliencyImageRequestRevision2);

      let horizon = VNDetectHorizonRequest::new();
      horizon.setRevision(VNDetectHorizonRequestRevision1);

      let document_segments = VNDetectDocumentSegmentationRequest::new();
      document_segments.setRevision(VNDetectDocumentSegmentationRequestRevision1);

      let aesthetics = VNCalculateImageAestheticsScoresRequest::new();
      aesthetics.setRevision(VNCalculateImageAestheticsScoresRequestRevision1);

      Self {
        classify,
        face_rectangles,
        face_landmarks,
        face_quality,
        human_rectangles: { human_rectangles },
        body_pose,
        body_pose_3d,
        hand_pose: { hand_pose },
        animals,
        animal_body_pose,
        person_instance_mask,
        person_segmentation,
        text,
        barcodes,
        attention_saliency,
        objectness_saliency,
        horizon,
        document_segments,
        aesthetics,
      }
    }
  }

  fn perform(&self, handler: &VNSequenceRequestHandler, data: &NSData) -> Result<(), AnalyzeError> {
    unsafe {
      let requests = NSArray::from_retained_slice(&[
        Retained::cast_unchecked::<VNRequest>(self.classify.clone()),
        Retained::cast_unchecked::<VNRequest>(self.face_rectangles.clone()),
        Retained::cast_unchecked::<VNRequest>(self.face_landmarks.clone()),
        Retained::cast_unchecked::<VNRequest>(self.face_quality.clone()),
        Retained::cast_unchecked::<VNRequest>(self.human_rectangles.clone()),
        Retained::cast_unchecked::<VNRequest>(self.body_pose.clone()),
        Retained::cast_unchecked::<VNRequest>(self.body_pose_3d.clone()),
        Retained::cast_unchecked::<VNRequest>(self.hand_pose.clone()),
        Retained::cast_unchecked::<VNRequest>(self.animals.clone()),
        Retained::cast_unchecked::<VNRequest>(self.animal_body_pose.clone()),
        Retained::cast_unchecked::<VNRequest>(self.person_instance_mask.clone()),
        Retained::cast_unchecked::<VNRequest>(self.person_segmentation.clone()),
        Retained::cast_unchecked::<VNRequest>(self.text.clone()),
        Retained::cast_unchecked::<VNRequest>(self.barcodes.clone()),
        Retained::cast_unchecked::<VNRequest>(self.attention_saliency.clone()),
        Retained::cast_unchecked::<VNRequest>(self.objectness_saliency.clone()),
        Retained::cast_unchecked::<VNRequest>(self.horizon.clone()),
        Retained::cast_unchecked::<VNRequest>(self.document_segments.clone()),
        Retained::cast_unchecked::<VNRequest>(self.aesthetics.clone()),
      ]);

      // The batched `performRequests:` runs ~19 detectors in one call.
      // A detector can fail two ways: a recoverable `NSError` (mapped
      // below), or a fatal Objective-C `NSException` that unwinds across
      // the FFI boundary and would abort the whole process. Guard the
      // perform with `objc2::exception::catch` so a raising detector
      // degrades to "no perform" — the per-detector `results()` pulls in
      // `analyze_keyframe` then return empty for the detectors that did
      // not complete, yielding a partial keyframe instead of an abort.
      // On a caught exception the captured Vision handles are discarded
      // and extraction proceeds, so no broken state is observed.
      guard_vision_ffi("performRequests", Ok(()), || {
        handler
          .performRequests_onImageData_error(&requests, data)
          .map_err(|e| {
            // Route NSError's localizedDescription through the bounded
            // FFI-string helper so a pathological error message cannot
            // drive the allocator into the abort path while the worker
            // is already trying to report a failure.
            let raw = e.localizedDescription();
            let message = ffi_nsstring_to_smolstr(&raw)
              .map(|m| Cow::Owned(String::from(m)))
              .unwrap_or(Cow::Borrowed(
                "apple-vision request failed (description elided)",
              ));
            AnalyzeError::new(AnalyzeErrorKind::RequestFailed, message)
          })
      })
    }
  }
}

#[cfg(target_vendor = "apple")]
impl VisionAnalyzer {
  /// Creates an analyzer with `options` baked into its retained
  /// Vision requests.
  ///
  /// Only
  /// [`maximum_hand_count`](AppleVisionHandPoseOptions::maximum_hand_count)
  /// is consumed here — Apple bakes that one into the request object
  /// itself. Every other knob is read per call by
  /// [`analyze_keyframe`](VisionAnalyzer::analyze_keyframe), so the
  /// analyzer carries no configuration of its own.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(options: &AnalyzeOptions) -> Self {
    Self {
      requests: VisionRequests::new(options),
    }
  }

  /// Logs the pinned revision of every Vision request this analyzer
  /// holds.
  ///
  /// Revisions are fixed at construction and a drift changes detection
  /// semantics **silently** — same API, different numbers. This is the
  /// diagnostic that makes the drift visible; wrap the call in your own
  /// span if you need to attribute it to a worker.
  #[cfg(feature = "tracing")]
  pub fn log_request_revisions(&self) {
    unsafe {
      tracing::info!(
        classify_rev = self.requests.classify.revision(),
        face_rectangles_rev = self.requests.face_rectangles.revision(),
        face_landmarks_rev = self.requests.face_landmarks.revision(),
        face_quality_rev = self.requests.face_quality.revision(),
        human_rectangles_rev = self.requests.human_rectangles.revision(),
        body_pose_rev = self.requests.body_pose.revision(),
        body_pose_3d_rev = self.requests.body_pose_3d.revision(),
        hand_pose_rev = self.requests.hand_pose.revision(),
        animals_rev = self.requests.animals.revision(),
        animal_body_pose_rev = self.requests.animal_body_pose.revision(),
        person_instance_mask_rev = self.requests.person_instance_mask.revision(),
        person_segmentation_rev = self.requests.person_segmentation.revision(),
        text_rev = self.requests.text.revision(),
        barcodes_rev = self.requests.barcodes.revision(),
        attention_saliency_rev = self.requests.attention_saliency.revision(),
        objectness_saliency_rev = self.requests.objectness_saliency.revision(),
        horizon_rev = self.requests.horizon.revision(),
        document_segments_rev = self.requests.document_segments.revision(),
        aesthetics_rev = self.requests.aesthetics.revision(),
        "initialized pinned Apple Vision request revisions"
      );
    }
  }

  /// Runs every Apple Vision request against `jpeg_data` and gathers
  /// the detections into an [`Analysis`].
  ///
  /// `D` names the output vocabulary and `options` supplies every
  /// per-detector gate. Nothing identifies the frame — no ids, no
  /// timestamp, no dimensions — because the engine has no such
  /// knowledge; composing this analysis into whatever record you store
  /// is the caller's job.
  ///
  /// Degradation is per detector: one Vision request that raises an
  /// Objective-C exception contributes an empty slot while the others
  /// still land. Individual detections are filtered before
  /// construction and refused ones are silently absent — there is no
  /// "dropped" counter. An `Err` therefore means no analysis happened
  /// at all.
  ///
  /// `options` is read per call, but
  /// [`VisionAnalyzer::new`] already baked
  /// [`maximum_hand_count`](AppleVisionHandPoseOptions::maximum_hand_count)
  /// into the retained hand-pose request: that one knob follows the
  /// analyzer, not the call.
  pub fn analyze_keyframe<D: Detections>(
    &self,
    jpeg_data: &[u8],
    options: &AnalyzeOptions,
  ) -> Result<Analysis<D>, AnalyzeError> {
    // Input-byte budget — refuse oversized payloads BEFORE
    // Foundation copies them into an NSData and doubles peak
    // memory. Surface as a structured error so the orchestrator
    // can decide whether to retry, log, or escalate (codex R17 F1).
    if jpeg_data.len() > MAX_INPUT_IMAGE_BYTES {
      return Err(AnalyzeError::new(
        AnalyzeErrorKind::RequestFailed,
        "input image exceeds MAX_INPUT_IMAGE_BYTES",
      ));
    }
    objc2::rc::autoreleasepool(|_| {
      let ns_data = NSData::with_bytes(jpeg_data);
      let handler = unsafe { VNSequenceRequestHandler::new() };
      self.requests.perform(&handler, &ns_data)?;

      // Per-detector Objective-C exception barrier. The batched
      // `perform` above runs every detector, but result extraction
      // re-enters Vision FFI per detector (`results()` + per-observation
      // accessors). Any one of those can raise an `NSException` that
      // `catch_unwind` cannot catch — so each `extract_*` is wrapped in
      // `objc2::exception::catch` (via `guard_vision_ffi`). A raising
      // detector contributes its empty fallback and the OTHER detectors'
      // results still land: the analysis degrades per detector, never
      // aborting the process.
      //
      // Shared mask budgets — both `extract_person_instance_masks`
      // and `extract_person_segmentation_masks` charge against the
      // SAME per-frame totals (count, bytes, AND attempts) so the
      // cap is enforced across both surfaces and across both the
      // success and failure paths (codex R14 F1 + R15). The mask
      // closures capture these `&mut` counters, hence `AssertUnwindSafe`:
      // a caught exception leaves a counter at its partial (over-counted,
      // never under-counted) value, the safe direction for a cap.
      let mut mask_total_bytes: usize = 0;
      let mut mask_total_count: usize = 0;
      let mut mask_total_attempts: usize = 0;
      let instance_masks = guard_vision_ffi("person_instance_mask", Vec::new(), || {
        self.extract_person_instance_masks::<D>(
          options,
          &mut mask_total_bytes,
          &mut mask_total_count,
          &mut mask_total_attempts,
        )
      });
      let segmentation_masks = guard_vision_ffi("person_segmentation", Vec::new(), || {
        self.extract_person_segmentation_masks::<D>(
          options,
          &mut mask_total_bytes,
          &mut mask_total_count,
          &mut mask_total_attempts,
        )
      });

      let mut analysis: Analysis<D> = Analysis::new();
      analysis
        .set_classifications(guard_vision_ffi("classify", Vec::new(), || {
          self.extract_classifications::<D>(options)
        }))
        .set_human_subjects(guard_vision_ffi("human_rectangles", Vec::new(), || {
          self.extract_human_subjects::<D>(options)
        }))
        // #48: ONE record per face — the rectangles pass is the detection
        // spine, annotated with capture quality by bbox overlap (see
        // `extract_faces`). The capture-quality pass contributes no
        // records of its own, so no face is counted twice.
        .set_faces(guard_vision_ffi("faces", Vec::new(), || {
          self.extract_faces::<D>(options)
        }))
        .set_face_landmarks(guard_vision_ffi("face_landmarks", Vec::new(), || {
          self.extract_face_landmarks::<D>(options)
        }))
        .set_body_poses(guard_vision_ffi("body_pose", Vec::new(), || {
          self.extract_body_poses::<D>(options)
        }))
        .set_hand_poses(guard_vision_ffi("hand_pose", Vec::new(), || {
          self.extract_hand_poses::<D>(options)
        }))
        // `extract_body_poses_3d` self-guards (inner
        // `objc2::exception::catch` under its `catch_unwind`); a
        // call-site guard here would put `catch_unwind` inside the
        // ObjC barrier and could not catch the foreign exception.
        .set_body_poses_3d(self.extract_body_poses_3d::<D>(options))
        .set_person_instance_masks(instance_masks)
        .set_person_segmentation_masks(segmentation_masks)
        .set_animal_subjects(guard_vision_ffi("animals", Vec::new(), || {
          self.extract_animal_subjects::<D>(options)
        }))
        .set_animal_body_poses(guard_vision_ffi("animal_body_pose", Vec::new(), || {
          self.extract_animal_body_poses::<D>(options)
        }))
        .set_text_detections(guard_vision_ffi("text", Vec::new(), || {
          self.extract_text_detections::<D>(options)
        }))
        .set_barcodes(guard_vision_ffi("barcodes", Vec::new(), || {
          self.extract_barcodes::<D>(options)
        }))
        .set_attention_saliency(guard_vision_ffi("attention_saliency", Vec::new(), || {
          self.extract_attention_saliency::<D>(options)
        }))
        .set_objectness_saliency(guard_vision_ffi("objectness_saliency", Vec::new(), || {
          self.extract_objectness_saliency::<D>(options)
        }))
        .set_horizon(guard_vision_ffi(
          "horizon",
          // The "no detection" sentinel — matches `extract_horizon`'s
          // own `empty`. `None` only if the vocabulary refuses even
          // that, which costs the slot rather than the frame.
          D::HorizonInfo::try_new(0.0, 0.0).ok(),
          || self.extract_horizon::<D>(options),
        ))
        .set_document_segments(guard_vision_ffi("document_segments", Vec::new(), || {
          self.extract_document_segments::<D>(options)
        }))
        .set_aesthetics(Some(guard_vision_ffi(
          "aesthetics",
          // The "no detection" sentinel — matches `extract_aesthetics`.
          D::Aesthetics::new(0.0, false),
          || self.extract_aesthetics::<D>(options),
        )));
      Ok(analysis)
    })
  }

  fn extract_classifications<D: Detections>(&self, options: &AnalyzeOptions) -> Vec<D::Detection> {
    let opts = options.classifications();
    let Some(results) = (unsafe { self.requests.classify.results() }) else {
      return Vec::new();
    };

    // Effective cap composes user-configured + hard ceiling so
    // with_capacity, take, and the emission guard all bound to
    // the SAME value (no `Vec::push` reallocation past the cap).
    let cap = effective_results_cap(opts.max_results());
    let mut tags = Vec::with_capacity(cap);
    for obs in results.iter().take(cap) {
      if tags.len() >= cap {
        break;
      }
      let Some(confidence) =
        sanitize_confidence(unsafe { obs.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      let identifier = unsafe { obs.identifier() };
      let Some(label) = ffi_nsstring_to_smolstr(&identifier) else {
        continue;
      };
      let label = normalize_classification_label(label);
      if !label.is_empty()
        && let Ok(detection) = D::Detection::try_new(&label, confidence)
      {
        tags.push(detection);
      }
    }

    tags
  }

  /// The detected faces of the frame, ONE record per face (issue #48).
  ///
  /// The face-rectangles pass is the detection spine — every detected face
  /// appears exactly once — and each face is annotated with the capture quality
  /// of its best bounding-box-overlapping capture-quality observation, or
  /// `None` when the capture-quality pass did not cover it (#20: a nil Vision
  /// reading and a join-miss both reach the contract seat as `None`, never as a
  /// `Some(0.0)` that would misrepresent "never measured" as "measured and
  /// terrible"). `min_capture_quality` then drops faces whose quality compares
  /// below it, treating an unmeasured face as `0.0` for THIS comparison only:
  /// the default (0.1) still drops quality-scored-low and unmeasured faces
  /// alike, while `min_capture_quality == 0.0` keeps every detected face. A
  /// face that passes unmeasured still carries `None`, not `Some(0.0)`, to
  /// `D::FaceDetection::try_new`.
  ///
  /// This replaces the previous TWO collections (`faces` from the capture-quality
  /// pass PLUS `face_rectangles` from the rectangles pass), which persisted a
  /// duplicate `keyframe_face` row per face under two `kind`s with identical
  /// bounding boxes and inflated face counts.
  fn extract_faces<D: Detections>(&self, options: &AnalyzeOptions) -> Vec<D::FaceDetection> {
    let Some(rect_results) = (unsafe { self.requests.face_rectangles.results() }) else {
      return Vec::new();
    };
    let rect_opts = options.face_rectangles();
    let capture_opts = options.face_capture();

    // The capture-quality pass as `(bbox, quality)` pairs, to annotate the
    // detection spine by bounding-box overlap. Built once. Non-finite quality
    // readings drop the pair (see `sanitize_capture_quality`); degenerate bboxes
    // are skipped.
    let scored: Vec<(D::BoundingBox, f32)> = match unsafe { self.requests.face_quality.results() } {
      Some(results) => results
        .iter()
        .take(MAX_VISION_RESULTS_PER_FRAME)
        .filter_map(|obs| {
          let quality =
            sanitize_capture_quality(unsafe { obs.faceCaptureQuality() }.map(|q| q.floatValue()))?;
          let bbox = vision_rect_to_bbox(unsafe { obs.boundingBox() }.standardize())?;
          Some((bbox, quality))
        })
        .collect(),
      None => Vec::new(),
    };

    let mut faces = Vec::with_capacity(rect_results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    for obs in rect_results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      let Some(confidence) =
        sanitize_confidence(unsafe { obs.confidence() }, rect_opts.min_confidence())
      else {
        continue;
      };
      let Some(bbox) = vision_rect_to_bbox(unsafe { obs.boundingBox() }.standardize()) else {
        continue;
      };
      let capture_quality = matched_capture_quality(&bbox, &scored);
      // The threshold gate is unchanged: an unmeasured (`None`) face
      // compares as `0.0` here, so it fails any positive minimum
      // exactly as a matched-and-zero face would (#20's fail-closed
      // precedent, carried over from the old `sanitize_capture_quality`
      // collapse). Only the COMPARISON substitutes `0.0` — the `Option`
      // itself, not this local, is what reaches `try_new` below, so a
      // face that clears a `min_capture_quality == 0.0` gate while
      // unmeasured still arrives at the contract seat as `None`.
      if capture_quality.unwrap_or(0.0) < capture_opts.min_capture_quality() {
        continue;
      }
      // `None` at every stage means the same thing to a consumer: no
      // usable angle. Vision omitting the `NSNumber?` entirely and
      // Vision reporting a non-finite reading both collapse to `None`
      // here, and neither is defaulted to `0.0` — that default is
      // exactly the collapse this seat exists to stop making (see
      // `contract::FaceDetection`).
      let roll = unsafe { obs.roll() }
        .map(|v| v.floatValue())
        .and_then(finite_f32);
      let yaw = unsafe { obs.yaw() }
        .map(|v| v.floatValue())
        .and_then(finite_f32);
      let pitch = unsafe { obs.pitch() }
        .map(|v| v.floatValue())
        .and_then(finite_f32);
      if let Ok(face) =
        D::FaceDetection::try_new(bbox, confidence, capture_quality, roll, yaw, pitch)
      {
        faces.push(face);
      }
    }

    faces
  }

  fn extract_face_landmarks<D: Detections>(
    &self,
    options: &AnalyzeOptions,
  ) -> Vec<D::FaceLandmarksDetection> {
    let Some(results) = (unsafe { self.requests.face_landmarks.results() }) else {
      return Vec::new();
    };
    let opts = options.face_landmarks();

    let mut detections = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    // Per-frame budgets:
    // - `total_points_remaining` — emission budget; tentative-
    //   committed (decremented in a shadow during region extraction
    //   and applied to the master only when the detection survives
    //   every gate).
    // - `total_landmark_attempts` — attempt budget; immediately
    //   committed every time `push_face_landmark_region` visits a
    //   region's slice, regardless of whether the parent detection
    //   ultimately survives. Bounds the FAILURE-PATH work that the
    //   emission budget alone could not catch (codex R17 F1, same
    //   policy as `MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME`).
    let mut total_points_remaining: usize = MAX_FACE_LANDMARK_POINTS_PER_FRAME;
    let mut total_landmark_attempts: usize = 0;
    for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      if total_points_remaining == 0
        || total_landmark_attempts >= MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME
      {
        break;
      }
      let Some(landmarks) = (unsafe { obs.landmarks() }) else {
        continue;
      };
      let Some(confidence) =
        sanitize_confidence(unsafe { landmarks.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      // Capture the face's Vision-coordinate bbox BEFORE the
      // schema-side flip+clamp so we can project landmark points
      // through it. Vision returns landmark points normalized to the
      // face bbox (not the image), per
      // `VNImagePointForFaceLandmarkPoint(p, faceBBox, w, h)` =
      // `(faceBBox.x + p.x * faceBBox.width,
      //   faceBBox.y + p.y * faceBBox.height)`.
      let face_rect_vision = unsafe { obs.boundingBox() }.standardize();
      // Validate the face bbox BEFORE walking landmarks so an
      // obviously-invalid observation does not spend the landmark
      // attempt budget (codex R17 F1 sub-rec). Re-using the same
      // schema-conversion guard the post-extraction commit uses.
      if vision_rect_to_bbox::<D::BoundingBox>(face_rect_vision).is_none() {
        continue;
      }

      // Tentative emission budget — commit point-budget consumption
      // ONLY after the detection survives every gate. Attempt budget
      // is committed immediately by the helper.
      let mut tentative_remaining = total_points_remaining;
      let regions = extract_face_landmark_regions::<D::FaceLandmarkRegion>(
        &landmarks,
        face_rect_vision,
        &mut tentative_remaining,
        &mut total_landmark_attempts,
      );
      if regions.len() < opts.min_region_count() {
        continue;
      }

      let Some(bbox) = vision_rect_to_bbox(face_rect_vision) else {
        continue;
      };
      let Ok(detection) = D::FaceLandmarksDetection::try_new(bbox, confidence, regions) else {
        // The confidence has already been sanitised; a failure here
        // is the vocabulary refusing a value the engine considers
        // valid — skip rather than abort the frame.
        continue;
      };
      // Commit the budget — the detection is being pushed.
      total_points_remaining = tentative_remaining;
      detections.push(detection);
    }

    detections
  }

  fn extract_human_subjects<D: Detections>(
    &self,
    options: &AnalyzeOptions,
  ) -> Vec<D::SubjectDetection> {
    let Some(results) = (unsafe { self.requests.human_rectangles.results() }) else {
      return Vec::new();
    };
    let opts = options.human_subjects();

    let mut humans = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      let Some(confidence) =
        sanitize_confidence(unsafe { obs.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      let Some(bbox) = vision_rect_to_bbox(unsafe { obs.boundingBox() }.standardize()) else {
        continue;
      };
      let Ok(detection) = D::Detection::try_new("person", confidence) else {
        continue;
      };
      humans.push(D::SubjectDetection::new(detection, bbox));
    }

    humans
  }

  fn extract_body_poses<D: Detections>(
    &self,
    options: &AnalyzeOptions,
  ) -> Vec<D::BodyPoseDetection> {
    let Some(results) = (unsafe { self.requests.body_pose.results() }) else {
      return Vec::new();
    };

    let mut body_poses = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      let Ok(points_by_joint) = (unsafe {
        obs.recognizedPointsForJointsGroupName_error(VNHumanBodyPoseObservationJointsGroupNameAll)
      }) else {
        continue;
      };

      // Bound the dictionary materialisation: `to_vecs()` allocates
      // two `Vec`s sized to `points_by_joint.len()`, which is FFI-
      // reported. Drop the pose entirely if the joint count
      // exceeds `MAX_POSE_JOINTS` rather than risking an allocator
      // abort on a corrupted/adversarial Vision dictionary.
      if points_by_joint.len() > MAX_POSE_JOINTS {
        continue;
      }
      let (joint_names, points) = points_by_joint.to_vecs();
      let mut joints = Vec::with_capacity(points.len());
      let mut min_x = f32::INFINITY;
      let mut min_y = f32::INFINITY;
      let mut max_x = f32::NEG_INFINITY;
      let mut max_y = f32::NEG_INFINITY;

      for (joint_name, point) in joint_names.into_iter().zip(points) {
        let Some(name) = ffi_nsstring_to_smolstr(&joint_name) else {
          continue;
        };
        if name.is_empty() {
          continue;
        }

        // Vision normalized points are lower-left origin; flip y for the
        // top-left schema convention before recording the joint or
        // deriving the bbox. A non-finite raw coordinate is dropped at
        // the source — partial-joint lists are valid for body pose so
        // we skip just this joint, not the whole pose.
        let Some((x, y)) = vision_point_to_normalized(unsafe { point.x() }, unsafe { point.y() })
        else {
          continue;
        };
        let Some(confidence) = sanitize_confidence(
          unsafe { point.confidence() },
          options.body_pose().min_joint_confidence(),
        ) else {
          continue;
        };

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);

        let Ok(joint) = D::BodyJoint::try_new(&name, x, y, confidence) else {
          continue;
        };
        joints.push(joint);
      }

      if joints.is_empty() {
        continue;
      }

      let Some(bbox) = pose_bbox_from_joint_bounds(min_x, min_y, max_x, max_y) else {
        // A pose with only one surviving joint (or perfectly colinear
        // joints) cannot produce a valid axis-aligned bbox; skip it
        // rather than emit a zero-extent box that the domain
        // validator would reject.
        continue;
      };
      // Observation confidence carries the per-pose score; sanitise it
      // against the same `[0, 1]` invariant. A non-finite observation
      // confidence cannot be emitted faithfully — drop the pose.
      let Some(pose_confidence) = sanitize_confidence(unsafe { obs.confidence() }, 0.0) else {
        continue;
      };

      joints.sort_by(|lhs, rhs| lhs.name().cmp(rhs.name()));
      if let Ok(pose) = D::BodyPoseDetection::try_new(bbox, pose_confidence, joints) {
        body_poses.push(pose);
      }
    }

    body_poses
  }

  fn extract_body_poses_3d<D: Detections>(
    &self,
    options: &AnalyzeOptions,
  ) -> Vec<D::BodyPose3DDetection> {
    // Two nested barriers, innermost FIRST. The `VNHumanBodyRecognizedPoint3D`
    // `position`/`confidence` msg_sends below have a `simd_float4x4` return
    // encoding that objc2 rejects: in a DEBUG build objc2's runtime
    // verification turns that into a *Rust panic* (caught by the outer
    // `catch_unwind`), but in a RELEASE build verification is compiled out and
    // the real Objective-C dispatch raises an `NSException` instead. That
    // foreign exception MUST be caught by the inner `objc2::exception::catch`
    // (`guard_vision_ffi`) — if it reached the outer `catch_unwind` it would
    // abort the process (`catch_unwind` cannot catch foreign exceptions). So
    // the ObjC barrier is innermost; the Rust-panic barrier wraps it.
    catch_unwind(AssertUnwindSafe(|| {
      guard_vision_ffi("body_pose_3d", Vec::new(), || {
        let Some(results) = (unsafe { self.requests.body_pose_3d.results() }) else {
          return Vec::new();
        };
        let Some(group_name) = (unsafe { VNHumanBodyPose3DObservationJointsGroupNameAll }) else {
          return Vec::new();
        };

        let mut body_poses = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
        for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
          let Ok(points_by_joint) =
            (unsafe { obs.recognizedPointsForJointsGroupName_error(group_name) })
          else {
            continue;
          };

          // Bound the dictionary materialisation: `to_vecs()` allocates
          // two `Vec`s sized to `points_by_joint.len()`, which is FFI-
          // reported. Drop the pose entirely if the joint count
          // exceeds `MAX_POSE_JOINTS` rather than risking an allocator
          // abort on a corrupted/adversarial Vision dictionary.
          if points_by_joint.len() > MAX_POSE_JOINTS {
            continue;
          }
          let (joint_names, points) = points_by_joint.to_vecs();
          let mut joints = Vec::with_capacity(points.len());

          for (joint_name, point) in joint_names.into_iter().zip(points) {
            let Some(name) = ffi_nsstring_to_smolstr(&joint_name) else {
              continue;
            };
            if name.is_empty() {
              continue;
            }

            let Some((x, y, z)) = extract_body_pose_3d_coordinates(&point) else {
              continue;
            };
            let raw_confidence: f32 = unsafe { objc2::msg_send![&*point, confidence] };
            let Some(confidence) = sanitize_confidence(
              raw_confidence,
              options.body_pose_3d().min_joint_confidence(),
            ) else {
              continue;
            };

            let Ok(joint) = D::Body3Joint::try_new(&name, x, y, z, confidence) else {
              continue;
            };
            joints.push(joint);
          }

          if joints.is_empty() {
            continue;
          }
          let Some(pose_confidence) = sanitize_confidence(unsafe { obs.confidence() }, 0.0) else {
            continue;
          };

          joints.sort_by(|lhs, rhs| lhs.name().cmp(rhs.name()));
          // See `sanitize_body_height_pair` — couples the
          // body_height substitution with the height_estimation enum
          // so `(0.0, UNKNOWN)` is the only fallback for non-finite
          // readings.
          let mapped_estimation =
            map_body_pose_3d_height_estimation(unsafe { obs.heightEstimation() });
          let (body_height, height_estimation) =
            sanitize_body_height_pair(unsafe { obs.bodyHeight() }, mapped_estimation);
          if let Ok(pose) =
            D::BodyPose3DDetection::try_new(pose_confidence, body_height, height_estimation, joints)
          {
            body_poses.push(pose);
          }
        }

        body_poses
      })
    }))
    .unwrap_or_else(|_| {
      #[cfg(feature = "tracing")]
      tracing::warn!("caught panic while extracting human body pose 3D; returning empty result");
      Vec::new()
    })
  }

  fn extract_hand_poses<D: Detections>(
    &self,
    options: &AnalyzeOptions,
  ) -> Vec<D::HandPoseDetection> {
    let Some(results) = (unsafe { self.requests.hand_pose.results() }) else {
      return Vec::new();
    };

    let mut hand_poses = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      let Ok(points_by_joint) = (unsafe {
        obs.recognizedPointsForJointsGroupName_error(VNHumanHandPoseObservationJointsGroupNameAll)
      }) else {
        continue;
      };

      // Bound the dictionary materialisation: `to_vecs()` allocates
      // two `Vec`s sized to `points_by_joint.len()`, which is FFI-
      // reported. Drop the pose entirely if the joint count
      // exceeds `MAX_POSE_JOINTS` rather than risking an allocator
      // abort on a corrupted/adversarial Vision dictionary.
      if points_by_joint.len() > MAX_POSE_JOINTS {
        continue;
      }
      let (joint_names, points) = points_by_joint.to_vecs();
      let mut joints = Vec::with_capacity(points.len());
      let mut min_x = f32::INFINITY;
      let mut min_y = f32::INFINITY;
      let mut max_x = f32::NEG_INFINITY;
      let mut max_y = f32::NEG_INFINITY;

      for (joint_name, point) in joint_names.into_iter().zip(points) {
        let Some(name) = ffi_nsstring_to_smolstr(&joint_name) else {
          continue;
        };
        if name.is_empty() {
          continue;
        }

        // Vision normalized points are lower-left origin; flip y for
        // the top-left schema convention. A non-finite raw coordinate
        // is dropped at the source — partial-joint hand lists are
        // valid so we skip only this joint.
        let Some((x, y)) = vision_point_to_normalized(unsafe { point.x() }, unsafe { point.y() })
        else {
          continue;
        };
        let Some(confidence) = sanitize_confidence(
          unsafe { point.confidence() },
          options.hand_pose().min_joint_confidence(),
        ) else {
          continue;
        };

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);

        let Ok(joint) = D::HandJoint::try_new(&name, x, y, confidence) else {
          continue;
        };
        joints.push(joint);
      }

      if joints.is_empty() {
        continue;
      }

      let Some(bbox) = pose_bbox_from_joint_bounds(min_x, min_y, max_x, max_y) else {
        continue;
      };
      let Some(pose_confidence) = sanitize_confidence(unsafe { obs.confidence() }, 0.0) else {
        continue;
      };

      joints.sort_by(|lhs, rhs| lhs.name().cmp(rhs.name()));
      if let Ok(pose) = D::HandPoseDetection::try_new(
        bbox,
        pose_confidence,
        map_hand_chirality(unsafe { obs.chirality() }),
        joints,
      ) {
        hand_poses.push(pose);
      }
    }

    hand_poses
  }

  fn extract_person_instance_masks<D: Detections>(
    &self,
    options: &AnalyzeOptions,
    total_mask_bytes: &mut usize,
    total_mask_count: &mut usize,
    total_mask_attempts: &mut usize,
  ) -> Vec<D::PersonInstanceMaskDetection> {
    let Some(results) = (unsafe { self.requests.person_instance_mask.results() }) else {
      return Vec::new();
    };
    let opts = options.person_instance_masks();

    // Per-frame budgets — count AND cumulative payload bytes — are
    // SHARED across both mask extractors via the caller's mutable
    // counters, so the cap holds across the keyframe as a whole.
    let mut masks = Vec::new();
    'outer: for observation in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      if *total_mask_count >= MAX_TOTAL_MASKS_PER_FRAME
        || *total_mask_bytes >= MAX_TOTAL_MASK_BYTES_PER_FRAME
      {
        break;
      }
      let Some(confidence) =
        sanitize_confidence(unsafe { observation.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      let instances = unsafe { observation.allInstances() };
      let mut instance_index = instances.firstIndex();
      // Track ATTEMPTS (every iteration), not just successful
      // emissions — otherwise a corrupted NSIndexSet whose entries
      // all fail generation/copy/u32 conversion can drive unbounded
      // traversal at full Vision-call cost per iteration
      // (codex R14 F2).
      let mut visited = 0usize;
      let inner_cap = opts
        .max_instances_per_observation()
        .min(MAX_NESTED_INSTANCES_PER_OBSERVATION);
      while instance_index != NSNotFound as usize {
        if visited >= inner_cap {
          break;
        }
        visited += 1;
        // Per-frame budget check: stop the entire extraction once
        // any ceiling is reached (success-path counters OR the
        // failure-path attempt counter from codex R15).
        if *total_mask_count >= MAX_TOTAL_MASKS_PER_FRAME
          || *total_mask_bytes >= MAX_TOTAL_MASK_BYTES_PER_FRAME
          || *total_mask_attempts >= MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME
        {
          break 'outer;
        }

        // Validate u32-fit of the instance index BEFORE generating
        // or copying the mask — overflowing here would force a
        // costly retry per-iteration; cheaper to skip up-front.
        let Ok(wire_instance_index) = u32::try_from(instance_index) else {
          instance_index = instances.indexGreaterThanIndex(instance_index);
          continue;
        };

        // Charge the per-frame attempt budget BEFORE the
        // expensive Vision call. Even if generation fails (Apple
        // returns Err), it still costs Vision time + intermediate
        // alloc, so the failure path must be frame-budgeted
        // (codex R15 F1).
        *total_mask_attempts = total_mask_attempts.saturating_add(1);
        let selected_instances = NSIndexSet::indexSetWithIndex(instance_index);
        let Ok(mask_buffer) =
          (unsafe { observation.generateMaskForInstances_error(&selected_instances) })
        else {
          instance_index = instances.indexGreaterThanIndex(instance_index);
          continue;
        };

        // Pre-allocation budget check: pass the remaining cumulative
        // budget into the copier so it rejects the mask BEFORE
        // allocating if the packed size would overshoot.
        let remaining_budget = MAX_TOTAL_MASK_BYTES_PER_FRAME.saturating_sub(*total_mask_bytes);
        let Some((bbox, width, height, data)) =
          copy_instance_mask_buffer::<D::BoundingBox>(&mask_buffer, remaining_budget)
        else {
          instance_index = instances.indexGreaterThanIndex(instance_index);
          continue;
        };

        let data_len = data.len();
        match D::PersonInstanceMaskDetection::try_new(
          bbox,
          confidence,
          wire_instance_index,
          width,
          height,
          &data,
        ) {
          Ok(mask) => {
            *total_mask_bytes = total_mask_bytes.saturating_add(data_len);
            *total_mask_count = total_mask_count.saturating_add(1);
            masks.push(mask);
          }
          Err(_) => {
            // Refused — `width`/`height` are already verified > 0 and
            // `data` is non-empty before reaching here, so this only
            // triggers on a vocabulary invariant the engine does not
            // model.
          }
        }

        instance_index = instances.indexGreaterThanIndex(instance_index);
      }
    }

    masks
  }

  fn extract_person_segmentation_masks<D: Detections>(
    &self,
    options: &AnalyzeOptions,
    total_mask_bytes: &mut usize,
    total_mask_count: &mut usize,
    total_mask_attempts: &mut usize,
  ) -> Vec<D::PersonSegmentationMask> {
    let Some(results) = (unsafe { self.requests.person_segmentation.results() }) else {
      return Vec::new();
    };
    let opts = options.person_segmentation_masks();

    // Shared per-frame mask budget — counters owned by the caller,
    // also charged by `extract_person_instance_masks`. The cumulative
    // cap holds across BOTH extractors, not per-extractor (codex
    // R14 F1).
    let mut masks = Vec::new();
    for observation in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      if *total_mask_count >= MAX_TOTAL_MASKS_PER_FRAME
        || *total_mask_bytes >= MAX_TOTAL_MASK_BYTES_PER_FRAME
        || *total_mask_attempts >= MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME
      {
        break;
      }
      let Some(confidence) =
        sanitize_confidence(unsafe { observation.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      // Charge the per-frame attempt budget before pulling the
      // pixel buffer and running the copy. Even a failing copy
      // path costs FFI traversal + bounded alloc, so the failure
      // path must be frame-budgeted (codex R15 F1, same policy as
      // the instance-mask extractor).
      *total_mask_attempts = total_mask_attempts.saturating_add(1);
      let pixel_buffer = unsafe { observation.pixelBuffer() };
      // Pre-allocation budget check: refuse the mask before alloc
      // if it would overshoot the per-frame cumulative budget.
      let remaining_budget = MAX_TOTAL_MASK_BYTES_PER_FRAME.saturating_sub(*total_mask_bytes);
      let Some((bbox, width, height, data)) =
        copy_instance_mask_buffer::<D::BoundingBox>(&pixel_buffer, remaining_budget)
      else {
        continue;
      };

      let data_len = data.len();
      if let Ok(mask) = D::PersonSegmentationMask::try_new(bbox, confidence, width, height, &data) {
        *total_mask_bytes = total_mask_bytes.saturating_add(data_len);
        *total_mask_count = total_mask_count.saturating_add(1);
        masks.push(mask);
      }
    }

    masks
  }

  fn extract_animal_subjects<D: Detections>(
    &self,
    options: &AnalyzeOptions,
  ) -> Vec<D::SubjectDetection> {
    unsafe {
      let Some(results) = self.requests.animals.results() else {
        return Vec::new();
      };

      let mut animals = Vec::with_capacity(MAX_TOTAL_ANIMAL_SUBJECTS_PER_FRAME);
      'outer: for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
        if animals.len() >= MAX_TOTAL_ANIMAL_SUBJECTS_PER_FRAME {
          break;
        }
        let labels = obs.labels();
        // Per-frame total cap: animal subjects can't multiply across
        // outer × inner past the hard ceiling. The inner per-obs
        // take cap remains so a single hostile observation can't
        // exhaust the budget on its own either.
        for label in labels.iter().take(MAX_NESTED_LABELS_PER_OBSERVATION) {
          if animals.len() >= MAX_TOTAL_ANIMAL_SUBJECTS_PER_FRAME {
            break 'outer;
          }
          let Some(confidence) =
            sanitize_confidence(label.confidence(), options.animals().min_confidence())
          else {
            continue;
          };
          let identifier = label.identifier();
          let Some(id) = ffi_nsstring_to_smolstr(&identifier) else {
            continue;
          };
          if !id.is_empty()
            && let Some(bbox) = vision_rect_to_bbox(obs.boundingBox().standardize())
            && let Ok(detection) = D::Detection::try_new(&id, confidence)
          {
            animals.push(D::SubjectDetection::new(detection, bbox));
          }
        }
      }

      animals
    }
  }

  fn extract_animal_body_poses<D: Detections>(
    &self,
    options: &AnalyzeOptions,
  ) -> Vec<D::AnimalPoseDetection> {
    let Some(results) = (unsafe { self.requests.animal_body_pose.results() }) else {
      return Vec::new();
    };
    let Some(group_name) = (unsafe { VNAnimalBodyPoseObservationJointsGroupNameAll }) else {
      return Vec::new();
    };

    let mut body_poses = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      let Ok(points_by_joint) =
        (unsafe { obs.recognizedPointsForJointsGroupName_error(group_name) })
      else {
        continue;
      };

      // Bound the dictionary materialisation: `to_vecs()` allocates
      // two `Vec`s sized to `points_by_joint.len()`, which is FFI-
      // reported. Drop the pose entirely if the joint count
      // exceeds `MAX_POSE_JOINTS` rather than risking an allocator
      // abort on a corrupted/adversarial Vision dictionary.
      if points_by_joint.len() > MAX_POSE_JOINTS {
        continue;
      }
      let (joint_names, points) = points_by_joint.to_vecs();
      let mut joints = Vec::with_capacity(points.len());
      let mut min_x = f32::INFINITY;
      let mut min_y = f32::INFINITY;
      let mut max_x = f32::NEG_INFINITY;
      let mut max_y = f32::NEG_INFINITY;

      for (joint_name, point) in joint_names.into_iter().zip(points) {
        let Some(name) = ffi_nsstring_to_smolstr(&joint_name) else {
          continue;
        };
        if name.is_empty() {
          continue;
        }

        // Vision normalized points are lower-left origin; flip y for
        // the top-left schema convention. A non-finite raw coordinate
        // is dropped at the source — partial-joint animal-pose lists
        // are valid so we skip only this joint.
        let Some((x, y)) = vision_point_to_normalized(unsafe { point.x() }, unsafe { point.y() })
        else {
          continue;
        };
        let Some(confidence) = sanitize_confidence(
          unsafe { point.confidence() },
          options.animal_pose().min_joint_confidence(),
        ) else {
          continue;
        };

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);

        let Ok(joint) = D::AnimalJoint::try_new(&name, x, y, confidence) else {
          continue;
        };
        joints.push(joint);
      }

      if joints.is_empty() {
        continue;
      }

      let Some(bbox) = pose_bbox_from_joint_bounds(min_x, min_y, max_x, max_y) else {
        continue;
      };
      let Some(pose_confidence) = sanitize_confidence(unsafe { obs.confidence() }, 0.0) else {
        continue;
      };

      joints.sort_by(|lhs, rhs| lhs.name().cmp(rhs.name()));
      if let Ok(pose) = D::AnimalPoseDetection::try_new(bbox, pose_confidence, joints) {
        body_poses.push(pose);
      }
    }

    body_poses
  }

  fn extract_text_detections<D: Detections>(
    &self,
    options: &AnalyzeOptions,
  ) -> Vec<D::TextDetection> {
    let Some(results) = self.requests.text.results() else {
      return Vec::new();
    };

    // Per-frame total cap on emitted text detections — bounds the
    // outer × inner candidate product that codex R13 surfaced.
    let mut text_detections = Vec::with_capacity(MAX_TOTAL_TEXT_DETECTIONS_PER_FRAME);
    'outer: for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      if text_detections.len() >= MAX_TOTAL_TEXT_DETECTIONS_PER_FRAME {
        break;
      }
      // Bound the requested candidate count to the hard per-observation cap
      // — Apple's topCandidates allocates an NSArray sized to the argument.
      let candidate_cap = options
        .text()
        .max_candidates_per_observation()
        .min(MAX_TEXT_CANDIDATES_PER_OBSERVATION);
      let candidates = obs.topCandidates(candidate_cap);
      for candidate in candidates.iter().take(candidate_cap) {
        if text_detections.len() >= MAX_TOTAL_TEXT_DETECTIONS_PER_FRAME {
          break 'outer;
        }
        // Bound the candidate string at MAX_FFI_STRING_BYTES before
        // routing through `to_smolstr` so a corrupted/adversarial
        // NSString length cannot drive the allocator into the abort
        // path.
        let raw_string = candidate.string();
        let Some(text) = ffi_nsstring_to_smolstr(&raw_string) else {
          continue;
        };
        if text.len() < options.text().min_text_len() {
          continue;
        }
        let Some(confidence) = sanitize_confidence(candidate.confidence(), 0.0) else {
          continue;
        };
        if let Some(bbox) = vision_rect_to_bbox(unsafe { obs.boundingBox() }.standardize())
          && let Ok(detection) = D::TextDetection::try_new(&text, confidence, bbox)
        {
          text_detections.push(detection);
        }
      }
    }
    text_detections
  }

  fn extract_barcodes<D: Detections>(&self, options: &AnalyzeOptions) -> Vec<D::BarcodeDetection> {
    let Some(results) = (unsafe { self.requests.barcodes.results() }) else {
      return Vec::new();
    };
    let opts = options.barcodes();

    let mut barcodes = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      let Some(confidence) =
        sanitize_confidence(unsafe { obs.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      if let Some(payload) = unsafe { obs.payloadStringValue() } {
        // Bound the payload + symbology at MAX_FFI_STRING_BYTES.
        let Some(s) = ffi_nsstring_to_smolstr(&payload) else {
          continue;
        };
        if s.len() >= opts.min_payload_len()
          && let Some(bbox) = vision_rect_to_bbox(unsafe { obs.boundingBox() }.standardize())
        {
          let raw_sym = unsafe { obs.symbology() };
          let Some(symbology) = ffi_nsstring_to_smolstr(&raw_sym) else {
            continue;
          };
          if let Ok(barcode) = D::BarcodeDetection::try_new(&s, &symbology, confidence, bbox) {
            barcodes.push(barcode);
          }
        }
      }
    }
    barcodes
  }

  fn extract_attention_saliency<D: Detections>(
    &self,
    options: &AnalyzeOptions,
  ) -> Vec<D::SaliencyRegion> {
    self.extract_saliency_regions::<D>(
      unsafe { self.requests.attention_saliency.results() },
      options.attention_saliency(),
    )
  }

  fn extract_objectness_saliency<D: Detections>(
    &self,
    options: &AnalyzeOptions,
  ) -> Vec<D::SaliencyRegion> {
    self.extract_saliency_regions::<D>(
      unsafe { self.requests.objectness_saliency.results() },
      options.objectness_saliency(),
    )
  }

  fn extract_saliency_regions<D: Detections>(
    &self,
    observations: Option<Retained<NSArray<VNSaliencyImageObservation>>>,
    opts: AppleVisionSaliencyOptions,
  ) -> Vec<D::SaliencyRegion> {
    let Some(observations) = observations else {
      return Vec::new();
    };

    // `total_cap` is the per-FRAME (not per-observation) cap.
    // Outer × inner emission must not exceed it. Track running
    // count across observations and stop the outer loop when the
    // budget is exhausted; `.iter().take(remaining)` on the inner
    // loop further bounds each observation's contribution.
    let total_cap = opts.max_regions().min(MAX_SALIENCY_REGIONS_PER_FRAME);
    let mut regions = Vec::with_capacity(total_cap);
    'outer: for observation in observations.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      if regions.len() >= total_cap {
        break;
      }
      let Some(objects) = (unsafe { observation.salientObjects() }) else {
        continue;
      };
      let remaining = total_cap - regions.len();
      for object in objects.iter().take(remaining) {
        if regions.len() >= total_cap {
          break 'outer;
        }
        let Some(confidence) =
          sanitize_confidence(unsafe { object.confidence() }, opts.min_confidence())
        else {
          continue;
        };

        let Some(bbox) = vision_rect_to_bbox(unsafe { object.boundingBox() }.standardize()) else {
          continue;
        };
        let Ok(region) = D::SaliencyRegion::try_new(bbox, confidence) else {
          continue;
        };
        regions.push(region);
      }
    }
    regions
  }

  fn extract_horizon<D: Detections>(&self, options: &AnalyzeOptions) -> Option<D::HorizonInfo> {
    // `try_new(0.0, 0.0)` is the canonical "no detection" sentinel.
    // When the output type was fixed and known to accept it this was
    // an `expect`; an open vocabulary may refuse, and losing the slot
    // beats panicking inside a worker thread.
    let empty = || D::HorizonInfo::try_new(0.0, 0.0).ok();
    let Some(results) = (unsafe { self.requests.horizon.results() }) else {
      return empty();
    };
    let Some(observation) = results.iter().next() else {
      return empty();
    };
    let Some(confidence) = sanitize_confidence(
      unsafe { observation.confidence() },
      options.horizon().min_confidence(),
    ) else {
      return empty();
    };

    // Drop the horizon detection entirely if the angle is non-finite —
    // there is no sensible default for a horizon line and downstream
    // visualisation would render a bogus tilt.
    let Some(angle) = finite_f32(unsafe { observation.angle() } as f32) else {
      return empty();
    };
    D::HorizonInfo::try_new(angle, confidence)
      .ok()
      .or_else(empty)
  }

  fn extract_document_segments<D: Detections>(
    &self,
    options: &AnalyzeOptions,
  ) -> Vec<D::DocumentSegment> {
    let Some(results) = (unsafe { self.requests.document_segments.results() }) else {
      return Vec::new();
    };
    let opts = options.document_segments();

    // Effective cap: user-configured max_segments AND hard ceiling.
    // with_capacity, take, and the emission guard all share `cap`.
    let cap = effective_results_cap(opts.max_segments());
    let mut segments = Vec::with_capacity(cap);
    for observation in results.iter().take(cap) {
      if segments.len() >= cap {
        break;
      }

      let Some(confidence) =
        sanitize_confidence(unsafe { observation.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      // Vision's named corners ("topLeft" etc.) refer to image-space
      // orientation but use the framework's lower-left-origin coordinate
      // system, so each corner's `y` must be flipped to land in the
      // top-left schema convention. The naming still matches afterwards
      // (the corner with the smallest `y` is still the top edge).
      // A non-finite corner means the quad is geometrically meaningless
      // — drop the whole detection rather than fabricate edge-aligned
      // corners that downstream validation would accept as real.
      let (Some(top_left), Some(top_right), Some(bottom_left), Some(bottom_right)) = (
        vision_point_to_normalized(
          unsafe { observation.topLeft() }.x,
          unsafe { observation.topLeft() }.y,
        ),
        vision_point_to_normalized(
          unsafe { observation.topRight() }.x,
          unsafe { observation.topRight() }.y,
        ),
        vision_point_to_normalized(
          unsafe { observation.bottomLeft() }.x,
          unsafe { observation.bottomLeft() }.y,
        ),
        vision_point_to_normalized(
          unsafe { observation.bottomRight() }.x,
          unsafe { observation.bottomRight() }.y,
        ),
      ) else {
        continue;
      };

      // Even after per-corner clamping, the resulting quad can be
      // degenerate (coincident corners, zero shoelace area, or
      // self-intersecting) when Vision returned an off-screen segment
      // or near-collinear corners. A validating vocabulary runs the
      // geometry guards (collapsed corners, zero area, bow-tie /
      // inconsistent winding); a refusal means the quad is not a real
      // document detection and the segment is dropped. Note the
      // corners go out in WINDING order — the locals above are bound
      // in raster order.
      let Ok(segment) =
        D::DocumentSegment::try_new(top_left, top_right, bottom_right, bottom_left, confidence)
      else {
        continue;
      };
      segments.push(segment);
    }

    segments
  }

  fn extract_aesthetics<D: Detections>(&self, options: &AnalyzeOptions) -> D::Aesthetics {
    // `new(0.0, false)` is the canonical "no detection" sentinel.
    let empty = D::Aesthetics::new(0.0, false);
    let Some(results) = (unsafe { self.requests.aesthetics.results() }) else {
      return empty;
    };
    let Some(obs) = results.iter().next() else {
      return empty;
    };
    // `NaN < threshold` would fail open. Force a finite check at the
    // gate so a glitched aesthetics score collapses to the default
    // (no detection) instead of being silently admitted to the wire.
    let Some(overall_score) = finite_f32(unsafe { obs.overallScore() }) else {
      return empty;
    };
    if overall_score < options.aesthetics().min_overall_score() {
      return empty;
    }

    D::Aesthetics::new(overall_score, unsafe { obs.isUtility() })
  }
}

#[cfg(target_vendor = "apple")]
fn normalize_classification_label(label: SmolStr) -> SmolStr {
  label.trim().to_ascii_lowercase_smolstr()
}

#[cfg(target_vendor = "apple")]
fn extract_body_pose_3d_coordinates(
  point: &VNHumanBodyRecognizedPoint3D,
) -> Option<(f32, f32, f32)> {
  let transform: SimdFloat4x4 = unsafe { objc2::msg_send![point, position] };
  let translation = transform.columns.get(3)?;
  let x = translation.0[0];
  let y = translation.0[1];
  let z = translation.0[2];
  if !(x.is_finite() && y.is_finite() && z.is_finite()) {
    return None;
  }
  Some((x, y, z))
}

#[cfg(target_vendor = "apple")]
fn map_hand_chirality(chirality: VNChirality) -> Chirality {
  match chirality {
    VNChirality::Left => Chirality::Left,
    VNChirality::Right => Chirality::Right,
    _ => Chirality::Unknown,
  }
}

/// Extract every named face-landmark region, projecting each point
/// from face-bbox-relative coordinates into image-normalized
/// coordinates (Vision lower-left) via `face_bbox_vision` before the
/// caller-side schema flip. Without this projection a non-full-frame
/// face emits landmarks in the wrong place but still passes `[0, 1]`
/// validation.
#[cfg(target_vendor = "apple")]
fn extract_face_landmark_regions<R: FaceLandmarkRegion>(
  landmarks: &VNFaceLandmarks2D,
  face_bbox_vision: CGRect,
  total_points_remaining: &mut usize,
  total_landmark_attempts: &mut usize,
) -> Vec<R> {
  let mut regions = Vec::new();
  for (name, region) in [
    ("allPoints", unsafe { landmarks.allPoints() }),
    ("faceContour", unsafe { landmarks.faceContour() }),
    ("leftEye", unsafe { landmarks.leftEye() }),
    ("rightEye", unsafe { landmarks.rightEye() }),
    ("leftEyebrow", unsafe { landmarks.leftEyebrow() }),
    ("rightEyebrow", unsafe { landmarks.rightEyebrow() }),
    ("nose", unsafe { landmarks.nose() }),
    ("noseCrest", unsafe { landmarks.noseCrest() }),
    ("medianLine", unsafe { landmarks.medianLine() }),
    ("outerLips", unsafe { landmarks.outerLips() }),
    ("innerLips", unsafe { landmarks.innerLips() }),
    ("leftPupil", unsafe { landmarks.leftPupil() }),
    ("rightPupil", unsafe { landmarks.rightPupil() }),
  ] {
    if *total_points_remaining == 0
      || *total_landmark_attempts >= MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME
    {
      break;
    }
    push_face_landmark_region(
      &mut regions,
      name,
      region,
      face_bbox_vision,
      total_points_remaining,
      total_landmark_attempts,
    );
  }
  regions
}

#[cfg(target_vendor = "apple")]
fn push_face_landmark_region<R: FaceLandmarkRegion>(
  regions: &mut Vec<R>,
  name: &'static str,
  region: Option<Retained<VNFaceLandmarkRegion2D>>,
  face_bbox_vision: CGRect,
  total_points_remaining: &mut usize,
  total_landmark_attempts: &mut usize,
) {
  if *total_points_remaining == 0
    || *total_landmark_attempts >= MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME
  {
    return;
  }
  let Some(region) = region else {
    return;
  };

  let point_count = unsafe { region.pointCount() };
  if point_count == 0 {
    return;
  }
  // Pre-validate the raw-slice preconditions
  // (count <= MAX_LANDMARK_POINTS AND count * size_of::<CGPoint>() <= isize::MAX)
  // BEFORE the unsafe slice construction. Same policy as
  // `validate_mask_dims_for_slice` and the FeaturePrint path —
  // every Vision boundary that builds a Rust slice from an FFI
  // pointer goes through a `validate_raw_slice_*` gate.
  if validate_raw_slice_elems::<CGPoint>(point_count, MAX_LANDMARK_POINTS).is_none() {
    return;
  }

  let points_ptr = unsafe { region.normalizedPoints() };
  if points_ptr.is_null() {
    return;
  }

  // Construct the unsafe slice to the CAPPED element count, not
  // the FFI-reported point_count: when remaining < point_count,
  // exposing the full slice to subsequent code would let
  // from_raw_parts trust more elements than we'll read (codex
  // R14 F4). The cap is `point_count.min(remaining_budget)` —
  // already validated against MAX_LANDMARK_POINTS above. Also
  // bound by the remaining attempt budget so we never visit more
  // points than the frame can afford to walk.
  let attempts_remaining =
    MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME.saturating_sub(*total_landmark_attempts);
  let region_cap = point_count
    .min(*total_points_remaining)
    .min(attempts_remaining);
  if region_cap == 0 {
    return;
  }
  // Charge the ATTEMPT budget for every point we're about to walk,
  // up front and unconditionally. Whether the points later survive
  // finite-checks or the parent detection survives its gates, the
  // walk itself is bounded by this budget (codex R17 F1).
  *total_landmark_attempts = total_landmark_attempts.saturating_add(region_cap);
  // SAFETY: `points_ptr` points at `point_count` valid `CGPoint`s
  // (Vision API contract). `region_cap <= point_count` and
  // `region_cap <= MAX_LANDMARK_POINTS` (verified via
  // `validate_raw_slice_elems::<CGPoint>` above), so the slice
  // length fits both the FFI buffer and the `isize::MAX` contract.
  let points = unsafe { std::slice::from_raw_parts(points_ptr, region_cap) };
  // The seat takes raw `(f32, f32)` tuples. Every point is already
  // sanitised through `vision_point_to_normalized` (finite + clamped
  // to `[0, 1]`), so a validating vocabulary cannot reject one.
  let mut emitted_points: Vec<(f32, f32)> = Vec::with_capacity(region_cap);
  for point in points.iter() {
    // Apple's convention: landmark points are normalized within the
    // face's normalized bbox (NOT the image). Project to image-
    // normalized Vision coordinates first, THEN route through
    // `vision_point_to_normalized` for the top-left flip + clamp +
    // finite check. A non-finite raw or projected component drops
    // only the offending point; partial-point regions are still
    // meaningful.
    let projected = project_landmark_to_image(*point, face_bbox_vision);
    if let Some((x, y)) = vision_point_to_normalized(projected.x, projected.y) {
      emitted_points.push((x, y));
    }
  }
  if emitted_points.is_empty() {
    return;
  }
  let emitted_len = emitted_points.len();
  let Ok(region) = R::try_new(name, &emitted_points) else {
    return;
  };
  // Decrement the shared budget by the number of points actually
  // emitted (finite-rejected points don't consume budget).
  *total_points_remaining = total_points_remaining.saturating_sub(emitted_len);

  regions.push(region);
}

#[cfg(target_vendor = "apple")]
fn map_body_pose_3d_height_estimation(
  estimation: VNHumanBodyPose3DObservationHeightEstimation,
) -> HeightEstimation {
  if estimation == VNHumanBodyPose3DObservationHeightEstimation::Measured {
    HeightEstimation::Measured
  } else if estimation == VNHumanBodyPose3DObservationHeightEstimation::Reference {
    HeightEstimation::Reference
  } else {
    HeightEstimation::Unknown
  }
}

/// Copy a Vision mask `CVPixelBuffer` into a packed byte payload plus
/// a normalized bounding box of the foreground and the buffer's own
/// `(width, height)`.
///
/// The returned payload is **always** 8 bits per pixel
/// (`width * height` bytes); Vision's two supported source formats
/// (`OneComponent32Float`, `OneComponent8`) are both normalised to
/// canonical u8 at the boundary so downstream consumers don't have
/// to disambiguate from the returned dimensions alone. f32 input
/// is mapped `v` → `(v.clamp(0.0, 1.0) * 255.0).round() as u8` with
/// non-finite values collapsing to `0` (background).
///
/// Returns `None` when the buffer is unlockable, has zero extent, a null
/// base address, an unsupported pixel format, fails one of the
/// stride/size sanity checks, or contains no foreground pixels (an
/// all-zero mask is represented by skipping the detection rather than
/// emitting one with a degenerate bbox). The lock is held via
/// [`CVPixelBufferLockGuard`] for the duration of the copy and is
/// released by `Drop` on every exit path — including a panic — so the
/// buffer cannot be left locked.
#[cfg(target_vendor = "apple")]
fn copy_instance_mask_buffer<B: BoundingBox>(
  pixel_buffer: &CVPixelBuffer,
  remaining_byte_budget: usize,
) -> Option<(B, u32, u32, Vec<u8>)> {
  let guard = CVPixelBufferLockGuard::lock(pixel_buffer, CVPixelBufferLockFlags::ReadOnly)?;
  copy_instance_mask_buffer_locked(guard.buffer(), remaining_byte_budget)
}

/// Internal worker that runs the locked copy and assembles the wire
/// payload. The caller is responsible for holding the
/// [`CVPixelBufferLockGuard`].
///
/// The returned payload is **always** 8 bits per pixel
/// (`width * height` bytes) regardless of the source pixel format.
/// Vision can emit either `kCVPixelFormatType_OneComponent32Float`
/// (4 bytes/pixel) or `kCVPixelFormatType_OneComponent8`
/// (1 byte/pixel); both are normalised to the canonical u8 wire
/// representation so downstream consumers don't have to disambiguate
/// from the returned dimensions alone. The f32 → u8 quantisation
/// is `(v.clamp(0.0, 1.0) * 255.0).round() as u8` with non-finite
/// inputs collapsed to `0` (background); see
/// [`process_mask_bytes_f32`] for the per-pixel logic.
#[cfg(target_vendor = "apple")]
#[allow(non_upper_case_globals)]
fn copy_instance_mask_buffer_locked<B: BoundingBox>(
  pixel_buffer: &CVPixelBuffer,
  remaining_byte_budget: usize,
) -> Option<(B, u32, u32, Vec<u8>)> {
  let width = CVPixelBufferGetWidth(pixel_buffer);
  let height = CVPixelBufferGetHeight(pixel_buffer);
  if width == 0 || height == 0 {
    return None;
  }
  // Pre-allocation budget check: refuse to allocate this mask if
  // its packed size (`width * height` bytes) would exceed the
  // caller's remaining per-frame budget. This prevents the peak
  // memory from briefly exceeding the per-frame cap by one full
  // mask payload — codex R13 finding F2.
  let output_payload = width.checked_mul(height)?;
  if output_payload > remaining_byte_budget {
    return None;
  }

  let pixel_format = CVPixelBufferGetPixelFormatType(pixel_buffer);
  let bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
  let base_address = CVPixelBufferGetBaseAddress(pixel_buffer) as *const u8;
  if base_address.is_null() || bytes_per_row == 0 {
    return None;
  }

  // Total foreground-mask byte count cannot overflow `usize`, and the
  // stride must be wide enough to hold one row of pixels of the
  // expected size — otherwise our row-slice indexing would read past
  // the end of the buffer.
  let bytes_per_pixel: usize = match pixel_format {
    kCVPixelFormatType_OneComponent32Float => core::mem::size_of::<f32>(),
    kCVPixelFormatType_OneComponent8 => 1,
    _ => return None,
  };
  let row_pixel_bytes = width.checked_mul(bytes_per_pixel)?;
  if bytes_per_row < row_pixel_bytes {
    return None;
  }
  let total_src_len = bytes_per_row.checked_mul(height)?;

  // Pre-validate the two mask preconditions that `from_raw_parts`
  // requires (`total_src_len <= isize::MAX`) and that the bounded
  // allocator requires (`width * height <= MAX_MASK_BYTES`).
  // Centralised in `validate_mask_dims_for_slice` so a corrupted or
  // adversarial `CVPixelBuffer` cannot reach the unsafe slice with
  // values that would either trigger UB or drive the worker into
  // the allocator's abort path.
  validate_mask_dims_for_slice(width, height, total_src_len)?;

  // FFI-truth check: `total_src_len = bytes_per_row * height` is
  // derived from the buffer's own metadata, but `CVPixelBufferGetDataSize`
  // reports the actual ALLOCATED size of the buffer's data plane.
  // A malformed buffer with valid `base_address` but inconsistent
  // stride/height metadata could otherwise let `from_raw_parts`
  // create an overlong slice (UB on the row reads). Reject if the
  // computed length exceeds the buffer's reported data size.
  let data_size: usize = CVPixelBufferGetDataSize(pixel_buffer);
  if total_src_len > data_size {
    return None;
  }

  // SAFETY: `base_address` points at a buffer whose allocated size
  // is `CVPixelBufferGetDataSize(pixel_buffer)` (verified just above
  // to be at least `total_src_len`); the buffer is locked by the
  // surrounding `CVPixelBufferLockGuard`. The pre-validation
  // satisfies the `from_raw_parts` contract
  // (`total_src_len <= isize::MAX` AND
  // `total_src_len <= data_size`) regardless of what Core Video
  // reports for the dimensions; the downstream bounded allocator
  // re-checks `width * height` against `MAX_MASK_BYTES`.
  let src = unsafe { std::slice::from_raw_parts(base_address, total_src_len) };

  // The mask seat reports its dimensions as `u32`. A mask whose
  // width or height exceeds `u32::MAX` cannot be represented
  // faithfully — we'd have to saturate to a value smaller than the
  // actual packed payload, which would silently desynchronise
  // consumers that size buffers from the metadata. Reject overflow
  // here so the detection is dropped rather than poisoning storage.
  let dim_width = u32::try_from(width).ok()?;
  let dim_height = u32::try_from(height).ok()?;

  let (bbox, packed) = match pixel_format {
    kCVPixelFormatType_OneComponent32Float => {
      process_mask_bytes_f32(width, height, bytes_per_row, src)?
    }
    kCVPixelFormatType_OneComponent8 => process_mask_bytes_u8(width, height, bytes_per_row, src)?,
    _ => return None,
  };

  Some((bbox, dim_width, dim_height, packed))
}

/// Walk an `OneComponent32Float` mask, quantise each pixel to 8 bits,
/// and derive a normalized foreground bbox. Returns `None` for an
/// all-zero mask so the caller skips emitting a detection.
///
/// The result is a `(bbox, packed_bytes)` pair where `packed_bytes`
/// has length `width * height` — i.e. one **u8** per pixel, NOT four
/// `f32` little-endian bytes. Vision emits f32 mask values in
/// `[0.0, 1.0]`; we map `v` → `(v.clamp(0.0, 1.0) * 255.0).round() as
/// u8`. Non-finite values (`NaN`, `±Inf`) collapse to `0`
/// (background), matching Vision's documented "non-finite = no
/// confidence in foreground" convention and keeping the wire payload
/// canonically 8-bit per pixel across both source pixel formats.
#[cfg(target_vendor = "apple")]
fn process_mask_bytes_f32<B: BoundingBox>(
  width: usize,
  height: usize,
  bytes_per_row: usize,
  src: &[u8],
) -> Option<(B, Vec<u8>)> {
  let src_row_pixel_bytes = width.checked_mul(core::mem::size_of::<f32>())?;
  let packed_len = width.checked_mul(height)?;
  // Bounded allocation: cap at `MAX_MASK_BYTES` and use
  // `try_reserve_exact` so an oversized or corrupted dimensions value
  // returns `None` instead of aborting the worker process via the
  // allocator's OOM path.
  let mut packed = try_alloc_packed_mask(packed_len)?;

  let mut min_x = usize::MAX;
  let mut min_y = usize::MAX;
  let mut max_x = 0usize;
  let mut max_y = 0usize;
  let mut has_foreground = false;

  for row in 0..height {
    let src_start = row.checked_mul(bytes_per_row)?;
    let src_end = src_start.checked_add(src_row_pixel_bytes)?;
    let src_row = src.get(src_start..src_end)?;
    let dst_start = row.checked_mul(width)?;
    let dst_end = dst_start.checked_add(width)?;
    let dst_row = packed.get_mut(dst_start..dst_end)?;
    for col in 0..width {
      let pixel_start = col.checked_mul(4)?;
      let pixel_end = pixel_start.checked_add(4)?;
      let bytes: [u8; 4] = src_row.get(pixel_start..pixel_end)?.try_into().ok()?;
      let value = f32::from_le_bytes(bytes);
      // f32 mask in `[0.0, 1.0]` → u8 mask in `[0, 255]`. Non-finite
      // values (`NaN`, `±Inf`) collapse to `0` (background) — Vision
      // documents non-finite as "no confidence", which is the same
      // semantic as background in the u8 representation.
      let quantised: u8 = if value.is_finite() {
        (value.clamp(0.0, 1.0) * 255.0).round() as u8
      } else {
        0
      };
      *dst_row.get_mut(col)? = quantised;
      if quantised > 0 {
        has_foreground = true;
        min_x = min_x.min(col);
        min_y = min_y.min(row);
        max_x = max_x.max(col);
        max_y = max_y.max(row);
      }
    }
  }

  if !has_foreground {
    // All-zero mask — skip the detection rather than emit one with a
    // degenerate bbox. A validating vocabulary rejects zero-extent
    // boxes, so a `default()`-style fallback would poison downstream
    // conversion.
    return None;
  }
  let bbox = normalized_bbox_from_pixel_bounds(min_x, min_y, max_x, max_y, width, height)?;
  Some((bbox, packed))
}

/// Walk an `OneComponent8` mask, copy it tightly packed, and derive a
/// normalized foreground bbox. Returns `None` for an all-zero mask.
#[cfg(target_vendor = "apple")]
fn process_mask_bytes_u8<B: BoundingBox>(
  width: usize,
  height: usize,
  bytes_per_row: usize,
  src: &[u8],
) -> Option<(B, Vec<u8>)> {
  let packed_len = width.checked_mul(height)?;
  // Bounded allocation: see `process_mask_bytes_f32` for the rationale.
  let mut packed = try_alloc_packed_mask(packed_len)?;

  let mut min_x = usize::MAX;
  let mut min_y = usize::MAX;
  let mut max_x = 0usize;
  let mut max_y = 0usize;
  let mut has_foreground = false;

  for row in 0..height {
    let src_start = row.checked_mul(bytes_per_row)?;
    let src_end = src_start.checked_add(width)?;
    let src_row = src.get(src_start..src_end)?;
    let dst_start = row.checked_mul(width)?;
    let dst_end = dst_start.checked_add(width)?;
    let dst_row = packed.get_mut(dst_start..dst_end)?;
    dst_row.copy_from_slice(src_row);
    for (col, value) in dst_row.iter().copied().enumerate() {
      if value > 0 {
        has_foreground = true;
        min_x = min_x.min(col);
        min_y = min_y.min(row);
        max_x = max_x.max(col);
        max_y = max_y.max(row);
      }
    }
  }

  if !has_foreground {
    return None;
  }
  let bbox = normalized_bbox_from_pixel_bounds(min_x, min_y, max_x, max_y, width, height)?;
  Some((bbox, packed))
}

/// Convert the foreground pixel bounds of a `CVPixelBuffer` mask into a
/// normalized bounding box in the top-left convention.
///
/// `CVPixelBuffer` rows are stored top-to-bottom in memory (row 0 is the
/// top of the image), so the natural mapping `min_y / height` is already
/// top-left and no y-flip is needed here.
/// Convert integer pixel bounds into a normalized `[0, 1]`
/// `BoundingBox`. The intermediate division is performed in `f64`
/// because R7's bounded mask cap (`MAX_MASK_BYTES = 64 MiB`) admits
/// widths above `2^24`, where consecutive `usize` values round to
/// the same `f32` (mantissa exhaustion). A naive
/// `min_x as f32 / width as f32` then produces `x = 1.0` with a
/// positive width on right-edge foreground at width `2^24 + 1`,
/// which violates the schema's `[0, 1]` invariant.
///
/// `f64` has 52 mantissa bits and represents every `usize` up to
/// `2^52` exactly on 64-bit targets; only the final narrow to `f32`
/// loses precision, which is invariant-safe because the result is
/// in `[0, 1]`. Returns `None` if the final box is refused — a
/// corrupted pixel-bound input cannot produce a box that downstream
/// storage would reject.
#[cfg(target_vendor = "apple")]
fn normalized_bbox_from_pixel_bounds<B: BoundingBox>(
  min_x: usize,
  min_y: usize,
  max_x: usize,
  max_y: usize,
  width: usize,
  height: usize,
) -> Option<B> {
  if width == 0 || height == 0 {
    return None;
  }
  // Compute all four EDGES in f64, then narrow to f32. The width
  // and height are derived from the narrowed edges (right - left,
  // bottom - top) rather than from a separate f64-division. This
  // guarantees the bbox is internally consistent in f32 arithmetic:
  // `left + width == right` exactly, and `right <= 1.0` by
  // construction (numerator <= denominator).
  //
  // Why edges-then-subtract instead of left-then-width: at widths
  // above the f32 mantissa exhaustion point (2^24+1, 2^25+1, …),
  // separate f32 narrowings of `min_x / width` and `(max_x + 1 - min_x)
  // / width` can land on (1.0, positive) for right-edge foreground,
  // producing `x == 1.0 && w > 0.0` — a box a validating
  // vocabulary does NOT reliably reject (an `x + w` check is also
  // f32 and rounds back to 1.0). Edge-based computation eliminates the
  // class entirely: the right edge is constructed directly as a
  // normalized value, not synthesised from `left + width`.
  //
  // `f64` has 52 mantissa bits — every `usize` up to `2^52` is
  // representable exactly on 64-bit targets. Only the final narrow
  // to `f32` loses precision, and only at values close enough to
  // `1.0` that the next-lower f32 differs by `2^-24 ≈ 6e-8`.
  let w64 = width as f64;
  let h64 = height as f64;
  // `max_x + 1` would overflow `usize::MAX`; the caller bounds
  // `max_x < width <= MAX_MASK_BYTES` (well below `usize::MAX`),
  // but use `checked_add` for defence-in-depth.
  let right_pixel = max_x.checked_add(1)?;
  let bottom_pixel = max_y.checked_add(1)?;
  if right_pixel > width || bottom_pixel > height || min_x > max_x || min_y > max_y {
    return None;
  }
  let left = (min_x as f64 / w64) as f32;
  let top = (min_y as f64 / h64) as f32;
  let right = (right_pixel as f64 / w64) as f32;
  let bottom = (bottom_pixel as f64 / h64) as f32;
  let w = right - left;
  let h = bottom - top;
  // Reject pathological f32 narrowings: a foreground bbox whose
  // left or top edge rounded to 1.0 (i.e. mantissa exhaustion
  // pushed an "almost-1.0" value over the line) is geometrically
  // a point, not a region. Drop it rather than emitting an
  // out-of-spec wire bbox.
  if !(left < 1.0 && top < 1.0) {
    return None;
  }
  if !(w > 0.0 && h > 0.0) {
    return None;
  }
  B::try_new(left, top, w, h).ok()
}

// ----- Non-macOS stub --------------------------------------------------------

/// Non-macOS stub for [`VisionAnalyzer`]. Apple's Vision.framework is
/// only available on macOS, so on every other target the analyzer
/// always reports [`AnalyzeErrorKind::Unsupported`] rather than
/// producing detections. The README promises the crate compiles
/// cleanly on non-macOS targets so downstream workspaces can keep
/// `avanalyze` in their dep tree unconditionally; this stub is what
/// makes that promise true.
#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct VisionAnalyzer;

#[cfg(not(target_vendor = "apple"))]
impl VisionAnalyzer {
  /// Constructs a non-macOS stub analyzer. The options are ignored —
  /// every `analyze_keyframe` call reports
  /// [`AnalyzeErrorKind::Unsupported`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AnalyzeOptions) -> Self {
    Self
  }

  /// Non-macOS stub: Apple's Vision.framework is only available on
  /// macOS, so this always reports
  /// [`AnalyzeErrorKind::Unsupported`]. `_jpeg_data` is ignored.
  pub fn analyze_keyframe<D: Detections>(
    &self,
    _jpeg_data: &[u8],
    _options: &AnalyzeOptions,
  ) -> Result<Analysis<D>, AnalyzeError> {
    Err(AnalyzeError::new(
      AnalyzeErrorKind::Unsupported,
      "Apple Vision.framework is only available on macOS",
    ))
  }
}
