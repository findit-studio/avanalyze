#![doc = include_str!("../README.md")]
// #![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

//! Long-running Apple Vision.framework service thread.
//!
//! Each worker thread owns an `AppleVisionAnalyzer` and processes keyframes
//! independently. Vision.framework is stateless per-request, so multiple
//! workers can run in parallel.
//!
//! Input: `Request` via crossbeam bounded channel
//! Output: `Reply` via callback back to the processor-local coordinator

#[cfg(target_os = "macos")]
use std::panic::{AssertUnwindSafe, catch_unwind};

#[cfg(target_os = "macos")]
use bytes::Bytes;
#[cfg(target_os = "macos")]
use mediaschema::{
  Aesthetics, AnimalAnalysis, BarcodeDetection, BodyPose3DDetection, BodyPose3DHeightEstimation,
  BodyPose3DJoint, BodyPoseDetection, BodyPoseJoint, BoundingBox, ClassificationDetection,
  Dimensions, DocumentSegment, FaceDetection, FaceLandmarkPoint, FaceLandmarkRegion,
  FaceLandmarksDetection, FeaturePrint, HandChirality, HandPoseDetection, HorizonInfo,
  HumanAnalysis, PersonInstanceMaskDetection, PersonSegmentationMask, SaliencyRegion,
  SubjectDetection, TextDetection,
};
use mediaschema::{ErrorInfo, Id, Keyframe, domain::ErrorCode};

use wire_ext::*;

// use tracing::{info, warn};

#[cfg(target_os = "macos")]
use objc2::{
  encode::{Encode, Encoding},
  rc::Retained,
};
#[cfg(target_os = "macos")]
use objc2_core_foundation::{CGPoint, CGRect};
#[cfg(target_os = "macos")]
use objc2_core_video::{
  CVPixelBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight,
  CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress,
  CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVPixelFormatType_OneComponent8,
  kCVPixelFormatType_OneComponent32Float, kCVReturnSuccess,
};
#[cfg(target_os = "macos")]
use objc2_foundation::{NSArray, NSData, NSIndexSet, NSNotFound};
#[cfg(target_os = "macos")]
use objc2_vision::*;
#[cfg(target_os = "macos")]
use smol_str::{SmolStr, StrExt, ToSmolStr};

pub use options::*;

mod options;
// `wire_ext` is platform-independent — it bridges mediaschema wire
// types to richer ergonomic builders. Both the macOS Vision engine and
// the non-macOS stub use `ErrorInfoExt::new` to construct errors.
mod wire_ext;

#[cfg(target_os = "macos")]
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
struct SimdFloat4([f32; 4]);

#[cfg(target_os = "macos")]
unsafe impl Encode for SimdFloat4 {
  const ENCODING: Encoding = Encoding::Unknown;
}

#[cfg(target_os = "macos")]
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
struct SimdFloat4x4 {
  columns: [SimdFloat4; 4],
}

#[cfg(target_os = "macos")]
unsafe impl Encode for SimdFloat4x4 {
  // Clang reports @encode(simd_float4x4) as "{?=[4]}" because the vector element
  // encoding is intentionally opaque.
  const ENCODING: Encoding = Encoding::Struct("?", &[Encoding::Array(4, &Encoding::Unknown)]);
}

// ----- Vision → mediaschema coordinate conversion ---------------------------

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
#[cfg(target_os = "macos")]
#[inline]
fn clamp01(value: f32) -> f32 {
  debug_assert!(
    value.is_finite(),
    "clamp01 expects finite input; got {value}"
  );
  value.clamp(0.0, 1.0)
}

/// Convert a Vision-framework normalized bounding box (lower-left
/// origin, y grows up) into the mediaschema convention (top-left
/// origin, y grows down) and intersect it with the unit square
/// `[0, 1] × [0, 1]`.
///
/// The schema documents `apple-vision convention: floats in [0.0, 1.0],
/// origin top-left` (see `mediaschema::domain ... NormCoord`), while
/// `VNObservation::boundingBox` is documented as a normalized rect in
/// image coordinates where `(0,0)` is the lower-left corner. Vision is
/// empirically loose about staying inside `[0, 1]` — partially
/// off-screen detections can produce `origin.x < 0`,
/// `origin.x + width > 1`, etc., which the validated domain
/// `BoundingBox::try_new` would reject. We clamp every component and
/// return `None` if the resulting rectangle is degenerate
/// (zero-width or zero-height); the detection is then dropped at the
/// engine layer instead of poisoning downstream storage.
///
/// `standardize()` is assumed to have already been called on `rect`;
/// the input `size` is non-negative.
#[cfg(target_os = "macos")]
fn vision_bbox_to_schema(rect: CGRect) -> Option<BoundingBox> {
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
  Some(BoundingBox::new(left, top, width, height))
}

/// Flip a Vision normalized point's y axis to match mediaschema's
/// top-left origin and clamp both components into `[0.0, 1.0]`.
/// `BoundingBox`, `Point2D`, `BodyPoseJoint` (2-D), `FaceLandmarkPoint`,
/// and `DocumentSegment` corners all share the top-left convention (see
/// `NormCoord` doc-comment in mediaschema). 3-D joints
/// (`BodyPose3DJoint`) are model-space metres and are NOT flipped or
/// clamped.
///
/// Returns `None` when either input coordinate is non-finite. A `NaN`
/// or `±Inf` from a glitched Vision observation is geometrically
/// meaningless and previously sanitised to `0.0` via `clamp01`, which
/// fabricated edge-aligned coordinates indistinguishable from real
/// detections. The caller decides whether a single bad point drops
/// the entire detection (e.g. a document quad without all four
/// corners) or just the offending point (e.g. one bad joint among
/// many).
#[cfg(target_os = "macos")]
#[inline]
fn vision_point_to_schema(x: f64, y: f64) -> Option<(f32, f32)> {
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
#[cfg(target_os = "macos")]
#[inline]
fn finite_f32(v: f32) -> Option<f32> {
  if v.is_finite() { Some(v) } else { None }
}

/// Upper bound on a single mask payload (post-packing, 8 bits per
/// pixel) before we refuse to allocate. 64 MiB covers any sane image
/// resolution Apple Vision returns today (8K = ~33 MiB at 8 bits per
/// pixel) and prevents a runaway / corrupted `width * height` from
/// driving the worker process into the allocator's abort path.
#[cfg(target_os = "macos")]
const MAX_MASK_BYTES: usize = 64 * 1024 * 1024;

/// Allocate a zero-initialised packed mask buffer with bounded
/// `try_reserve_exact`. Returns `None` on either bound violation or
/// allocator failure — both surface to the caller as a dropped mask
/// detection rather than aborting the process.
#[cfg(target_os = "macos")]
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
/// Distinguishes three states explicitly:
/// - `Some(finite)` — Vision provided a real measurement; pass it
///   through.
/// - `Some(0.0)` — Vision did NOT provide a value (the underlying
///   `NSNumber?` was `None`). Map to `0.0` so the caller's threshold
///   comparison fails closed for any positive minimum.
/// - `None` — Vision provided a non-finite value (`NaN` / `±Inf`).
///   Caller MUST drop the detection: a non-finite reading is not a
///   real measurement, and substituting `0.0` would silently admit
///   the detection through any `min_capture_quality = 0.0`
///   configuration.
#[cfg(target_os = "macos")]
#[inline]
fn sanitize_capture_quality(raw: Option<f32>) -> Option<f32> {
  match raw {
    Some(v) => finite_f32(v),
    None => Some(0.0),
  }
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
#[cfg(target_os = "macos")]
#[inline]
fn sanitize_body_height_pair(
  raw_height: f32,
  measured_or_reference: BodyPose3DHeightEstimation,
) -> (f32, BodyPose3DHeightEstimation) {
  match finite_f32(raw_height) {
    Some(finite) => (finite, measured_or_reference),
    None => (0.0, BODY_POSE_3D_HEIGHT_ESTIMATION_UNKNOWN),
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
#[cfg(target_os = "macos")]
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
/// then route through [`vision_point_to_schema`] for the schema-side
/// top-left flip + `[0, 1]` clamp + finite check.
#[cfg(target_os = "macos")]
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
/// box that the validated domain `BoundingBox::try_new` rejects.
/// Callers should skip the pose detection on `None`; the joints alone
/// do not carry enough geometry to construct a valid box.
#[cfg(target_os = "macos")]
fn pose_bbox_from_joint_bounds(
  min_x: f32,
  min_y: f32,
  max_x: f32,
  max_y: f32,
) -> Option<BoundingBox> {
  if !(min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()) {
    return None;
  }
  let width = max_x - min_x;
  let height = max_y - min_y;
  if width <= 0.0 || height <= 0.0 {
    return None;
  }
  Some(BoundingBox::new(min_x, min_y, width, height))
}

/// Validate a raw Vision `confidence` value against the configured
/// per-request minimum and the wire/domain `Confidence` invariant
/// (finite, in `[0.0, 1.0]`). Returns `None` if the value is
/// non-finite, outside `[0, 1]`, or below `min` — the caller drops
/// the detection in that case. A simple `value < min` threshold
/// previously let `NaN` through (since every NaN comparison is
/// false) and accepted `>1.0` values, both of which mediaschema's
/// domain `Confidence::try_new` rejects.
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
struct CVPixelBufferLockGuard<'a> {
  buffer: &'a CVPixelBuffer,
  flags: CVPixelBufferLockFlags,
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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

// #[derive(Debug, Clone, Copy)]
// pub struct Service(());

// impl ThreadService for Service {
//   type Input = Request;
//   type Options = ServiceOptions;
//   type SpawnError = SpawnError;
//   type Handle = ThreadHandles<Self::Input>;

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   fn name() -> &'static str {
//     "apple-vision"
//   }

//   fn health_spec(options: &Self::Options) -> ThreadServiceHealthSpec {
//     ThreadServiceHealthSpec::new(options.num_workers.max(1), ServiceHealthConfig::default())
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   fn spawn(
//     options: Self::Options,
//     ctx: ThreadServiceContext,
//   ) -> Result<Self::Handle, Self::SpawnError>
//   where
//     Self: Sized,
//     Self::Handle: findit_service::MessageHandle<Self::Input>,
//   {
//     let (tx, rx) = unbounded::<Self::Input>();
//     let (shutdown, health_reporter, health_handle, health_config) = ctx.into_parts();
//     let mut handles = Vec::with_capacity(options.num_workers);

//     for idx in 0..options.num_workers {
//       let rx = rx.clone();
//       let shutdown = shutdown.clone();
//       let opts = options.clone();
//       let health = health_reporter.clone();
//       let handle = std::thread::Builder::new()
//         .name(format!("{}-{idx}", Self::name()))
//         .spawn(move || {
//           run_apple_vision_worker(
//             Self::name(),
//             idx,
//             rx,
//             shutdown,
//             opts,
//             health,
//             health_config.heartbeat_interval(),
//           )
//         })
//         .map_err(|error| SpawnError::io("failed to spawn worker thread", error))?;
//       handles.push(handle);
//     }

//     Ok(ThreadHandles::with_named_service_health(
//       Self::name(),
//       tx,
//       handles,
//       Some(health_handle),
//     ))
//   }
// }

// impl ProviderIdentifier for Service {
//   const KEY: ProviderKey = ProviderKey::internal_after(
//     Lifecycle::Video(VideoLifecycle::KeyframeExtract),
//     Lifecycle::Video(VideoLifecycle::VisionAnalysis),
//     "apple-vision",
//   );
//   const IMPLEMENTATION_HASH: u64 = 0;
// }

// impl ProviderThreadService for Service {
//   const KIND: ProviderKind = ProviderKind::Standard;

//   type LifecycleInput = Request;
//   type LifecycleOutput = Reply;
// }

// /// Messages sent from processor tasks to the Apple Vision service.
// pub struct Request {
//   video_id: Id,
//   scene_id: Id,
//   keyframes: Arc<[Identified<Bytes>]>,
//   reply: Callback,
// }

// impl Request {
//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn new(
//     video_id: Id,
//     scene_id: Id,
//     keyframes: Arc<[Identified<Bytes>]>,
//     reply: Callback,
//   ) -> Self {
//     Self {
//       video_id,
//       scene_id,
//       keyframes,
//       reply,
//     }
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub const fn video_id(&self) -> Id {
//     self.video_id
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_video_id(&mut self, video_id: Id) -> &mut Self {
//     self.video_id = video_id;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_video_id(mut self, video_id: Id) -> Self {
//     self.set_video_id(video_id);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub const fn scene_id(&self) -> Id {
//     self.scene_id
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_scene_id(&mut self, scene_id: Id) -> &mut Self {
//     self.scene_id = scene_id;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_scene_id(mut self, scene_id: Id) -> Self {
//     self.set_scene_id(scene_id);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn keyframes(&self) -> &[Identified<Bytes>] {
//     &self.keyframes
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_keyframes(&mut self, keyframes: Arc<[Identified<Bytes>]>) -> &mut Self {
//     self.keyframes = keyframes;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_keyframes(mut self, keyframes: Arc<[Identified<Bytes>]>) -> Self {
//     self.set_keyframes(keyframes);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn reply(&self) -> &Callback {
//     &self.reply
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_reply(&mut self, reply: Callback) -> &mut Self {
//     self.reply = reply;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_reply(mut self, reply: Callback) -> Self {
//     self.set_reply(reply);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn into_parts(self) -> (Id, Id, Arc<[Identified<Bytes>]>, Callback) {
//     (self.video_id, self.scene_id, self.keyframes, self.reply)
//   }
// }

// pub struct Reply {
//   scene_id: Id,
//   results: Vec<Keyframe>,
//   errors: Vec<ErrorInfo>,
// }

// impl Reply {
//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn new(scene_id: Id, results: Vec<Keyframe>, errors: Vec<ErrorInfo>) -> Self {
//     Self {
//       scene_id,
//       results,
//       errors,
//     }
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub const fn scene_id(&self) -> Id {
//     self.scene_id
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_scene_id(&mut self, scene_id: Id) -> &mut Self {
//     self.scene_id = scene_id;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_scene_id(mut self, scene_id: Id) -> Self {
//     self.set_scene_id(scene_id);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn results(&self) -> &[Keyframe] {
//     &self.results
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_results(&mut self, results: Vec<Keyframe>) -> &mut Self {
//     self.results = results;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_results(mut self, results: Vec<Keyframe>) -> Self {
//     self.set_results(results);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn errors(&self) -> &[ErrorInfo] {
//     &self.errors
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_errors(&mut self, errors: Vec<ErrorInfo>) -> &mut Self {
//     self.errors = errors;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_errors(mut self, errors: Vec<ErrorInfo>) -> Self {
//     self.set_errors(errors);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn into_parts(self) -> (Id, Vec<Keyframe>, Vec<ErrorInfo>) {
//     (self.scene_id, self.results, self.errors)
//   }
// }

// fn handle_message(worker_id: usize, analyzer: &VisionAnalyzer, request: Request) {
//   let (video_id, scene_id, keyframes, reply) = request.into_parts();
//   let svc = Service::name();

//   #[cfg(feature = "tracing")]
//   tracing::info!(service = svc, worker = worker_id, video_id = %video_id, scene_id = %scene_id, "analyzing scene");

//   let mut results = Vec::with_capacity(keyframes.len());
//   let mut errors = Vec::new();

//   for keyframe in keyframes.iter() {
//     match analyzer.analyze_keyframe(scene_id, keyframe.id(), keyframe.data()) {
//       Ok(r) => results.push(r),
//       Err(e) => {
//         #[cfg(feature = "tracing")]
//         tracing::warn!(
//           service = svc,
//           worker = worker_id,
//           video_id = %video_id,
//           keyframe_id = %keyframe.id(),
//           err = %e,
//           "Apple Vision analysis failed"
//         );
//         errors.push(apple_vision_keyframe_error(keyframe.id(), e));
//       }
//     }
//   }

//   reply(Reply::new(scene_id, results, errors));
// }

/// Apple Vision analyzer — one per worker thread.
///
/// Construct one [`VisionAnalyzer`] per worker thread via
/// [`VisionAnalyzer::new`]. The analyzer owns retained `VNRequest`
/// Objective-C objects that carry per-call state across
/// `performRequests` / `results()`, so they are *not* safe to share
/// across threads or clone. The upcoming service-framework layer
/// constructs one fresh analyzer per worker rather than cloning a
/// single shared instance — `Clone` is intentionally not implemented to
/// make that contract a compile-time error.
#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct VisionAnalyzer {
  opts: ServiceOptions,
  requests: VisionRequests,
}

#[cfg(target_os = "macos")]
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
  feature_print: Retained<VNGenerateImageFeaturePrintRequest>,
}

fn apple_vision_error(code: ErrorCode, message: impl Into<String>) -> ErrorInfo {
  ErrorInfo::new(code, message.into())
}

// Used by the (currently commented) service-framework `handle_message` plumbing
// — kept here so we don't have to rewrite the error path when the service
// block is re-enabled.
#[allow(dead_code)]
fn apple_vision_keyframe_error(keyframe_id: Id, error: ErrorInfo) -> ErrorInfo {
  apple_vision_error(
    error.code(),
    format!("keyframe {:?}: {}", keyframe_id, error.message()),
  )
}

#[cfg(target_os = "macos")]
impl VisionRequests {
  fn new(opts: ServiceOptions) -> Self {
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
      hand_pose.setMaximumHandCount(opts.hand_pose().maximum_hand_count());
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

      let feature_print = VNGenerateImageFeaturePrintRequest::new();
      feature_print.setRevision(VNGenerateImageFeaturePrintRequestRevision2);

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
        feature_print,
      }
    }
  }

  fn perform(&self, handler: &VNSequenceRequestHandler, data: &NSData) -> Result<(), ErrorInfo> {
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
        Retained::cast_unchecked::<VNRequest>(self.feature_print.clone()),
      ]);

      handler
        .performRequests_onImageData_error(&requests, data)
        .map_err(|e| {
          apple_vision_error(
            ErrorCode::AppleVisionRequestFailed,
            e.localizedDescription().to_string(),
          )
        })
    }
  }
}

#[cfg(target_os = "macos")]
impl VisionAnalyzer {
  /// Creates a new Apple Vision analyzer with the specified options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(opts: ServiceOptions) -> Self {
    Self {
      requests: VisionRequests::new(opts.clone()),
      opts,
    }
  }

  #[cfg(feature = "tracing")]
  #[allow(dead_code)] // called from the (currently commented) service-framework block
  fn log_request_revisions(&self, svc: &'static str, worker_id: usize) {
    unsafe {
      tracing::info!(
        service = svc,
        worker = worker_id,
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
        feature_print_rev = self.requests.feature_print.revision(),
        "initialized pinned Apple Vision request revisions"
      );
    }
  }

  /// Run every Apple Vision request configured at construction time against
  /// the supplied JPEG bytes and gather the resulting detections into a
  /// fully-populated [`Keyframe`].
  ///
  /// `scene_id` and `keyframe_id` are attached verbatim to the returned
  /// `Keyframe`; the engine does not derive or generate identifiers itself.
  pub fn analyze_keyframe(
    &self,
    scene_id: Id,
    keyframe_id: Id,
    jpeg_data: &[u8],
  ) -> Result<Keyframe, ErrorInfo> {
    objc2::rc::autoreleasepool(|_| {
      let ns_data = NSData::with_bytes(jpeg_data);
      let handler = unsafe { VNSequenceRequestHandler::new() };
      self.requests.perform(&handler, &ns_data)?;

      Ok(
        Keyframe::default()
          .with_id(keyframe_id)
          .with_scene_id(scene_id)
          .with_classifications(self.extract_classifications())
          .with_humans(
            HumanAnalysis::new()
              .with_subjects(self.extract_human_subjects())
              .with_faces(self.extract_faces())
              .with_face_rectangles(self.extract_face_rectangles())
              .with_face_landmarks(self.extract_face_landmarks())
              .with_body_poses(self.extract_body_poses())
              .with_hand_poses(self.extract_hand_poses())
              .with_body_poses_3d(self.extract_body_poses_3d())
              .with_instance_masks(self.extract_person_instance_masks())
              .with_segmentation_masks(self.extract_person_segmentation_masks()),
          )
          .with_animals(
            AnimalAnalysis::new()
              .with_subjects(self.extract_animal_subjects())
              .with_body_poses(self.extract_animal_body_poses()),
          )
          .with_text_detections(self.extract_text_detections())
          .with_barcodes(self.extract_barcodes())
          .with_attention_saliency(self.extract_attention_saliency())
          .with_objectness_saliency(self.extract_objectness_saliency())
          .with_horizon(self.extract_horizon())
          .with_document_segments(self.extract_document_segments())
          .with_feature_print(self.extract_feature_print())
          .with_aesthetics(self.extract_aesthetics()),
      )
    })
  }

  fn extract_classifications(&self) -> Vec<ClassificationDetection> {
    let opts = self.opts.classifications();
    let Some(results) = (unsafe { self.requests.classify.results() }) else {
      return Vec::new();
    };

    let mut tags = Vec::with_capacity(results.len().min(opts.max_results()));
    for obs in results.iter() {
      let Some(confidence) =
        sanitize_confidence(unsafe { obs.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      let label = normalize_classification_label(unsafe { obs.identifier() }.to_smolstr());
      if !label.is_empty() {
        tags.push(ClassificationDetection::new(label, confidence));
        if tags.len() >= opts.max_results() {
          break;
        }
      }
    }

    tags
  }

  fn extract_faces(&self) -> Vec<FaceDetection> {
    let Some(results) = (unsafe { self.requests.face_quality.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.face_capture();

    let mut faces = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let Some(confidence) =
        sanitize_confidence(unsafe { obs.confidence() }, opts.min_confidence())
      else {
        continue;
      };
      // See `sanitize_capture_quality` for the three-state policy
      // (absent → 0.0, finite → pass-through, non-finite → drop).
      let Some(capture_quality) =
        sanitize_capture_quality(unsafe { obs.faceCaptureQuality() }.map(|q| q.floatValue()))
      else {
        continue;
      };
      if capture_quality < opts.min_capture_quality() {
        continue;
      }

      let Some(bbox) = vision_bbox_to_schema(unsafe { obs.boundingBox() }.standardize()) else {
        continue;
      };
      faces.push(
        FaceDetection::default()
          .with_bbox(bbox)
          .with_confidence(confidence)
          .with_capture_quality(capture_quality)
          .with_roll(
            unsafe { obs.roll() }
              .map(|v| v.floatValue())
              .and_then(finite_f32)
              .unwrap_or(0.0),
          )
          .with_yaw(
            unsafe { obs.yaw() }
              .map(|v| v.floatValue())
              .and_then(finite_f32)
              .unwrap_or(0.0),
          )
          .with_pitch(
            unsafe { obs.pitch() }
              .map(|v| v.floatValue())
              .and_then(finite_f32)
              .unwrap_or(0.0),
          ),
      );
    }

    faces
  }

  fn extract_face_rectangles(&self) -> Vec<FaceDetection> {
    let Some(results) = (unsafe { self.requests.face_rectangles.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.face_rectangles();

    let mut faces = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let Some(confidence) =
        sanitize_confidence(unsafe { obs.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      let Some(bbox) = vision_bbox_to_schema(unsafe { obs.boundingBox() }.standardize()) else {
        continue;
      };
      faces.push(
        FaceDetection::default()
          .with_bbox(bbox)
          .with_confidence(confidence)
          .with_roll(
            unsafe { obs.roll() }
              .map(|v| v.floatValue())
              .and_then(finite_f32)
              .unwrap_or(0.0),
          )
          .with_yaw(
            unsafe { obs.yaw() }
              .map(|v| v.floatValue())
              .and_then(finite_f32)
              .unwrap_or(0.0),
          )
          .with_pitch(
            unsafe { obs.pitch() }
              .map(|v| v.floatValue())
              .and_then(finite_f32)
              .unwrap_or(0.0),
          ),
      );
    }

    faces
  }

  fn extract_face_landmarks(&self) -> Vec<FaceLandmarksDetection> {
    let Some(results) = (unsafe { self.requests.face_landmarks.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.face_landmarks();

    let mut detections = Vec::with_capacity(results.len());
    for obs in results.iter() {
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

      let regions = extract_face_landmark_regions(&landmarks, face_rect_vision);
      if regions.len() < opts.min_region_count() {
        continue;
      }

      let Some(bbox) = vision_bbox_to_schema(face_rect_vision) else {
        continue;
      };
      detections.push(FaceLandmarksDetection::new(bbox, confidence, regions));
    }

    detections
  }

  fn extract_human_subjects(&self) -> Vec<SubjectDetection> {
    let Some(results) = (unsafe { self.requests.human_rectangles.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.human_subjects();

    let mut humans = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let Some(confidence) =
        sanitize_confidence(unsafe { obs.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      let Some(bbox) = vision_bbox_to_schema(unsafe { obs.boundingBox() }.standardize()) else {
        continue;
      };
      humans.push(SubjectDetection::new(
        SmolStr::from("person"),
        confidence,
        bbox,
      ));
    }

    humans
  }

  fn extract_body_poses(&self) -> Vec<BodyPoseDetection> {
    let Some(results) = (unsafe { self.requests.body_pose.results() }) else {
      return Vec::new();
    };

    let mut body_poses = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let Ok(points_by_joint) = (unsafe {
        obs.recognizedPointsForJointsGroupName_error(VNHumanBodyPoseObservationJointsGroupNameAll)
      }) else {
        continue;
      };

      let (joint_names, points) = points_by_joint.to_vecs();
      let mut joints = Vec::with_capacity(points.len());
      let mut min_x = f32::INFINITY;
      let mut min_y = f32::INFINITY;
      let mut max_x = f32::NEG_INFINITY;
      let mut max_y = f32::NEG_INFINITY;

      for (joint_name, point) in joint_names.into_iter().zip(points) {
        let name = joint_name.to_smolstr();
        if name.is_empty() {
          continue;
        }

        // Vision normalized points are lower-left origin; flip y for the
        // top-left schema convention before recording the joint or
        // deriving the bbox. A non-finite raw coordinate is dropped at
        // the source — partial-joint lists are valid for body pose so
        // we skip just this joint, not the whole pose.
        let Some((x, y)) = vision_point_to_schema(unsafe { point.x() }, unsafe { point.y() })
        else {
          continue;
        };
        let Some(confidence) = sanitize_confidence(
          unsafe { point.confidence() },
          self.opts.body_pose().min_joint_confidence(),
        ) else {
          continue;
        };

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);

        joints.push(BodyPoseJoint::new(name, x, y, confidence));
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
      body_poses.push(BodyPoseDetection::new(bbox, pose_confidence, joints));
    }

    body_poses
  }

  fn extract_body_poses_3d(&self) -> Vec<BodyPose3DDetection> {
    catch_unwind(AssertUnwindSafe(|| {
      let Some(results) = (unsafe { self.requests.body_pose_3d.results() }) else {
        return Vec::new();
      };
      let Some(group_name) = (unsafe { VNHumanBodyPose3DObservationJointsGroupNameAll }) else {
        return Vec::new();
      };

      let mut body_poses = Vec::with_capacity(results.len());
      for obs in results.iter() {
        let Ok(points_by_joint) =
          (unsafe { obs.recognizedPointsForJointsGroupName_error(group_name) })
        else {
          continue;
        };

        let (joint_names, points) = points_by_joint.to_vecs();
        let mut joints = Vec::with_capacity(points.len());

        for (joint_name, point) in joint_names.into_iter().zip(points) {
          let name = joint_name.to_smolstr();
          if name.is_empty() {
            continue;
          }

          let Some((x, y, z)) = extract_body_pose_3d_coordinates(&point) else {
            continue;
          };
          let raw_confidence: f32 = unsafe { objc2::msg_send![&*point, confidence] };
          let Some(confidence) = sanitize_confidence(
            raw_confidence,
            self.opts.body_pose_3d().min_joint_confidence(),
          ) else {
            continue;
          };

          joints.push(BodyPose3DJoint::new(name, x, y, z, confidence));
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
        body_poses.push(BodyPose3DDetection::new(
          pose_confidence,
          body_height,
          height_estimation,
          joints,
        ));
      }

      body_poses
    }))
    .unwrap_or_else(|_| {
      #[cfg(feature = "tracing")]
      tracing::warn!("caught panic while extracting human body pose 3D; returning empty result");
      Vec::new()
    })
  }

  fn extract_hand_poses(&self) -> Vec<HandPoseDetection> {
    let Some(results) = (unsafe { self.requests.hand_pose.results() }) else {
      return Vec::new();
    };

    let mut hand_poses = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let Ok(points_by_joint) = (unsafe {
        obs.recognizedPointsForJointsGroupName_error(VNHumanHandPoseObservationJointsGroupNameAll)
      }) else {
        continue;
      };

      let (joint_names, points) = points_by_joint.to_vecs();
      let mut joints = Vec::with_capacity(points.len());
      let mut min_x = f32::INFINITY;
      let mut min_y = f32::INFINITY;
      let mut max_x = f32::NEG_INFINITY;
      let mut max_y = f32::NEG_INFINITY;

      for (joint_name, point) in joint_names.into_iter().zip(points) {
        let name = joint_name.to_smolstr();
        if name.is_empty() {
          continue;
        }

        // Vision normalized points are lower-left origin; flip y for
        // the top-left schema convention. A non-finite raw coordinate
        // is dropped at the source — partial-joint hand lists are
        // valid so we skip only this joint.
        let Some((x, y)) = vision_point_to_schema(unsafe { point.x() }, unsafe { point.y() })
        else {
          continue;
        };
        let Some(confidence) = sanitize_confidence(
          unsafe { point.confidence() },
          self.opts.hand_pose().min_joint_confidence(),
        ) else {
          continue;
        };

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);

        joints.push(BodyPoseJoint::new(name, x, y, confidence));
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
      hand_poses.push(HandPoseDetection::new(
        bbox,
        pose_confidence,
        map_hand_chirality(unsafe { obs.chirality() }),
        joints,
      ));
    }

    hand_poses
  }

  fn extract_person_instance_masks(&self) -> Vec<PersonInstanceMaskDetection> {
    let Some(results) = (unsafe { self.requests.person_instance_mask.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.person_instance_masks();

    let mut masks = Vec::new();
    for observation in results.iter() {
      let Some(confidence) =
        sanitize_confidence(unsafe { observation.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      let instances = unsafe { observation.allInstances() };
      let mut instance_index = instances.firstIndex();
      let mut emitted = 0usize;
      while instance_index != NSNotFound as usize {
        if emitted >= opts.max_instances_per_observation() {
          break;
        }

        let selected_instances = NSIndexSet::indexSetWithIndex(instance_index);
        let Ok(mask_buffer) =
          (unsafe { observation.generateMaskForInstances_error(&selected_instances) })
        else {
          instance_index = instances.indexGreaterThanIndex(instance_index);
          continue;
        };

        let Some((bbox, dimensions, data)) = copy_instance_mask_buffer(&mask_buffer) else {
          instance_index = instances.indexGreaterThanIndex(instance_index);
          continue;
        };

        // `instance_index` is a `usize` from `NSIndexSet`; the wire
        // type stores `u32`. Saturating to `0` would silently merge
        // distinct instances — reject and continue instead.
        let Ok(wire_instance_index) = u32::try_from(instance_index) else {
          instance_index = instances.indexGreaterThanIndex(instance_index);
          continue;
        };

        masks.push(PersonInstanceMaskDetection::new(
          bbox,
          confidence,
          wire_instance_index,
          dimensions,
          data,
        ));
        emitted += 1;

        instance_index = instances.indexGreaterThanIndex(instance_index);
      }
    }

    masks
  }

  fn extract_person_segmentation_masks(&self) -> Vec<PersonSegmentationMask> {
    let Some(results) = (unsafe { self.requests.person_segmentation.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.person_segmentation_masks();

    let mut masks = Vec::with_capacity(results.len());
    for observation in results.iter() {
      let Some(confidence) =
        sanitize_confidence(unsafe { observation.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      let pixel_buffer = unsafe { observation.pixelBuffer() };
      let Some((bbox, dimensions, data)) = copy_instance_mask_buffer(&pixel_buffer) else {
        continue;
      };

      masks.push(PersonSegmentationMask::new(
        bbox, confidence, dimensions, data,
      ));
    }

    masks
  }

  fn extract_animal_subjects(&self) -> Vec<SubjectDetection> {
    unsafe {
      let Some(results) = self.requests.animals.results() else {
        return Vec::new();
      };

      let mut animals = Vec::new();
      for obs in results.iter() {
        let labels = obs.labels();
        for label in labels.iter() {
          let Some(confidence) =
            sanitize_confidence(label.confidence(), self.opts.animals().min_confidence())
          else {
            continue;
          };
          let id = label.identifier().to_smolstr();
          if !id.is_empty()
            && let Some(bbox) = vision_bbox_to_schema(obs.boundingBox().standardize())
          {
            animals.push(SubjectDetection::new(id, confidence, bbox));
          }
        }
      }

      animals
    }
  }

  fn extract_animal_body_poses(&self) -> Vec<BodyPoseDetection> {
    let Some(results) = (unsafe { self.requests.animal_body_pose.results() }) else {
      return Vec::new();
    };
    let Some(group_name) = (unsafe { VNAnimalBodyPoseObservationJointsGroupNameAll }) else {
      return Vec::new();
    };

    let mut body_poses = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let Ok(points_by_joint) =
        (unsafe { obs.recognizedPointsForJointsGroupName_error(group_name) })
      else {
        continue;
      };

      let (joint_names, points) = points_by_joint.to_vecs();
      let mut joints = Vec::with_capacity(points.len());
      let mut min_x = f32::INFINITY;
      let mut min_y = f32::INFINITY;
      let mut max_x = f32::NEG_INFINITY;
      let mut max_y = f32::NEG_INFINITY;

      for (joint_name, point) in joint_names.into_iter().zip(points) {
        let name = joint_name.to_smolstr();
        if name.is_empty() {
          continue;
        }

        // Vision normalized points are lower-left origin; flip y for
        // the top-left schema convention. A non-finite raw coordinate
        // is dropped at the source — partial-joint animal-pose lists
        // are valid so we skip only this joint.
        let Some((x, y)) = vision_point_to_schema(unsafe { point.x() }, unsafe { point.y() })
        else {
          continue;
        };
        let Some(confidence) = sanitize_confidence(
          unsafe { point.confidence() },
          self.opts.animal_pose().min_joint_confidence(),
        ) else {
          continue;
        };

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);

        joints.push(BodyPoseJoint::new(name, x, y, confidence));
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
      body_poses.push(BodyPoseDetection::new(bbox, pose_confidence, joints));
    }

    body_poses
  }

  fn extract_text_detections(&self) -> Vec<TextDetection> {
    let Some(results) = self.requests.text.results() else {
      return Vec::new();
    };

    let mut text_detections = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let candidates = obs.topCandidates(self.opts.text().max_candidates_per_observation());
      for candidate in candidates.iter() {
        let text = candidate.string().to_smolstr();
        if text.len() < self.opts.text().min_text_len() {
          continue;
        }
        let Some(confidence) = sanitize_confidence(candidate.confidence(), 0.0) else {
          continue;
        };
        if let Some(bbox) = vision_bbox_to_schema(unsafe { obs.boundingBox() }.standardize()) {
          text_detections.push(TextDetection::new(text, confidence, bbox));
        }
      }
    }
    text_detections
  }

  fn extract_barcodes(&self) -> Vec<BarcodeDetection> {
    let Some(results) = (unsafe { self.requests.barcodes.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.barcodes();

    let mut barcodes = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let Some(confidence) =
        sanitize_confidence(unsafe { obs.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      if let Some(payload) = unsafe { obs.payloadStringValue() } {
        let s = payload.to_smolstr();
        if s.len() >= opts.min_payload_len()
          && let Some(bbox) = vision_bbox_to_schema(unsafe { obs.boundingBox() }.standardize())
        {
          let symbology = unsafe { obs.symbology() }.to_smolstr();
          barcodes.push(BarcodeDetection::new(s, symbology, confidence, bbox));
        }
      }
    }
    barcodes
  }

  fn extract_attention_saliency(&self) -> Vec<SaliencyRegion> {
    self.extract_saliency_regions(
      unsafe { self.requests.attention_saliency.results() },
      self.opts.attention_saliency(),
    )
  }

  fn extract_objectness_saliency(&self) -> Vec<SaliencyRegion> {
    self.extract_saliency_regions(
      unsafe { self.requests.objectness_saliency.results() },
      self.opts.objectness_saliency(),
    )
  }

  fn extract_saliency_regions(
    &self,
    observations: Option<Retained<NSArray<VNSaliencyImageObservation>>>,
    opts: AppleVisionSaliencyOptions,
  ) -> Vec<SaliencyRegion> {
    let Some(observations) = observations else {
      return Vec::new();
    };

    let mut regions = Vec::new();
    for observation in observations.iter() {
      let Some(objects) = (unsafe { observation.salientObjects() }) else {
        continue;
      };
      for object in objects.iter().take(opts.max_regions()) {
        let Some(confidence) =
          sanitize_confidence(unsafe { object.confidence() }, opts.min_confidence())
        else {
          continue;
        };

        let Some(bbox) = vision_bbox_to_schema(unsafe { object.boundingBox() }.standardize())
        else {
          continue;
        };
        regions.push(SaliencyRegion::new(bbox, confidence));
      }
    }
    regions
  }

  fn extract_horizon(&self) -> HorizonInfo {
    let Some(results) = (unsafe { self.requests.horizon.results() }) else {
      return HorizonInfo::default();
    };
    let Some(observation) = results.iter().next() else {
      return HorizonInfo::default();
    };
    let Some(confidence) = sanitize_confidence(
      unsafe { observation.confidence() },
      self.opts.horizon().min_confidence(),
    ) else {
      return HorizonInfo::default();
    };

    // Drop the horizon detection entirely if the angle is non-finite —
    // there is no sensible default for a horizon line and downstream
    // visualisation would render a bogus tilt.
    let Some(angle) = finite_f32(unsafe { observation.angle() } as f32) else {
      return HorizonInfo::default();
    };
    HorizonInfo::new(angle, confidence)
  }

  fn extract_document_segments(&self) -> Vec<DocumentSegment> {
    let Some(results) = (unsafe { self.requests.document_segments.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.document_segments();

    let mut segments = Vec::with_capacity(results.len());
    for observation in results.iter() {
      if segments.len() >= opts.max_segments() {
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
        vision_point_to_schema(
          unsafe { observation.topLeft() }.x,
          unsafe { observation.topLeft() }.y,
        ),
        vision_point_to_schema(
          unsafe { observation.topRight() }.x,
          unsafe { observation.topRight() }.y,
        ),
        vision_point_to_schema(
          unsafe { observation.bottomLeft() }.x,
          unsafe { observation.bottomLeft() }.y,
        ),
        vision_point_to_schema(
          unsafe { observation.bottomRight() }.x,
          unsafe { observation.bottomRight() }.y,
        ),
      ) else {
        continue;
      };

      // Even after per-corner clamping, the resulting quad can be
      // degenerate (coincident corners, zero shoelace area, or
      // self-intersecting) when Vision returned an off-screen segment
      // or near-collinear corners. mediaschema's domain
      // `DocumentSegment::try_new` runs the same geometry guards
      // (collapsed corners, zero area, bow-tie / inconsistent winding)
      // that downstream consumers will apply, so we validate via that
      // constructor and only emit the wire segment on success.
      if mediaschema::domain::aggregates::video::DocumentSegment::try_new(
        top_left,
        top_right,
        bottom_right,
        bottom_left,
        confidence,
      )
      .is_err()
      {
        continue;
      }

      segments.push(
        DocumentSegment::default()
          .with_top_left(top_left)
          .with_top_right(top_right)
          .with_bottom_left(bottom_left)
          .with_bottom_right(bottom_right)
          .with_confidence(confidence),
      );
    }

    segments
  }

  fn extract_aesthetics(&self) -> Aesthetics {
    let Some(results) = (unsafe { self.requests.aesthetics.results() }) else {
      return Aesthetics::default();
    };
    let Some(obs) = results.iter().next() else {
      return Aesthetics::default();
    };
    // `NaN < threshold` would fail open. Force a finite check at the
    // gate so a glitched aesthetics score collapses to the default
    // (no detection) instead of being silently admitted to the wire.
    let Some(overall_score) = finite_f32(unsafe { obs.overallScore() }) else {
      return Aesthetics::default();
    };
    if overall_score < self.opts.aesthetics().min_overall_score() {
      return Aesthetics::default();
    }

    Aesthetics::new(overall_score, unsafe { obs.isUtility() })
  }

  fn extract_feature_print(&self) -> FeaturePrint {
    let Some(results) = (unsafe { self.requests.feature_print.results() }) else {
      return FeaturePrint::default();
    };
    let Some(obs) = results.iter().next() else {
      return FeaturePrint::default();
    };
    let count = unsafe { obs.elementCount() };
    if count < self.opts.feature_print().min_element_count() {
      return FeaturePrint::default();
    }

    let ns_data = unsafe { obs.data() };
    let len = ns_data.len();
    let ptr: *const std::ffi::c_void = unsafe { objc2::msg_send![&*ns_data, bytes] };
    if ptr.is_null() || len == 0 {
      return FeaturePrint::default();
    }

    let data = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }.to_vec();
    let element_type = u32::try_from(unsafe { obs.elementType() }.0).unwrap_or_default();
    FeaturePrint::new(Bytes::from(data), element_type)
  }
}

#[cfg(target_os = "macos")]
fn normalize_classification_label(label: SmolStr) -> SmolStr {
  label.trim().to_ascii_lowercase_smolstr()
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
fn map_hand_chirality(chirality: VNChirality) -> HandChirality {
  match chirality {
    VNChirality::Left => HAND_CHIRALITY_LEFT,
    VNChirality::Right => HAND_CHIRALITY_RIGHT,
    _ => HAND_CHIRALITY_UNKNOWN,
  }
}

/// Extract every named face-landmark region, projecting each point
/// from face-bbox-relative coordinates into image-normalized
/// coordinates (Vision lower-left) via `face_bbox_vision` before the
/// caller-side schema flip. Without this projection a non-full-frame
/// face emits landmarks in the wrong place but still passes `[0, 1]`
/// validation.
#[cfg(target_os = "macos")]
fn extract_face_landmark_regions(
  landmarks: &VNFaceLandmarks2D,
  face_bbox_vision: CGRect,
) -> Vec<FaceLandmarkRegion> {
  let mut regions = Vec::new();
  push_face_landmark_region(
    &mut regions,
    "allPoints",
    unsafe { landmarks.allPoints() },
    face_bbox_vision,
  );
  push_face_landmark_region(
    &mut regions,
    "faceContour",
    unsafe { landmarks.faceContour() },
    face_bbox_vision,
  );
  push_face_landmark_region(
    &mut regions,
    "leftEye",
    unsafe { landmarks.leftEye() },
    face_bbox_vision,
  );
  push_face_landmark_region(
    &mut regions,
    "rightEye",
    unsafe { landmarks.rightEye() },
    face_bbox_vision,
  );
  push_face_landmark_region(
    &mut regions,
    "leftEyebrow",
    unsafe { landmarks.leftEyebrow() },
    face_bbox_vision,
  );
  push_face_landmark_region(
    &mut regions,
    "rightEyebrow",
    unsafe { landmarks.rightEyebrow() },
    face_bbox_vision,
  );
  push_face_landmark_region(
    &mut regions,
    "nose",
    unsafe { landmarks.nose() },
    face_bbox_vision,
  );
  push_face_landmark_region(
    &mut regions,
    "noseCrest",
    unsafe { landmarks.noseCrest() },
    face_bbox_vision,
  );
  push_face_landmark_region(
    &mut regions,
    "medianLine",
    unsafe { landmarks.medianLine() },
    face_bbox_vision,
  );
  push_face_landmark_region(
    &mut regions,
    "outerLips",
    unsafe { landmarks.outerLips() },
    face_bbox_vision,
  );
  push_face_landmark_region(
    &mut regions,
    "innerLips",
    unsafe { landmarks.innerLips() },
    face_bbox_vision,
  );
  push_face_landmark_region(
    &mut regions,
    "leftPupil",
    unsafe { landmarks.leftPupil() },
    face_bbox_vision,
  );
  push_face_landmark_region(
    &mut regions,
    "rightPupil",
    unsafe { landmarks.rightPupil() },
    face_bbox_vision,
  );
  regions
}

#[cfg(target_os = "macos")]
fn push_face_landmark_region(
  regions: &mut Vec<FaceLandmarkRegion>,
  name: &'static str,
  region: Option<Retained<VNFaceLandmarkRegion2D>>,
  face_bbox_vision: CGRect,
) {
  let Some(region) = region else {
    return;
  };

  let point_count = unsafe { region.pointCount() };
  if point_count == 0 {
    return;
  }

  let points_ptr = unsafe { region.normalizedPoints() };
  if points_ptr.is_null() {
    return;
  }

  let points = unsafe { std::slice::from_raw_parts(points_ptr, point_count) };
  let points = points
    .iter()
    .filter_map(|point| {
      // Apple's convention: landmark points are normalized within the
      // face's normalized bbox (NOT the image). Project to image-
      // normalized Vision coordinates first, THEN route through
      // `vision_point_to_schema` for the top-left flip + clamp +
      // finite check. A non-finite raw or projected component drops
      // only the offending point; partial-point regions are still
      // meaningful.
      let projected = project_landmark_to_image(*point, face_bbox_vision);
      let (x, y) = vision_point_to_schema(projected.x, projected.y)?;
      Some(FaceLandmarkPoint::new(x, y))
    })
    .collect::<Vec<_>>();
  if points.is_empty() {
    return;
  }

  regions.push(FaceLandmarkRegion::new(name, points));
}

#[cfg(target_os = "macos")]
fn map_body_pose_3d_height_estimation(
  estimation: VNHumanBodyPose3DObservationHeightEstimation,
) -> BodyPose3DHeightEstimation {
  if estimation == VNHumanBodyPose3DObservationHeightEstimation::Measured {
    BODY_POSE_3D_HEIGHT_ESTIMATION_MEASURED
  } else if estimation == VNHumanBodyPose3DObservationHeightEstimation::Reference {
    BODY_POSE_3D_HEIGHT_ESTIMATION_REFERENCE
  } else {
    BODY_POSE_3D_HEIGHT_ESTIMATION_UNKNOWN
  }
}

/// Copy a Vision mask `CVPixelBuffer` into a packed `Bytes` payload plus
/// a normalized bounding box of the foreground.
///
/// The returned payload is **always** 8 bits per pixel
/// (`width * height` bytes); Vision's two supported source formats
/// (`OneComponent32Float`, `OneComponent8`) are both normalised to
/// canonical u8 at the boundary so downstream consumers don't have
/// to disambiguate from the [`Dimensions`] metadata alone. f32 input
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
#[cfg(target_os = "macos")]
fn copy_instance_mask_buffer(
  pixel_buffer: &CVPixelBuffer,
) -> Option<(BoundingBox, Dimensions, Bytes)> {
  let guard = CVPixelBufferLockGuard::lock(pixel_buffer, CVPixelBufferLockFlags::ReadOnly)?;
  copy_instance_mask_buffer_locked(guard.buffer())
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
/// from the [`Dimensions`] metadata alone. The f32 → u8 quantisation
/// is `(v.clamp(0.0, 1.0) * 255.0).round() as u8` with non-finite
/// inputs collapsed to `0` (background); see
/// [`process_mask_bytes_f32`] for the per-pixel logic.
#[cfg(target_os = "macos")]
#[allow(non_upper_case_globals)]
fn copy_instance_mask_buffer_locked(
  pixel_buffer: &CVPixelBuffer,
) -> Option<(BoundingBox, Dimensions, Bytes)> {
  let width = CVPixelBufferGetWidth(pixel_buffer);
  let height = CVPixelBufferGetHeight(pixel_buffer);
  if width == 0 || height == 0 {
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

  // SAFETY: `base_address` points at a buffer of at least
  // `bytes_per_row * height` bytes (Core Video contract); the buffer
  // is locked by the surrounding `CVPixelBufferLockGuard`. The
  // pre-validation above satisfies the `from_raw_parts` contract
  // (`total_src_len <= isize::MAX`) regardless of what Core Video
  // reports for the dimensions; the downstream bounded allocator
  // re-checks `width * height` against `MAX_MASK_BYTES`.
  let src = unsafe { std::slice::from_raw_parts(base_address, total_src_len) };

  // The wire `Dimensions` stores `u32`. A mask whose width or height
  // exceeds `u32::MAX` cannot be represented faithfully — we'd have
  // to saturate the dimensions to a value smaller than the actual
  // packed payload, which would silently desynchronise consumers
  // that size buffers from the metadata. Reject overflow here so the
  // detection is dropped rather than poisoning storage.
  let dim_width = u32::try_from(width).ok()?;
  let dim_height = u32::try_from(height).ok()?;

  let (bbox, packed) = match pixel_format {
    kCVPixelFormatType_OneComponent32Float => {
      process_mask_bytes_f32(width, height, bytes_per_row, src)?
    }
    kCVPixelFormatType_OneComponent8 => process_mask_bytes_u8(width, height, bytes_per_row, src)?,
    _ => return None,
  };

  Some((
    bbox,
    Dimensions::new(dim_width, dim_height),
    Bytes::from(packed),
  ))
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
#[cfg(target_os = "macos")]
fn process_mask_bytes_f32(
  width: usize,
  height: usize,
  bytes_per_row: usize,
  src: &[u8],
) -> Option<(BoundingBox, Vec<u8>)> {
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
    // degenerate bbox. The validated domain `BoundingBox::try_new`
    // rejects zero-extent boxes, so the previous
    // `BoundingBox::default()` fallback would poison downstream
    // conversion.
    return None;
  }
  let bbox = normalized_bbox_from_pixel_bounds(min_x, min_y, max_x, max_y, width, height);
  Some((bbox, packed))
}

/// Walk an `OneComponent8` mask, copy it tightly packed, and derive a
/// normalized foreground bbox. Returns `None` for an all-zero mask.
#[cfg(target_os = "macos")]
fn process_mask_bytes_u8(
  width: usize,
  height: usize,
  bytes_per_row: usize,
  src: &[u8],
) -> Option<(BoundingBox, Vec<u8>)> {
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
  let bbox = normalized_bbox_from_pixel_bounds(min_x, min_y, max_x, max_y, width, height);
  Some((bbox, packed))
}

/// Convert the foreground pixel bounds of a `CVPixelBuffer` mask into a
/// normalized [`BoundingBox`] in the top-left schema convention.
///
/// `CVPixelBuffer` rows are stored top-to-bottom in memory (row 0 is the
/// top of the image), so the natural mapping `min_y / height` is already
/// top-left and no y-flip is needed here.
#[cfg(target_os = "macos")]
fn normalized_bbox_from_pixel_bounds(
  min_x: usize,
  min_y: usize,
  max_x: usize,
  max_y: usize,
  width: usize,
  height: usize,
) -> BoundingBox {
  let x = min_x as f32 / width as f32;
  let y = min_y as f32 / height as f32;
  let w = (max_x + 1 - min_x) as f32 / width as f32;
  let h = (max_y + 1 - min_y) as f32 / height as f32;
  BoundingBox::new(x, y, w, h)
}

// ----- Non-macOS stub --------------------------------------------------------

/// Non-macOS stub for [`VisionAnalyzer`]. Apple's Vision.framework is
/// only available on macOS, so on every other target the analyzer
/// always reports an [`ErrorCode::AppleVisionFailed`] platform error
/// rather than producing detections. The README promises the crate
/// compiles cleanly on non-macOS targets so downstream workspaces can
/// keep `avanalyze` in their dep tree unconditionally; this stub is
/// what makes that promise true.
#[cfg(not(target_os = "macos"))]
#[derive(Debug)]
pub struct VisionAnalyzer {
  // Keep the options around so a future native cross-platform
  // engine can swap in here without breaking the public API.
  #[allow(dead_code)]
  opts: ServiceOptions,
}

#[cfg(not(target_os = "macos"))]
impl VisionAnalyzer {
  /// Construct a non-macOS stub analyzer. The configuration is
  /// retained but unused — every `analyze_keyframe` call returns
  /// [`ErrorCode::AppleVisionFailed`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(opts: ServiceOptions) -> Self {
    Self { opts }
  }

  /// Non-macOS stub: Apple's Vision.framework is only available on
  /// macOS, so this always returns
  /// [`ErrorCode::AppleVisionFailed`] with an explanatory message.
  /// `_jpeg_data` is ignored.
  pub fn analyze_keyframe(
    &self,
    _scene_id: Id,
    _keyframe_id: Id,
    _jpeg_data: &[u8],
  ) -> Result<Keyframe, ErrorInfo> {
    Err(apple_vision_error(
      ErrorCode::AppleVisionFailed,
      "Apple Vision.framework is only available on macOS",
    ))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Regression: `HumanAnalysis::with_body_poses_3d` previously dropped
  /// its input on the floor. The wire `HumanAnalysis.body_poses_3d`
  /// field has existed since the mediaschema mono-consolidation, so
  /// the setter must persist the provided detections. Platform-
  /// independent — the wire builder does not depend on Vision.
  #[test]
  fn body_poses_3d_survives_through_human_analysis() {
    use mediaschema::{BodyPose3DDetection, HumanAnalysis};
    let pose = BodyPose3DDetection::default();
    let analysis = HumanAnalysis::new().with_body_poses_3d(vec![pose]);
    assert_eq!(analysis.body_poses_3d.len(), 1);
  }

  /// Non-macOS `VisionAnalyzer` stub must report an Apple-Vision
  /// platform error on every `analyze_keyframe` call.
  #[cfg(not(target_os = "macos"))]
  #[test]
  fn non_macos_stub_reports_unavailable() {
    use mediaschema::{Id, domain::ErrorCode};
    let analyzer = VisionAnalyzer::new(ServiceOptions::new());
    let err = analyzer
      .analyze_keyframe(Id::default(), Id::default(), &[])
      .expect_err("stub must return Err");
    assert_eq!(err.code(), ErrorCode::AppleVisionFailed);
  }
}

#[cfg(all(test, target_os = "macos"))]
mod macos_tests {
  use super::*;
  use mediaschema::domain::aggregates::video::BoundingBox as DomainBoundingBox;
  use objc2_core_foundation::{CGPoint, CGRect, CGSize};

  /// `vision_bbox_to_schema` must flip y. A Vision rect of
  /// `(0.1, 0.2, 0.3, 0.4)` (lower-left origin) maps to
  /// `(0.1, 1.0 - (0.2 + 0.4), 0.3, 0.4)` = `(0.1, 0.4, 0.3, 0.4)`
  /// in the schema's top-left convention.
  #[test]
  fn vision_bbox_to_schema_flips_y() {
    let rect = CGRect::new(CGPoint::new(0.1, 0.2), CGSize::new(0.3, 0.4));
    let bbox = vision_bbox_to_schema(rect).expect("in-range rect must clamp to itself");
    assert!((bbox.x - 0.1).abs() < 1e-6, "x: {}", bbox.x);
    assert!((bbox.y - 0.4).abs() < 1e-6, "y: {}", bbox.y);
    assert!((bbox.width - 0.3).abs() < 1e-6, "w: {}", bbox.width);
    assert!((bbox.height - 0.4).abs() < 1e-6, "h: {}", bbox.height);
  }

  /// Lock the flipped full-image result against the validating domain
  /// `BoundingBox::try_new` to ensure the components still satisfy the
  /// `[0, 1]` invariant after the flip.
  #[test]
  fn vision_bbox_to_schema_full_image_round_trip() {
    let rect = CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(1.0, 1.0));
    let bbox = vision_bbox_to_schema(rect).expect("unit rect must clamp to itself");
    assert_eq!(bbox.x, 0.0);
    assert_eq!(bbox.y, 0.0);
    assert_eq!(bbox.width, 1.0);
    assert_eq!(bbox.height, 1.0);
    DomainBoundingBox::try_new(bbox.x, bbox.y, bbox.width, bbox.height)
      .expect("full-image bbox stays valid after flip");
  }

  /// A Vision rect that spills off the right edge (`origin.x + width > 1`)
  /// must be clamped to the unit square. Domain `BoundingBox::try_new`
  /// would reject the un-clamped result, so without clamping a partially
  /// off-screen detection would poison downstream conversion.
  #[test]
  fn vision_bbox_clamps_right_spill() {
    // Vision rect: origin (0.8, 0.4), size (0.5, 0.2) — right edge at 1.3.
    let rect = CGRect::new(CGPoint::new(0.8, 0.4), CGSize::new(0.5, 0.2));
    let bbox = vision_bbox_to_schema(rect).expect("partial overlap must produce a bbox");
    // Clamped right edge is 1.0 → width = 0.2 (1.0 - 0.8).
    assert!((bbox.x - 0.8).abs() < 1e-6, "x: {}", bbox.x);
    assert!((bbox.width - 0.2).abs() < 1e-6, "w: {}", bbox.width);
    // y in schema space: 1.0 - (0.4 + 0.2) = 0.4 (in-range, no clamp).
    assert!((bbox.y - 0.4).abs() < 1e-6, "y: {}", bbox.y);
    assert!((bbox.height - 0.2).abs() < 1e-6, "h: {}", bbox.height);
    DomainBoundingBox::try_new(bbox.x, bbox.y, bbox.width, bbox.height)
      .expect("clamped bbox satisfies the [0,1] invariant");
  }

  /// A Vision rect that spills off the bottom (`origin.y < 0` in
  /// Vision = `y + height > 1` in schema) must be clamped to the unit
  /// square so the domain validator does not reject it.
  #[test]
  fn vision_bbox_clamps_bottom_spill() {
    // Vision rect: origin (0.1, -0.1), size (0.3, 0.4) — Vision bottom edge
    // at y = -0.1, top edge at y = 0.3.
    // Schema: top = 1.0 - (−0.1 + 0.4) = 0.7, bottom = 1.0 - (−0.1) = 1.1.
    let rect = CGRect::new(CGPoint::new(0.1, -0.1), CGSize::new(0.3, 0.4));
    let bbox = vision_bbox_to_schema(rect).expect("partial overlap must produce a bbox");
    // Bottom clamped to 1.0 → height = 1.0 - 0.7 = 0.3.
    assert!((bbox.x - 0.1).abs() < 1e-6, "x: {}", bbox.x);
    assert!((bbox.y - 0.7).abs() < 1e-6, "y: {}", bbox.y);
    assert!((bbox.width - 0.3).abs() < 1e-6, "w: {}", bbox.width);
    assert!((bbox.height - 0.3).abs() < 1e-6, "h: {}", bbox.height);
    DomainBoundingBox::try_new(bbox.x, bbox.y, bbox.width, bbox.height)
      .expect("clamped bbox satisfies the [0,1] invariant");
  }

  /// A Vision rect entirely outside the unit square must yield `None`
  /// so the detection is skipped rather than producing a degenerate
  /// wire bbox.
  #[test]
  fn vision_bbox_fully_offscreen_yields_none() {
    let rect = CGRect::new(CGPoint::new(1.5, 0.5), CGSize::new(0.3, 0.4));
    assert!(vision_bbox_to_schema(rect).is_none());
  }

  /// A Vision rect that intersects the unit square only at a single
  /// edge must yield `None` (the intersection has zero width).
  #[test]
  fn vision_bbox_edge_only_yields_none() {
    // Right edge at exactly x = 1.0, left edge at x = 1.0 — zero width.
    let rect = CGRect::new(CGPoint::new(1.0, 0.5), CGSize::new(0.0, 0.4));
    assert!(vision_bbox_to_schema(rect).is_none());
  }

  /// `NaN` from Vision (occasionally seen for off-image rects) must
  /// not propagate — the helper sanitises non-finite components to
  /// `0.0`. A `NaN` `origin.x` collapses left and right to 0.0, so the
  /// rectangle has zero width after clamping and is reported as
  /// `None` (the detection is dropped).
  #[test]
  fn vision_bbox_handles_nan_origin() {
    let rect = CGRect::new(CGPoint::new(f64::NAN, 0.0), CGSize::new(0.3, 0.4));
    assert!(vision_bbox_to_schema(rect).is_none());
  }

  /// `NaN` in a single size component still produces a usable
  /// rectangle iff the surviving edges enclose a non-zero area. A
  /// finite `origin.x`/`width` keeps the horizontal extent live; a
  /// `NaN` `origin.y` collapses the vertical extent to zero and the
  /// rectangle is dropped.
  #[test]
  fn vision_bbox_handles_nan_y_origin() {
    let rect = CGRect::new(CGPoint::new(0.1, f64::NAN), CGSize::new(0.3, 0.4));
    assert!(vision_bbox_to_schema(rect).is_none());
  }

  /// 2D points flip y AND clamp to `[0, 1]`. A Vision point that lands
  /// outside `[0, 1]` after the flip is clamped to the unit edge so
  /// downstream validation accepts it.
  #[test]
  fn vision_point_to_schema_flips_y_only() {
    let (x, y) = vision_point_to_schema(0.25, 0.75).expect("finite point");
    assert!((x - 0.25).abs() < 1e-6);
    assert!((y - 0.25).abs() < 1e-6);
  }

  /// Out-of-range Vision points clamp to `[0, 1]`.
  #[test]
  fn vision_point_to_schema_clamps_out_of_range() {
    let (x, y) = vision_point_to_schema(1.2, -0.3).expect("finite point");
    assert_eq!(x, 1.0);
    // `y = 1.0 - (-0.3) = 1.3` → clamped to 1.0.
    assert_eq!(y, 1.0);
  }

  /// Non-finite Vision points are rejected at the source: a `NaN`,
  /// `+Inf`, or `-Inf` in either component returns `None` so the
  /// caller can decide whether to drop the point or the whole
  /// detection. Previously the helper collapsed the bad component to
  /// `0.0` via `clamp01`, which fabricated edge-aligned coordinates
  /// that the domain validator could not distinguish from real
  /// detections.
  #[test]
  fn vision_point_to_schema_rejects_non_finite() {
    assert!(vision_point_to_schema(f64::NAN, 0.5).is_none());
    assert!(vision_point_to_schema(0.5, f64::NAN).is_none());
    assert!(vision_point_to_schema(f64::INFINITY, 0.5).is_none());
    assert!(vision_point_to_schema(0.5, f64::INFINITY).is_none());
    assert!(vision_point_to_schema(f64::NEG_INFINITY, 0.5).is_none());
    assert!(vision_point_to_schema(0.5, f64::NEG_INFINITY).is_none());
    // Finite path still works.
    assert!(vision_point_to_schema(0.1, 0.2).is_some());
  }

  /// A document quad with even one non-finite corner must be dropped
  /// in its entirety — a quad is geometrically meaningless without
  /// all four corners. This test mirrors the per-detection pattern
  /// the extractor uses (`let (Some(tl), Some(tr), Some(bl),
  /// Some(br)) = (...) else { continue; }`): if any corner returns
  /// `None`, the whole quad is rejected. Partial-corner emission
  /// would be a regression.
  #[test]
  fn document_quad_with_non_finite_corner_is_dropped() {
    // Three good corners + one NaN corner — overall the quad must
    // be dropped. We exercise each corner position to confirm the
    // "any None drops the whole quad" semantics.
    let good = (0.1_f64, 0.1_f64);
    let bad = (f64::NAN, 0.5_f64);

    for (tl, tr, bl, br) in [
      (bad, good, good, good),
      (good, bad, good, good),
      (good, good, bad, good),
      (good, good, good, bad),
    ] {
      let result = (
        vision_point_to_schema(tl.0, tl.1),
        vision_point_to_schema(tr.0, tr.1),
        vision_point_to_schema(bl.0, bl.1),
        vision_point_to_schema(br.0, br.1),
      );
      assert!(
        !matches!(result, (Some(_), Some(_), Some(_), Some(_))),
        "quad with non-finite corner survived: {result:?}",
      );
    }
  }

  /// `normalized_bbox_from_pixel_bounds` must NOT flip the y axis —
  /// `CVPixelBuffer` rows are top-to-bottom, so row 0 is the top edge
  /// and the natural mapping `min_y / height` is already in top-left
  /// convention.
  #[test]
  fn pixel_bounds_to_normalized_bbox_does_not_flip() {
    // A 100x100 mask with the foreground rectangle in rows 10..=29,
    // columns 5..=24. The expected normalized bbox is
    // `(5/100, 10/100, 20/100, 20/100)` in top-left convention.
    let bbox = normalized_bbox_from_pixel_bounds(5, 10, 24, 29, 100, 100);
    assert!((bbox.x - 0.05).abs() < 1e-6);
    assert!((bbox.y - 0.10).abs() < 1e-6);
    assert!((bbox.width - 0.20).abs() < 1e-6);
    assert!((bbox.height - 0.20).abs() < 1e-6);
  }

  /// An all-zero 8-bit mask must yield `None` so the caller skips the
  /// detection. Previously the buffer returned `Some` with
  /// `BoundingBox::default()` (a zero-extent box), which the domain
  /// `BoundingBox::try_new` would later reject.
  #[test]
  fn empty_8bit_mask_yields_none() {
    let src = vec![0u8; 4 * 4]; // 4×4 all-zero mask, tight stride.
    assert!(process_mask_bytes_u8(4, 4, 4, &src).is_none());
  }

  /// An all-zero 32-bit-float mask must also yield `None`. Same
  /// rationale as the 8-bit case.
  #[test]
  fn empty_32fp_mask_yields_none() {
    let src = vec![0u8; 4 * 4 * 4]; // 4×4 all-zero f32 mask.
    assert!(process_mask_bytes_f32(4, 4, 16, &src).is_none());
  }

  /// An 8-bit mask with one foreground pixel at row 1, col 2 of a
  /// 4×4 buffer must round-trip the bbox and the packed bytes.
  #[test]
  fn single_pixel_8bit_mask_round_trip() {
    let mut src = vec![0u8; 16];
    // Row 1, column 2 — stride 4.
    src[6] = 0xFF;
    let (bbox, packed) = process_mask_bytes_u8(4, 4, 4, &src).expect("foreground produces Some");
    assert!((bbox.x - 0.5).abs() < 1e-6, "x: {}", bbox.x);
    assert!((bbox.y - 0.25).abs() < 1e-6, "y: {}", bbox.y);
    assert!((bbox.width - 0.25).abs() < 1e-6, "w: {}", bbox.width);
    assert!((bbox.height - 0.25).abs() < 1e-6, "h: {}", bbox.height);
    // Packed bytes mirror the input (tight stride === input stride).
    assert_eq!(packed, src);
  }

  /// A 32-fp mask with one foreground pixel quantises to a single u8
  /// in the canonical 8-bit-per-pixel wire payload. `0.75 * 255 =
  /// 191.25 → 191` after `round()`. The packed buffer is `width *
  /// height` bytes, NOT `width * height * size_of::<f32>()`, since
  /// the f32 source is normalised to u8 at the boundary.
  #[test]
  fn single_pixel_32fp_mask_round_trip() {
    let mut src = vec![0u8; 4 * 4 * 4];
    let value: f32 = 0.75;
    let bytes = value.to_le_bytes();
    // Row 1, column 2 — 4 bytes per pixel, 16 bytes per row.
    let src_offset = 16 + 8;
    src[src_offset..src_offset + 4].copy_from_slice(&bytes);
    let (bbox, packed) = process_mask_bytes_f32(4, 4, 16, &src).expect("foreground produces Some");
    assert!((bbox.x - 0.5).abs() < 1e-6, "x: {}", bbox.x);
    assert!((bbox.y - 0.25).abs() < 1e-6, "y: {}", bbox.y);
    // Canonical 8-bit payload: 4×4 = 16 bytes.
    assert_eq!(packed.len(), 4 * 4);
    // Row 1, column 2 — 4 bytes per row in the u8 output, so offset = 4 + 2.
    let dst_offset = 4 + 2;
    assert_eq!(packed[dst_offset], 191, "0.75 → 191 after u8 quantisation");
    // Every other byte stays at 0 (background).
    for (idx, &b) in packed.iter().enumerate() {
      if idx != dst_offset {
        assert_eq!(b, 0, "background pixel {idx} must be 0");
      }
    }
  }

  /// f32 mask values at the canonical interior {0.0, 0.5, 1.0} plus a
  /// `NaN` background pixel must quantise to {0, 128, 255, 0} in the
  /// u8 wire payload. Pins the brief's documented mapping.
  #[test]
  fn f32_mask_quantises_canonical_values_and_nan() {
    // 4×1 row: [0.0, 0.5, 1.0, NaN].
    let mut src = vec![0u8; 4 * 4];
    src[0..4].copy_from_slice(&0.0_f32.to_le_bytes());
    src[4..8].copy_from_slice(&0.5_f32.to_le_bytes());
    src[8..12].copy_from_slice(&1.0_f32.to_le_bytes());
    src[12..16].copy_from_slice(&f32::NAN.to_le_bytes());
    let (_, packed) = process_mask_bytes_f32(4, 1, 16, &src).expect("foreground present");
    assert_eq!(packed.len(), 4, "canonical 8-bit-per-pixel payload");
    assert_eq!(packed[0], 0, "0.0 → 0");
    // 0.5 * 255 = 127.5; `round()` ties-to-even on .5 in Rust uses
    // banker's rounding... actually `f32::round()` is half-away-
    // from-zero: 127.5 → 128.
    assert_eq!(packed[1], 128, "0.5 → 128");
    assert_eq!(packed[2], 255, "1.0 → 255");
    assert_eq!(packed[3], 0, "NaN → 0 (background)");
  }

  /// f32 mask values outside `[0, 1]` (e.g. a glitched Vision frame
  /// with negative or super-saturated mask probabilities) must clamp
  /// to the endpoints in the u8 output rather than wrap or silently
  /// produce garbage. `+Inf` and `-Inf` collapse to `0` (background)
  /// like `NaN`.
  #[test]
  fn f32_mask_quantises_out_of_range_and_infinity() {
    // 4×1 row: [-0.5, 1.5, +Inf, -Inf].
    let mut src = vec![0u8; 4 * 4];
    src[0..4].copy_from_slice(&(-0.5_f32).to_le_bytes());
    src[4..8].copy_from_slice(&1.5_f32.to_le_bytes());
    src[8..12].copy_from_slice(&f32::INFINITY.to_le_bytes());
    src[12..16].copy_from_slice(&f32::NEG_INFINITY.to_le_bytes());
    // Foreground = packed[1] (1.5 clamps to 255). The rest collapse
    // to 0 (background), so the mask is technically a single-pixel
    // foreground at column 1.
    let (_, packed) = process_mask_bytes_f32(4, 1, 16, &src).expect("foreground at col 1");
    assert_eq!(packed[0], 0, "-0.5 clamps to 0");
    assert_eq!(packed[1], 255, "1.5 clamps to 255");
    assert_eq!(packed[2], 0, "+Inf → 0 (background)");
    assert_eq!(packed[3], 0, "-Inf → 0 (background)");
  }

  /// A stride wider than `width * bytes_per_pixel` (the buffer has
  /// per-row padding) must still produce the correct tightly-packed
  /// output.
  #[test]
  fn padded_stride_8bit_mask_packs_correctly() {
    // 3×2 mask, stride = 8 (5 bytes of right-padding per row).
    let mut src = vec![0u8; 16];
    src[0] = 1; // row 0, col 0.
    src[10] = 1; // row 1, col 2 (offset 8 + 2).
    let (bbox, packed) = process_mask_bytes_u8(3, 2, 8, &src).expect("foreground produces Some");
    assert_eq!(packed.len(), 3 * 2);
    assert_eq!(packed, [1, 0, 0, 0, 0, 1]);
    // Foreground spans cols 0..=2 and rows 0..=1 — bbox is the whole mask.
    assert!((bbox.x - 0.0).abs() < 1e-6);
    assert!((bbox.y - 0.0).abs() < 1e-6);
    assert!((bbox.width - 1.0).abs() < 1e-6);
    assert!((bbox.height - 1.0).abs() < 1e-6);
  }

  /// A pose with only one surviving joint cannot derive a non-degenerate
  /// bbox. The helper must report `None` so the pose extractor skips
  /// it instead of emitting a zero-extent box that the domain
  /// validator would reject.
  #[test]
  fn pose_bbox_from_single_joint_yields_none() {
    assert!(pose_bbox_from_joint_bounds(0.5, 0.5, 0.5, 0.5).is_none());
  }

  /// A pose where every joint shares the same x (perfectly vertical
  /// limbs) has zero-width bbox and must be reported as `None`.
  #[test]
  fn pose_bbox_from_vertical_joints_yields_none() {
    assert!(pose_bbox_from_joint_bounds(0.5, 0.1, 0.5, 0.9).is_none());
  }

  /// A pose where every joint shares the same y has zero-height bbox
  /// and must be reported as `None`.
  #[test]
  fn pose_bbox_from_horizontal_joints_yields_none() {
    assert!(pose_bbox_from_joint_bounds(0.1, 0.5, 0.9, 0.5).is_none());
  }

  /// A pose with at least one joint per axis produces a valid bbox.
  #[test]
  fn pose_bbox_from_diagonal_joints_is_valid() {
    let bbox =
      pose_bbox_from_joint_bounds(0.1, 0.2, 0.4, 0.6).expect("non-degenerate joints yield Some");
    assert!((bbox.x - 0.1).abs() < 1e-6);
    assert!((bbox.y - 0.2).abs() < 1e-6);
    assert!((bbox.width - 0.3).abs() < 1e-6);
    assert!((bbox.height - 0.4).abs() < 1e-6);
    mediaschema::domain::aggregates::video::BoundingBox::try_new(
      bbox.x,
      bbox.y,
      bbox.width,
      bbox.height,
    )
    .expect("pose-derived bbox satisfies domain invariants");
  }

  /// Non-finite joint coordinates (NaN/Inf from a glitched Vision
  /// observation) must short-circuit before reaching the
  /// `BoundingBox::new` constructor.
  #[test]
  fn pose_bbox_from_nan_joints_yields_none() {
    assert!(pose_bbox_from_joint_bounds(f32::NAN, 0.5, 0.5, 0.5).is_none());
    assert!(pose_bbox_from_joint_bounds(0.1, 0.1, f32::INFINITY, 0.5).is_none());
  }

  /// A document quad whose corners survive per-coord clamp but
  /// collapse to a degenerate shape (e.g. all four corners on a
  /// vertical line because they all clamped to `x = 0.0`) must be
  /// rejected by the domain validator, which the extractor runs
  /// pre-emission.
  #[test]
  fn document_quad_with_collapsed_corners_is_rejected_by_domain() {
    // All four corners at (0.0, 0.0) — collapsed quad.
    let p = (0.0_f32, 0.0_f32);
    assert!(
      mediaschema::domain::aggregates::video::DocumentSegment::try_new(p, p, p, p, 0.9).is_err()
    );
  }

  /// A bow-tie quad (TL & BR swapped) is self-intersecting; the
  /// domain validator rejects it, so the extractor must skip it.
  #[test]
  fn document_quad_bowtie_is_rejected_by_domain() {
    let tl = (0.1_f32, 0.1_f32);
    let tr = (0.9_f32, 0.1_f32);
    let br = (0.1_f32, 0.9_f32);
    let bl = (0.9_f32, 0.9_f32);
    assert!(
      mediaschema::domain::aggregates::video::DocumentSegment::try_new(tl, tr, br, bl, 0.9)
        .is_err()
    );
  }

  /// A well-formed quad passes the domain validator and produces a
  /// valid wire segment.
  #[test]
  fn document_quad_well_formed_is_accepted_by_domain() {
    let tl = (0.1_f32, 0.1_f32);
    let tr = (0.9_f32, 0.1_f32);
    let br = (0.9_f32, 0.9_f32);
    let bl = (0.1_f32, 0.9_f32);
    mediaschema::domain::aggregates::video::DocumentSegment::try_new(tl, tr, br, bl, 0.9)
      .expect("well-formed unit quad is valid");
  }

  // ──────────────── R6 fixes (codex round 6) ────────────────

  /// `finite_f32` returns `Some(v)` only for finite inputs. NaN and
  /// both infinities collapse to `None`.
  #[test]
  fn finite_f32_rejects_non_finite() {
    assert_eq!(finite_f32(0.0), Some(0.0));
    assert_eq!(finite_f32(-1.5), Some(-1.5));
    assert_eq!(finite_f32(1.0), Some(1.0));
    assert_eq!(finite_f32(f32::NAN), None);
    assert_eq!(finite_f32(f32::INFINITY), None);
    assert_eq!(finite_f32(f32::NEG_INFINITY), None);
  }

  /// `try_alloc_packed_mask` enforces a hard upper bound. A request
  /// above `MAX_MASK_BYTES` returns `None` immediately without
  /// touching the allocator, so a corrupted dimensions value cannot
  /// drive the worker into the allocator's abort path.
  #[test]
  fn try_alloc_packed_mask_rejects_oversize() {
    assert!(try_alloc_packed_mask(MAX_MASK_BYTES).is_some());
    assert!(try_alloc_packed_mask(MAX_MASK_BYTES + 1).is_none());
  }

  /// Within the cap, `try_alloc_packed_mask` returns a zero-init
  /// buffer of the requested length.
  #[test]
  fn try_alloc_packed_mask_zero_inits_at_requested_length() {
    let buf = try_alloc_packed_mask(64).expect("64 byte allocation");
    assert_eq!(buf.len(), 64);
    assert!(buf.iter().all(|&b| b == 0));
  }

  /// `process_mask_bytes_u8` and `process_mask_bytes_f32` propagate
  /// the bounded allocation: feeding dimensions whose product
  /// exceeds the cap returns `None` instead of attempting the alloc.
  /// We pick a dimension product just above `MAX_MASK_BYTES`. The
  /// source slice need not be filled with content past the cap —
  /// the function returns at the allocation step before reading any
  /// pixel.
  #[test]
  fn process_mask_bytes_u8_caps_allocation() {
    // (MAX_MASK_BYTES + 1) bytes of packed output. Choose dims that
    // multiply to that value.
    let width = MAX_MASK_BYTES + 1;
    let height = 1;
    // Empty src is fine — the function returns before reading it.
    assert!(process_mask_bytes_u8(width, height, width, &[]).is_none());
  }

  /// Project a face-bbox-relative landmark point into the image's
  /// normalized Vision coordinates. A landmark at the face's centre
  /// (`0.5, 0.5` face-relative) on a face bbox of
  /// `(origin = (0.2, 0.3), size = (0.4, 0.2))` (Vision lower-left)
  /// projects to `(0.2 + 0.5 * 0.4, 0.3 + 0.5 * 0.2) = (0.4, 0.4)`.
  #[test]
  fn project_landmark_to_image_centres_landmark() {
    let face = CGRect::new(CGPoint::new(0.2, 0.3), CGSize::new(0.4, 0.2));
    let projected = project_landmark_to_image(CGPoint::new(0.5, 0.5), face);
    assert!((projected.x - 0.4).abs() < 1e-9);
    assert!((projected.y - 0.4).abs() < 1e-9);
  }

  /// Projection composes with the schema flip. A landmark at the
  /// face's lower-left corner (`(0, 0)` face-relative) on a non-unit
  /// face bbox lands at the face's lower-left in image-normalized
  /// coords. After the schema-side y-flip, the schema-y equals
  /// `1.0 - (face.origin.y + 0 * face.height)`.
  #[test]
  fn project_landmark_then_schema_flip_matches_face_corner() {
    // Face bbox in Vision lower-left: origin (0.2, 0.3), size 0.4×0.2.
    // Face's lower-left landmark = (0, 0) face-relative.
    let face = CGRect::new(CGPoint::new(0.2, 0.3), CGSize::new(0.4, 0.2));
    let projected = project_landmark_to_image(CGPoint::new(0.0, 0.0), face);
    let (sx, sy) =
      vision_point_to_schema(projected.x, projected.y).expect("projected lower-left is finite");
    assert!((sx - 0.2).abs() < 1e-6, "schema-x: {sx}");
    // Vision lower-left at face y = 0.3 → schema-y = 1.0 - 0.3 = 0.7.
    assert!((sy - 0.7).abs() < 1e-6, "schema-y: {sy}");
  }

  /// A non-finite landmark component drops the offending point at
  /// the schema-flip stage even when the face bbox is well-formed.
  /// `project_landmark_to_image` propagates the non-finite component
  /// (`0.2 + NaN * 0.4 = NaN`) and `vision_point_to_schema` rejects
  /// it.
  #[test]
  fn projected_non_finite_landmark_is_rejected() {
    let face = CGRect::new(CGPoint::new(0.2, 0.3), CGSize::new(0.4, 0.2));
    let projected = project_landmark_to_image(CGPoint::new(f64::NAN, 0.5), face);
    assert!(vision_point_to_schema(projected.x, projected.y).is_none());
  }

  // ──────────────── R7 fixes (codex round 7) ────────────────

  /// `sanitize_capture_quality` distinguishes absent from corrupt:
  /// `None` (Vision did not provide a value) collapses to `Some(0.0)`
  /// — fail-closed against any positive threshold; `Some(non_finite)`
  /// collapses to `None` so the caller drops the detection
  /// unconditionally (any `min_capture_quality = 0.0` configuration
  /// would otherwise admit a non-finite reading as a 0.0-quality
  /// face).
  #[test]
  fn sanitize_capture_quality_absent_maps_to_zero() {
    assert_eq!(sanitize_capture_quality(None), Some(0.0));
  }

  #[test]
  fn sanitize_capture_quality_finite_passes_through() {
    assert_eq!(sanitize_capture_quality(Some(0.75)), Some(0.75));
    assert_eq!(sanitize_capture_quality(Some(0.0)), Some(0.0));
    assert_eq!(sanitize_capture_quality(Some(1.0)), Some(1.0));
  }

  /// THE key regression: a non-finite captureQuality must NOT be
  /// substituted with a real value. The previous R6 code returned
  /// `unwrap_or(0.0)` which passed any `min_capture_quality = 0.0`
  /// configuration and admitted the detection. `sanitize_capture_quality`
  /// returns `None` so the caller's `let Some(_) = ... else { continue }`
  /// drops the detection regardless of the configured threshold.
  #[test]
  fn sanitize_capture_quality_non_finite_returns_none() {
    assert_eq!(sanitize_capture_quality(Some(f32::NAN)), None);
    assert_eq!(sanitize_capture_quality(Some(f32::INFINITY)), None);
    assert_eq!(sanitize_capture_quality(Some(f32::NEG_INFINITY)), None);
  }

  /// A finite body_height pairs with whatever height_estimation enum
  /// Vision reported. The pair is forwarded unchanged.
  #[test]
  fn sanitize_body_height_pair_finite_preserves_estimation() {
    let measured = BODY_POSE_3D_HEIGHT_ESTIMATION_MEASURED;
    let (h, e) = sanitize_body_height_pair(1.75, measured);
    assert!((h - 1.75).abs() < 1e-6);
    assert_eq!(e, measured);

    let reference = BODY_POSE_3D_HEIGHT_ESTIMATION_REFERENCE;
    let (h, e) = sanitize_body_height_pair(0.42, reference);
    assert!((h - 0.42).abs() < 1e-6);
    assert_eq!(e, reference);
  }

  /// THE key regression: when body_height is non-finite, the
  /// estimation enum MUST be forced to UNKNOWN. Preserving
  /// MEASURED/REFERENCE while substituting 0.0 would tell consumers
  /// there is a known 0-metre subject — a worse semantic than
  /// "unknown estimate".
  #[test]
  fn sanitize_body_height_pair_non_finite_forces_unknown() {
    for raw in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
      // Even with a Measured input the result must be UNKNOWN.
      let (h, e) = sanitize_body_height_pair(raw, BODY_POSE_3D_HEIGHT_ESTIMATION_MEASURED);
      assert_eq!(h, 0.0, "non-finite must collapse to 0.0 (raw = {raw:?})");
      assert_eq!(
        e, BODY_POSE_3D_HEIGHT_ESTIMATION_UNKNOWN,
        "non-finite must force UNKNOWN (raw = {raw:?})",
      );
      // Same for Reference.
      let (h, e) = sanitize_body_height_pair(raw, BODY_POSE_3D_HEIGHT_ESTIMATION_REFERENCE);
      assert_eq!(h, 0.0);
      assert_eq!(e, BODY_POSE_3D_HEIGHT_ESTIMATION_UNKNOWN);
    }
  }

  /// `validate_mask_dims_for_slice` rejects an output-payload that
  /// would exceed `MAX_MASK_BYTES`, even when the source slice length
  /// is small. This guards the bounded allocator from being asked
  /// for an impossible amount.
  #[test]
  fn validate_mask_dims_rejects_oversize_output() {
    assert!(validate_mask_dims_for_slice(MAX_MASK_BYTES, 1, 0).is_some());
    assert!(validate_mask_dims_for_slice(MAX_MASK_BYTES + 1, 1, 0).is_none());
  }

  /// `validate_mask_dims_for_slice` rejects a source-slice length
  /// over `isize::MAX`. This is the `from_raw_parts` contract; a
  /// corrupted `CVPixelBuffer` reporting a huge `bytes_per_row *
  /// height` must be dropped before the unsafe slice construction.
  #[test]
  fn validate_mask_dims_rejects_isize_overflow_source() {
    assert!(validate_mask_dims_for_slice(1, 1, isize::MAX as usize).is_some());
    assert!(validate_mask_dims_for_slice(1, 1, (isize::MAX as usize).wrapping_add(1)).is_none());
  }

  /// `width * height` overflow returns `None` (the `checked_mul`
  /// inside).
  #[test]
  fn validate_mask_dims_rejects_dim_overflow() {
    assert!(validate_mask_dims_for_slice(usize::MAX, 2, 0).is_none());
  }
}
