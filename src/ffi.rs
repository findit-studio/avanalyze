//! The Vision FFI boundary every entry point shares.
//!
//! Each entry point in this crate owns its own `VNRequest` objects —
//! that is the whole point of the split. What they cannot own
//! separately is the boundary itself: Apple's lower-left coordinates,
//! its unbounded FFI-reported arrays, and the Objective-C exceptions
//! that must never reach a Rust unwind. Those live here, once.
//!
//! Nothing in this module is public API.

use std::{
  borrow::Cow,
  ffi::{c_char, c_void},
  panic::AssertUnwindSafe,
};

use objc2::{Message, exception::catch as catch_objc_exception, rc::Retained};
use objc2_core_foundation::{CFData, CFIndex, CFMutableData, CFRetained, CGPoint, CGRect};
use objc2_core_graphics::{
  CGBitmapInfo, CGColorRenderingIntent, CGColorSpace, CGDataProvider, CGImage, CGImageAlphaInfo,
  CGImageByteOrderInfo,
};
use objc2_foundation::{NSArray, NSData, NSDictionary};
use objc2_vision::{VNRequest, VNSequenceRequestHandler};
use smol_str::{SmolStr, ToSmolStr};

use crate::{
  AnalyzeError, AnalyzeErrorKind, BoundingBox, PixelFormat, PixelPlane,
  plane::MAX_DECODED_IMAGE_BYTES,
};

// ----- resource ceilings shared by more than one entry point ----------------

/// Upper bound on the number of detection results from a single
/// Vision request before we refuse to pre-allocate OR iterate the
/// FFI-reported `results` array. Apple's per-frame extractor
/// outputs cap out in the low hundreds at most (text recognition,
/// face capture, etc.); 4096 is a generous defence-in-depth
/// ceiling against a corrupted / adversarial `NSArray` length that
/// would otherwise drive either the initial `Vec::with_capacity`
/// or the in-loop `Vec::push` reallocation into the allocator's
/// abort path. Every `for x in results.iter()` is bounded with
/// `.iter().take(cap)` — `cap` being this ceiling, unless the call
/// site already holds a tighter bound of its own — so the emitted
/// count cannot exceed the cap, independently of whatever
/// configured `max_results` / `max_segments` / … the call site
/// uses inside the loop.
pub(crate) const MAX_VISION_RESULTS_PER_FRAME: usize = 4096;

/// Upper bound on the number of joints / recognised points read out
/// of a single pose observation's joint dictionary — the
/// `max_entries` every pose call site hands
/// [`read_pose_joints`]. Apple's body-pose / hand-pose /
/// animal-pose joint counts are fixed by the SDK (~17 body, ~21
/// hand, ~25 animal); 256 leaves headroom against future API
/// expansion while still capping both the allocation and the
/// enumeration work a corrupted / adversarial Vision dictionary can
/// demand. Over the cap the pose is DROPPED, never truncated.
pub(crate) const MAX_POSE_JOINTS: usize = 256;

/// Upper bound on the input image byte length accepted by any entry
/// point. Pre-validates the payload BEFORE Foundation copies it into
/// an `NSData`, so an oversized or hostile input cannot double the
/// worker's peak memory and drive the allocator into the abort path.
/// 64 MiB covers an extremely generous keyframe (Apple's typical
/// keyframe encoded JPEG is well under 1 MiB); inputs above that
/// surface as a structured [`AnalyzeError`] instead of an alloc-side
/// crash.
pub(crate) const MAX_INPUT_IMAGE_BYTES: usize = 64 * 1024 * 1024;

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
pub(crate) const MAX_FFI_STRING_BYTES: usize = 4096;

// ----- the cumulative per-call pose budget ----------------------------------

/// Upper bound on the joints ONE pose call may EMIT, summed across
/// every observation in the frame.
///
/// Apple's pose models report roughly 17–25 joints per subject, so 4096
/// emitted joints is about two hundred subjects at twenty joints each —
/// orders past any real frame. It is also what caps the joint NAMES a
/// call can retain: 4096 joints at [`MAX_FFI_STRING_BYTES`] each is
/// 16 MiB even before [`MAX_POSE_JOINT_NAME_BYTES_PER_CALL`], which
/// tightens that to 1 MiB.
pub(crate) const MAX_POSE_JOINTS_PER_CALL: usize = 4096;

/// Upper bound on the joints ONE pose call may WALK, charged one unit
/// per joint entry as that entry is walked — before the entry's keyed
/// lookup, and before any filtering — so a corrupted observation set
/// cannot drive unbounded work on the rejection path.
///
/// Twice [`MAX_POSE_JOINTS_PER_CALL`]: a frame whose joints are walked
/// and then filtered out — by the name, coordinate, confidence or
/// vocabulary gates, none of which spends the emission budget — still
/// has room to reach the emission ceiling before the attempt ceiling
/// trips.
pub(crate) const MAX_POSE_JOINT_ATTEMPTS_PER_CALL: usize = 8192;

/// Upper bound on the joint-NAME bytes one pose call may retain.
///
/// [`MAX_POSE_JOINTS_PER_CALL`] alone still leaves 16 MiB of names
/// reachable (4096 joints × [`MAX_FFI_STRING_BYTES`] each). A real
/// frame's entire joint roster is a few hundred bytes of short Apple
/// identifiers, so 1 MiB keeps four orders of magnitude of headroom
/// while closing the gap between "bounded" and "bounded at a size worth
/// allocating".
pub(crate) const MAX_POSE_JOINT_NAME_BYTES_PER_CALL: usize = 1 << 20;

/// The cumulative joint budget ONE pose call spends, across every
/// observation in the frame.
///
/// # Why the per-observation caps are not enough
///
/// They multiply. [`MAX_VISION_RESULTS_PER_FRAME`] observations each
/// offering [`MAX_POSE_JOINTS`] joints is 1,048,576 joints for a single
/// internally consistent — every individual cap respected — adversarial
/// result, and at [`MAX_FFI_STRING_BYTES`] per joint name that is
/// gigabytes of names alone, allocated infallibly, so the failure mode
/// is the allocator's abort path rather than a dropped pose. A
/// per-observation cap bounds each FACTOR; this budget is what bounds
/// the PRODUCT.
///
/// # How it is spent
///
/// Three counters, charged in two places: the attempt counter one unit
/// per joint entry, inside the dictionary walk and before that entry's
/// keyed lookup ([`charge_joint_visit`](Self::charge_joint_visit)); the
/// joint and name-byte counters once the pose is built and about to be
/// emitted ([`admit_pose`](Self::admit_pose)), so they measure work
/// actually retained.
///
/// # Attempt accounting precedes every rejection branch
///
/// The attempt charge used to be one bulk `charge_attempts(pairs.len())`
/// AFTER the joint dictionary had been read — which meant it never ran
/// on the read's own rejection paths. A dictionary reporting 256 entries
/// but enumerating 255 was allocated for, enumerated, and keyed-looked-up
/// entry by entry, and only then refused; the call site dropped the pose
/// and moved to the next observation with the ceiling untouched. Across
/// [`MAX_VISION_RESULTS_PER_FRAME`] observations that is a million-odd
/// entry walks against an 8192-attempt budget that never moved.
///
/// A reader that walks 256 entries and then reports failure has SPENT
/// that work whether or not a pose came out of it, so charging only on
/// success bounds the productive path and not the walk. The charge
/// therefore sits inside the walk, fused with the ceiling test, one unit
/// per entry visited — the same shape as
/// [`MaskBudget::charge_walk_step`](crate::person_mask::MaskBudget::charge_walk_step)
/// and
/// [`charge_landmark_region_visit`](crate::face_landmarks::charge_landmark_region_visit).
/// No ceiling VALUE changed with the move: for conforming Vision output
/// a pose's walk costs exactly the joints it enumerates, as it always
/// did, and the emitted poses and every `try_new` call are identical.
///
/// # A refusal charges nothing, and means STOP
///
/// Neither method ever partially charges. On a refusal the budget is
/// left exactly as it was, so a pose that does not fit cannot silently
/// consume the remainder on its way out — and a refusal means STOP, not
/// skip: the extractors `break`, because a pose is emitted whole or not
/// at all and skipping ahead to a smaller one would make the output
/// depend on the order Vision happened to report its observations in.
/// This is the landmark budget's "refuse the whole contour" rule
/// ([`region_fits_budget`](crate::face_landmarks::region_fits_budget)),
/// applied to poses.
#[derive(Debug)]
pub(crate) struct PoseBudget {
  joints_remaining: usize,
  attempts_remaining: usize,
  name_bytes_remaining: usize,
}

impl PoseBudget {
  /// A full budget, for one pose call.
  pub(crate) const fn new() -> Self {
    Self {
      joints_remaining: MAX_POSE_JOINTS_PER_CALL,
      attempts_remaining: MAX_POSE_JOINT_ATTEMPTS_PER_CALL,
      name_bytes_remaining: MAX_POSE_JOINT_NAME_BYTES_PER_CALL,
    }
  }

  /// Gate the walk of ONE joint entry on the call's attempt ceiling and
  /// charge it, indivisibly.
  ///
  /// `false` means the ceiling is reached and the caller stops walking;
  /// the budget is left untouched, so a refusal charges nothing. This is
  /// the ONLY place the attempt counter is spent, and it is spent before
  /// the entry's keyed lookup, so no rejection branch downstream of it —
  /// the reader's own included — can be reached without having paid.
  ///
  /// The subtraction is checked rather than wrapping: a counter already
  /// at zero reads as exhausted forever, never as a fresh budget.
  #[inline]
  pub(crate) const fn charge_joint_visit(&mut self) -> bool {
    let Some(attempts_remaining) = self.attempts_remaining.checked_sub(1) else {
      return false;
    };
    self.attempts_remaining = attempts_remaining;
    true
  }

  /// Whether a COMPLETED pose of `joints` joints carrying `name_bytes`
  /// of joint names fits what is left, charging both on success.
  ///
  /// `false` means the caller drops this whole pose and stops. Both
  /// subtractions are checked, and both are computed before either is
  /// stored, so a pose that exceeds one ceiling is not charged against
  /// the other — and `usize::MAX` on either argument refuses without
  /// wrapping or panicking.
  pub(crate) fn admit_pose(&mut self, joints: usize, name_bytes: usize) -> bool {
    let Some(joints_remaining) = self.joints_remaining.checked_sub(joints) else {
      return false;
    };
    let Some(name_bytes_remaining) = self.name_bytes_remaining.checked_sub(name_bytes) else {
      return false;
    };
    self.joints_remaining = joints_remaining;
    self.name_bytes_remaining = name_bytes_remaining;
    true
  }
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
#[inline]
pub(crate) fn clamp01(value: f32) -> f32 {
  debug_assert!(
    value.is_finite(),
    "clamp01 expects finite input; got {value}"
  );
  value.clamp(0.0, 1.0)
}

/// Convert a Vision-framework normalized bounding box (lower-left
/// origin, y grows up) into a consumer's [`BoundingBox`] (top-left
/// origin, y grows down), intersected with the unit square.
///
/// [`BoundingBox`] documents floats in `[0.0, 1.0]` with a top-left
/// origin, while `VNObservation::boundingBox` is documented as a
/// normalized rect in image coordinates where `(0,0)` is the lower-left
/// corner, so the top edge in the contract's space is
/// `1.0 - (origin.y + size.height)`.
///
/// Vision is empirically loose about staying inside the unit square:
/// partially off-screen detections produce `origin.x < 0`,
/// `origin.x + width > 1`, and the like, which a validating
/// [`BoundingBox::try_new`] would reject. The four EDGES are clamped
/// rather than the origin and the extents, so the result can never have
/// `x + width > 1`.
///
/// Three refusals, all of them `None`:
///
/// - **a non-finite component.** Any `NaN` / `±Inf` in the raw
///   rectangle means the box is geometrically meaningless. Dropping it
///   here is what stops [`clamp01`] (which used to collapse non-finite
///   to `0.0`) from fabricating an edge-aligned rectangle that
///   downstream validation would accept.
/// - **a degenerate clamped rectangle** — zero-width or zero-height,
///   i.e. one that intersects the frame in nothing but an edge. A box
///   with no raw area lands here too: `clamp01` is monotone
///   non-decreasing, so a non-positive raw extent forces
///   `clamp01(x + width) <= clamp01(x)` and the clamped extent is `0.0`.
/// - **a vocabulary that declines the box on its own terms.** The clamp
///   above already guarantees the invariants a validating vocabulary
///   checks (finite, `[0, 1]`, positive extent, `left + width <= 1.0`),
///   so a refusal here is a domain rule of the consumer's own, not a
///   regression in the upstream guards. Either way the detection is
///   dropped at the engine layer instead of poisoning downstream
///   storage.
///
/// `standardize()` is assumed to have already been called on `rect`;
/// the input `size` is non-negative.
pub(crate) fn vision_rect_to_bbox<B: BoundingBox>(rect: CGRect) -> Option<B> {
  // Vision lower-left → schema top-left: the top edge in schema space
  // is `1.0 - (origin.y + size.height)`.
  let raw_x = rect.origin.x as f32;
  let raw_y = (1.0 - (rect.origin.y + rect.size.height)) as f32;
  let raw_width = rect.size.width as f32;
  let raw_height = rect.size.height as f32;

  if !(raw_x.is_finite() && raw_y.is_finite() && raw_width.is_finite() && raw_height.is_finite()) {
    return None;
  }

  let left = clamp01(raw_x);
  let top = clamp01(raw_y);
  let right = clamp01(raw_x + raw_width);
  let bottom = clamp01(raw_y + raw_height);
  let width = (right - left).max(0.0);
  let height = (bottom - top).max(0.0);
  if width <= 0.0 || height <= 0.0 {
    return None;
  }
  B::try_new(left, top, width, height).ok()
}

/// Flip a Vision normalized point's y axis to match the contract's
/// top-left origin and clamp both components into `[0.0, 1.0]`.
/// Bounding boxes, 2-D pose joints, face-landmark points, and
/// document-segment corners all share the top-left convention. 3-D
/// joints are model-space metres and are NOT flipped or clamped.
///
/// Returns `None` when either input coordinate is non-finite. A `NaN`
/// or `±Inf` from a glitched Vision observation is geometrically
/// meaningless and previously sanitised to `0.0` via `clamp01`, which
/// fabricated edge-aligned coordinates indistinguishable from real
/// detections. The caller decides whether a single bad point drops
/// the entire detection (e.g. a document quad without all four
/// corners) or just the offending point (e.g. one bad joint among
/// many).
#[inline]
pub(crate) fn vision_point_to_normalized(x: f64, y: f64) -> Option<(f32, f32)> {
  let x32 = x as f32;
  let flipped_y = (1.0 - y) as f32;
  if !x32.is_finite() || !flipped_y.is_finite() {
    return None;
  }
  Some((clamp01(x32), clamp01(flipped_y)))
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
#[inline]
pub(crate) fn project_landmark_to_image(point: CGPoint, face_bbox_vision: CGRect) -> CGPoint {
  CGPoint {
    x: face_bbox_vision.origin.x + point.x * face_bbox_vision.size.width,
    y: face_bbox_vision.origin.y + point.y * face_bbox_vision.size.height,
  }
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
#[inline]
pub(crate) fn finite_f32(v: f32) -> Option<f32> {
  if v.is_finite() { Some(v) } else { None }
}

/// Validate a raw Vision `confidence` value against the configured
/// per-request minimum and the wire/domain `Confidence` invariant
/// (finite, in `[0.0, 1.0]`). Returns `None` if the value is
/// non-finite, outside `[0, 1]`, or below `min` — the caller drops
/// the detection in that case. A simple `value < min` threshold
/// previously let `NaN` through (since every NaN comparison is
/// false) and accepted `>1.0` values, both of which a validating
/// vocabulary rejects.
#[inline]
pub(crate) fn sanitize_confidence(value: f32, min: f32) -> Option<f32> {
  if value.is_finite() && (0.0..=1.0).contains(&value) && value >= min {
    Some(value)
  } else {
    None
  }
}

/// Derive an axis-aligned bounding box from the min/max of a pose's
/// surviving joint coordinates. Returns `None` when the extent in
/// either axis is zero — a single joint, or joints that are perfectly
/// colinear horizontally/vertically, would otherwise produce a wire
/// box that a validating [`BoundingBox::try_new`] rejects.
/// Callers should skip the pose detection on `None`; the joints alone
/// do not carry enough geometry to construct a valid box.
pub(crate) fn pose_bbox_from_joint_bounds<B: BoundingBox>(
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

// ----- bounded FFI reads ----------------------------------------------------

/// What one pose observation's joint dictionary yielded.
///
/// Three outcomes, not two, because the two refusals mean opposite
/// things to the caller: a dictionary that lied about itself is ONE bad
/// observation, and the extractor skips it and reads the next; a call
/// that has spent its walk budget is done, and the extractor stops.
/// Collapsing them into a single `None` is what let a frame of malformed
/// dictionaries be walked one after another with nothing to stop it.
pub(crate) enum PoseJoints<K: Message, V: Message> {
  /// Every entry walked was paid for.
  Read(Vec<(Retained<K>, Retained<V>)>),
  /// The dictionary did not describe itself honestly. Everything walked
  /// before that was discovered has been paid for; the caller drops this
  /// pose and continues.
  Malformed,
  /// The call's attempt ceiling was reached mid-walk. The caller stops
  /// extracting entirely.
  Exhausted,
}

/// Read a Vision-provided dictionary into owned pairs, bounded
/// independently of everything the dictionary reports about itself.
///
/// This is what the four pose joint-dictionary sites use instead of
/// `NSDictionary::to_vecs()`. `to_vecs()` presents a safe surface over
/// an UNSAFE bulk copy: it reads the dictionary's `count`, allocates
/// two `Vec`s of exactly that capacity, invokes the **deprecated,
/// unbounded** `getObjects:andKeys:` — Apple's own header says it "is
/// unsafe because it could potentially cause buffer overruns. You
/// should use -getObjects:andKeys:count:" — and then `set_len`s both
/// vectors to the reported count. A dictionary whose bulk copy
/// disagrees with its `count` therefore writes past the allocations
/// (count too low) or hands out uninitialised pointers (count too
/// high). Either is heap corruption / UB, and it happens BEFORE any
/// Rust-side check can reject the dictionary — a length guard in front
/// of the call cannot help, because the guard reads the very number
/// that is not to be believed.
///
/// So the count is not trusted here at all. `count` and fast
/// enumeration are two independent answers from the same object, and
/// nothing obliges them to agree; this reader asks for both and
/// refuses the read unless they do. The enumeration is `.take()`n at
/// one past the cap so the WORK is bounded even for a dictionary that
/// enumerates forever, and the allocation is bounded because
/// `reported <= max_entries` is checked before `with_capacity`.
///
/// Pairing is by keyed lookup rather than by index. `to_vecs()`
/// returns two parallel vectors that call sites `zip`, which assumes
/// `keys[i]` belongs with `values[i]`; [`NSDictionary::objectForKey`]
/// pairs each joint name with the value actually stored under it.
///
/// Every entry the enumeration hands out is charged to `budget` as it is
/// walked — see [`collect_dictionary_pairs`], which this forwards to
/// with the count the dictionary reports about itself.
///
/// [`PoseJoints::Malformed`] means "drop the whole pose";
/// [`PoseJoints::Exhausted`] means "stop reading this frame".
pub(crate) fn read_pose_joints<K: Message, V: Message>(
  dict: &NSDictionary<K, V>,
  max_entries: usize,
  budget: &mut PoseBudget,
) -> PoseJoints<K, V> {
  collect_dictionary_pairs(dict, dict.len(), max_entries, budget)
}

/// The decision behind [`read_pose_joints`], with the dictionary's
/// self-reported count as an explicit argument so it can be lied to
/// under test — which is the whole point, because that count is exactly
/// the value this function refuses to trust. Under a real Vision
/// dictionary the only caller passes `dict.len()`.
///
/// Four refusals, all of them [`PoseJoints::Malformed`], and all of them
/// meaning **drop the pose** rather than truncate it:
///
/// - **over the cap** — `reported > max_entries`. The pre-existing
///   size guard, unchanged in meaning. Nothing is walked, so nothing is
///   charged.
/// - **the enumeration yields MORE than the count claimed.** Checked
///   before each push, so `pairs` never grows past the capacity
///   `reported` bought. This is the half that `to_vecs()` would have
///   turned into a write past the allocation.
/// - **a key with no value.** A sound `NSDictionary` cannot hold a nil
///   value, so this is a malformed dictionary and the whole read is
///   refused.
/// - **the enumeration yields FEWER than the count claimed.** This is
///   the half that `to_vecs()` would have turned into `set_len` over
///   uninitialised pointers.
///
/// # The walk pays as it goes
///
/// Every one of those refusals is DOWNSTREAM of entries already walked,
/// and walking an entry is real work: an enumeration step, a retain, and
/// — but for the two checks above it — a keyed lookup across the FFI
/// boundary. So the walk is admitted per entry rather than the read
/// being charged per success: `budget.charge_joint_visit()` is the first
/// act of each iteration, before the over-enumeration check and before
/// `objectForKey`, and a refusal there ends the read as
/// [`PoseJoints::Exhausted`] — the extractor's signal to stop the frame,
/// distinct from the malformed-dictionary signal to skip one pose.
///
/// A malformed dictionary therefore pays for exactly the entries it
/// walked before its lie was discovered, and a conforming one pays for
/// exactly the entries it enumerates — the same total the removed bulk
/// `charge_attempts(pairs.len())` charged, so no ceiling value moved and
/// no conforming frame's output changed.
pub(crate) fn collect_dictionary_pairs<K: Message, V: Message>(
  dict: &NSDictionary<K, V>,
  reported: usize,
  max_entries: usize,
  budget: &mut PoseBudget,
) -> PoseJoints<K, V> {
  if reported > max_entries {
    return PoseJoints::Malformed;
  }
  // Safe now: `reported <= max_entries`, so a corrupted / adversarial
  // count cannot drive the infallible allocation into the abort path.
  let mut pairs = Vec::with_capacity(reported);
  // One past the cap: enough to observe an over-enumeration, never
  // enough to let one run away.
  for key in dict.keys().take(max_entries + 1) {
    // The ceiling test AND the charge for this entry, as one step,
    // ahead of both rejection branches below and ahead of the keyed
    // lookup between them. An entry the enumeration handed out has been
    // walked whatever the reader decides about it next.
    if !budget.charge_joint_visit() {
      return PoseJoints::Exhausted;
    }
    if pairs.len() == reported {
      return PoseJoints::Malformed;
    }
    let Some(value) = dict.objectForKey(&key) else {
      return PoseJoints::Malformed;
    };
    pairs.push((key, value));
  }
  if pairs.len() != reported {
    return PoseJoints::Malformed;
  }
  PoseJoints::Read(pairs)
}

/// Convert an FFI-sourced `NSString` to a Rust `SmolStr` after
/// verifying its UTF-8 byte length is within
/// [`MAX_FFI_STRING_BYTES`]. Returns `None` if the `NSString`'s
/// reported byte length exceeds the bound; callers drop the
/// offending field (text candidate / barcode payload /
/// classification label / joint name) rather than driving the
/// allocator into the abort path. The length query is FFI but
/// allocation-free.
pub(crate) fn ffi_nsstring_to_smolstr(ns_str: &objc2_foundation::NSString) -> Option<SmolStr> {
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

/// Compute the effective per-extractor cap as
/// `min(user_configured_max, MAX_VISION_RESULTS_PER_FRAME)`. Use
/// this for ALL of: `Vec::with_capacity(cap)`, `.iter().take(cap)`,
/// and the in-loop `if emitted.len() >= cap { break; }` guard.
/// Composing the three around the SAME `cap` value bounds both
/// capacity and emission to the hard ceiling, regardless of what
/// the caller configured.
#[inline]
pub(crate) fn effective_results_cap(user_max: usize) -> usize {
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
#[allow(dead_code)]
#[inline]
pub(crate) fn validate_raw_slice_bytes(byte_len: usize, max_bytes: usize) -> Option<()> {
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
#[inline]
pub(crate) fn validate_raw_slice_elems<T>(elem_count: usize, max_elems: usize) -> Option<()> {
  if elem_count > max_elems {
    return None;
  }
  let byte_len = elem_count.checked_mul(core::mem::size_of::<T>())?;
  if byte_len > isize::MAX as usize {
    return None;
  }
  Some(())
}

// ----- the native barrier ---------------------------------------------------

/// The bytes reserved for a caught exception's message.
///
/// Apple's raises carry a model path and an error code — the ANE
/// refusal is a URL plus a sentence — and 512 bytes holds those whole
/// while keeping the buffer on the stack of a function that runs on
/// every guarded call. Anything longer is truncated by the trampoline,
/// which NUL-terminates regardless.
const GUARD_MESSAGE_CAPACITY: usize = 512;

/// What [`avanalyze_0_6_guard`] reports about the call it ran.
///
/// The numbers are the trampoline's, spelled out in
/// `src/objc_cxx_barrier.mm`; this is where they are given meaning. A
/// status the trampoline never returns has none, and
/// [`guard_native`] refuses rather than guesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GuardStatus {
  /// The guarded call returned normally.
  Completed,
  /// An Objective-C exception — `NSException` or any other `@throw`n
  /// object — was caught.
  ObjCException,
  /// A `std::exception` was caught. This is the one Apple's C++ layers
  /// (Espresso, ANECF) throw natively, before their Objective-C
  /// wrappers turn it into an `NSException`.
  CxxException,
}

impl GuardStatus {
  /// Reads a trampoline status, or `None` for a number it never
  /// returns.
  const fn from_raw(status: i32) -> Option<Self> {
    match status {
      0 => Some(Self::Completed),
      1 => Some(Self::ObjCException),
      2 => Some(Self::CxxException),
      _ => None,
    }
  }

  /// How the caught exception is named in the error message.
  const fn describe(self) -> &'static str {
    match self {
      Self::Completed => "completed",
      Self::ObjCException => "an Objective-C exception",
      Self::CxxException => "a C++ exception",
    }
  }
}

/// Run `f` under a barrier NO Apple framework exception can cross, and
/// report a caught one as an [`AnalyzeErrorKind::Environment`] refusal.
///
/// This is the crate's floor. Every other guard is built on it, and
/// every path that can enter CoreML — the nine entry-point
/// constructors, the image preparation, the perform, each extraction —
/// runs inside one.
///
/// # Why a C++ frame and not [`objc2::exception::catch`]
///
/// Because objc2's barrier is `@catch (id)`: it matches Objective-C
/// objects and lets everything else keep unwinding, which is correct
/// for what it promises and is not a barrier against the layer beneath.
/// Vision sits on CoreML, CoreML on Espresso and ANECF, and those are
/// C++ libraries. The `NSException` a denied Neural Engine produces is
/// one their Objective-C wrapper made; nothing in the API contract says
/// the wrapper is total, and what escapes it reaches Rust as a foreign
/// exception that aborts the process at the first
/// [`catch_unwind`](std::panic::catch_unwind) it meets — `fatal runtime
/// error: Rust cannot catch foreign exceptions`.
///
/// `src/objc_cxx_barrier.mm` names both worlds in one `try`, which only
/// an Objective-C++ translation unit can do, and returns a status
/// instead of unwinding.
///
/// # Nothing catches between the raise and the trampoline
///
/// The callback below is `extern "C-unwind"` and installs no
/// `catch_unwind`, and that is the load-bearing part of this design
/// rather than an omission. A Rust `catch_unwind` anywhere between the
/// raise and the C++ `try` would claim the exception first — its
/// landing pad matches every exception class — and abort on the
/// mismatch. The containment has to sit in the frame that can name what
/// it caught.
///
/// # A Rust panic is not this barrier's to catch
///
/// The trampoline's clauses are all NAMED C++ types and there is no
/// `catch (...)`, so a panic raised inside `f` passes THROUGH the C++
/// frame and is caught by the Rust runtime that raised it — the only
/// arrangement the Rust Reference sanctions, since it gives no
/// guarantees for a foreign runtime that disposes of or rethrows a Rust
/// panic payload. Concretely that is what lets
/// [`BodyPoser::extract_3d`](crate::BodyPoser) keep the `catch_unwind`
/// it wraps this barrier in for objc2's debug-build encoding checks,
/// and what keeps a CONSUMER's panic — their vocabulary constructors
/// run inside the extraction guards — a panic rather than a dead
/// process.
///
/// The price is one named residual: a C++ throw of a type unrelated to
/// `std::exception` is not caught either, and behaves exactly as it did
/// before this barrier existed. `src/objc_cxx_barrier.mm` argues that
/// trade, and `src/tests/native_barrier.rs` pins it.
///
/// # It needs `panic = "unwind"`
///
/// The raise crosses one Rust frame — the `run` callback below — on its
/// way to the C++
/// `try`, because the code that calls Vision is Rust. Under
/// `panic = "abort"` rustc puts an abort-on-unwind shim on every
/// `extern "C-unwind"` boundary, so the exception dies in that frame
/// and this function never sees it.
///
/// That is not a constraint this barrier introduced.
/// [`objc2::exception::catch`], which this crate has always used and
/// which is built on the identical geometry — a Rust `extern "C-unwind"`
/// callback inside an Objective-C `@try` — documents the same
/// limitation, and the `catch_unwind` in
/// [`BodyPoser::extract_3d`](crate::BodyPoser) is equally inert there.
/// A `panic = "abort"` consumer therefore gets exactly the behaviour
/// they had before this barrier existed.
///
/// # Safety of the closure
///
/// `f` may capture `&mut` state and non-`UnwindSafe` handles; no
/// unwind-safety bound is asked for, because nothing here observes the
/// closure's captures after a caught exception. On a catch the
/// closure's frame has already been unwound, its borrows dropped, and
/// this function returns an error built from the message alone.
pub(crate) fn guard_native<R, F: FnOnce() -> R>(
  site: &'static str,
  f: F,
) -> Result<R, AnalyzeError> {
  /// The closure and its result, reached through the `void *` the
  /// trampoline hands back to [`run`].
  struct Call<F, R> {
    body: Option<F>,
    out: Option<R>,
  }

  /// The callback the trampoline calls, once.
  ///
  /// `C-unwind`, because an Apple raise has to unwind THROUGH this
  /// frame to reach the `try` outside it; a `C` declaration would mark
  /// it `nounwind` and turn that into an abort.
  unsafe extern "C-unwind" fn run<F: FnOnce() -> R, R>(context: *mut c_void) {
    // SAFETY: the only caller is the trampoline below, which is handed
    // `context` as a pointer to the `Call` on this function's caller's
    // stack. That `Call` outlives the trampoline call, and nothing else
    // borrows it while the call is in flight, so the exclusive
    // reference is sound and unaliased.
    let call = unsafe { &mut *context.cast::<Call<F, R>>() };
    if let Some(body) = call.body.take() {
      call.out = Some(body());
    }
  }

  let mut call = Call {
    body: Some(f),
    out: None,
  };
  let mut message = [0u8; GUARD_MESSAGE_CAPACITY];
  // SAFETY: `run::<F, R>` matches the trampoline's `void (*)(void *)`
  // and is handed the matching context; the buffer is writable for the
  // length passed, which is its own. The call may return through a
  // rethrown Rust panic, which is why the declaration is `C-unwind`.
  let status = unsafe {
    avanalyze_0_6_guard(
      run::<F, R>,
      core::ptr::from_mut(&mut call).cast(),
      message.as_mut_ptr().cast(),
      message.len(),
    )
  };

  match GuardStatus::from_raw(status) {
    Some(GuardStatus::Completed) => call.out.ok_or_else(|| {
      // Unreachable while the trampoline honours its contract: a
      // completed status means the callback ran, and the callback's
      // only exit sets `out`. Refused rather than unwrapped, because
      // this crate's answer to an impossible native state is an error,
      // not a panic.
      AnalyzeError::new(
        AnalyzeErrorKind::Environment,
        format!("{site}: the native barrier reported success without running the guarded call"),
      )
    }),
    Some(caught) => {
      let detail = guard_message(&message);
      #[cfg(feature = "tracing")]
      tracing::warn!(
        site,
        status = caught.describe(),
        detail = %detail,
        "Apple's native stack raised across the FFI boundary; the call is refused",
      );
      Err(AnalyzeError::new(
        AnalyzeErrorKind::Environment,
        format!("{site}: {} — {detail}", caught.describe()),
      ))
    }
    None => Err(AnalyzeError::new(
      AnalyzeErrorKind::Environment,
      format!("{site}: the native barrier returned an unknown status {status}"),
    )),
  }
}

/// Read the trampoline's message buffer back as text.
///
/// The buffer is NUL-terminated by the trampoline and truncated to fit,
/// which can split a UTF-8 sequence — so the conversion is lossy rather
/// than fallible: a caught exception must always produce a message, and
/// a replacement character in a diagnostic is better than no diagnostic.
/// A buffer with no NUL at all is a broken contract, and reads as empty.
fn guard_message(buffer: &[u8]) -> Cow<'_, str> {
  let bytes =
    core::ffi::CStr::from_bytes_until_nul(buffer).map_or(&[][..], |message| message.to_bytes());
  String::from_utf8_lossy(bytes)
}

// `C-unwind`, not `C`: the trampoline names only the C++ types it can
// handle, so an exception it does not name — a Rust panic on its way
// out of the guarded closure, above all — keeps unwinding straight
// through it. A `C` declaration would mark the frame `nounwind` and
// turn that crossing into an abort instead of a panic the caller can
// catch.
unsafe extern "C-unwind" {
  /// `src/objc_cxx_barrier.mm` — one `try`, three outcomes, and
  /// everything it cannot name left to keep unwinding.
  ///
  /// The `0_6` is the crate's major.minor, for the reason the simd
  /// shim's declaration below spells out.
  fn avanalyze_0_6_guard(
    body: unsafe extern "C-unwind" fn(*mut c_void),
    context: *mut c_void,
    message: *mut c_char,
    message_capacity: usize,
  ) -> i32;
}

/// Run `f` under the crate's exception barriers, returning `fallback`
/// (the empty/degraded result for that detector) if Apple's Vision
/// framework raises across the FFI boundary.
///
/// Two barriers, and each catches what the other does not.
/// [`objc2::exception::catch`] is the inner one: it converts an
/// unwinding Objective-C object into a `Result`, and a caught
/// exception's `name`/`reason` is logged via the exception's (safe)
/// `Display` impl. [`guard_native`] is the outer one, and it is what
/// stands between this crate and everything objc2's `@catch (id)`
/// deliberately lets through — the C++ exceptions of the CoreML stack
/// underneath Vision.
///
/// Either way one misbehaving detector degrades to an empty result for
/// that detector instead of taking the whole worker (and pipeline)
/// down. Rust's [`std::panic::catch_unwind`] (used in the 3-D body-pose
/// path for a *Rust*-panic quirk) explicitly **cannot** stand in for
/// either: a foreign exception reaching it aborts the entire process
/// with `fatal runtime error: Rust cannot catch foreign exceptions`.
///
/// `detector` names the Vision request whose perform/result-extraction
/// raised, so the warning pins the culprit.
///
/// The closure is wrapped in [`AssertUnwindSafe`] internally: it
/// captures retained `VNRequest` handles (which are not
/// `RefUnwindSafe`) and, for the mask extractors, `&mut` budget
/// counters. Asserting unwind-safety is sound here — on a caught
/// exception the closure's borrows are dropped and `fallback` is
/// returned, so no half-updated state is ever observed. A partially
/// advanced mask counter only ever over-counts (the conservative
/// direction for a resource cap), never under-counts.
///
/// # A caught perform is reported, not swallowed
///
/// Returning `fallback` makes a caught exception look, to the caller,
/// exactly like a closure that had nothing to report — which is right
/// for an EXTRACTION (nothing was read, so nothing is emitted) and
/// wrong for a PERFORM. A perform's product is not its return value
/// but the state it leaves on the request objects, and those objects
/// outlive the call. So [`perform`] does not degrade to `Ok(())`: it
/// reports [`Performed::Raised`], and its callers must not read any
/// request state after one — see [`Performed`] for what that state
/// would be.
pub(crate) fn guard_vision_ffi<R>(detector: &'static str, fallback: R, f: impl FnOnce() -> R) -> R {
  // The nesting is load-bearing. objc2's barrier is INSIDE, so an
  // Objective-C raise keeps the handling and the logging it always had;
  // the native barrier is OUTSIDE, so a C++ throw — which objc2's
  // `@catch (id)` does not match and does not stop — meets a frame that
  // can name it before it reaches any Rust one.
  match guard_native(detector, || catch_objc_exception(AssertUnwindSafe(f))) {
    Ok(Ok(value)) => value,
    Ok(Err(exception)) => {
      #[cfg(feature = "tracing")]
      match &exception {
        // The `Exception` `Display` impl is safe: it checks
        // `isKindOfClass: NSException` internally and renders the
        // reason only when present, so logging it cannot itself raise.
        Some(exc) => tracing::warn!(
          detector,
          exception = %exc,
          "Apple Vision raised an Objective-C exception; skipping this detector and returning a \
           partial result",
        ),
        None => tracing::warn!(
          detector,
          "Apple Vision raised a nil Objective-C exception; skipping this detector and returning \
           a partial result",
        ),
      }
      #[cfg(not(feature = "tracing"))]
      let _ = (detector, exception);
      fallback
    }
    // `guard_native` has already logged what it caught, and the
    // degradation contract for an extraction is the same whichever
    // world the exception came from: nothing was read, so nothing is
    // emitted.
    Err(_error) => fallback,
  }
}

// `C-unwind`, not `C`: the send inside can raise an Objective-C
// exception, which unwinds out through this boundary on its way to the
// `objc2::exception::catch` the caller sits inside. Declaring it `C`
// would mark it `nounwind`, and an unwind out of a `nounwind` frame is
// undefined behaviour rather than a caught exception. objc2 declares
// `objc_msgSend` and `objc2-exception-helper`'s own entry point the
// same way, for the same reason. The Objective-C side is compiled with
// exception support so its frame is unwind-transparent too — see
// `build.rs`.
unsafe extern "C-unwind" {
  /// `src/objc_simd_shim.m` — one message send, emitted by Clang.
  ///
  /// The `0_6` is the crate's major.minor, not decoration: C has no
  /// namespaces, so an unversioned name would collide with a second
  /// avanalyze in the same graph. `build.rs` scopes the archive and
  /// fails the build if the tag stops matching the package version.
  fn avanalyze_0_6_vn_point3d_position(receiver: *mut objc2::runtime::AnyObject, out: *mut f32);
}

/// Reads `-[VNPoint3D position]` off `receiver`, as 16 floats in
/// Apple's column-major memory order.
///
/// # Why this is not a `msg_send!`
///
/// Because Rust cannot receive the return type, on either of the two
/// levels the failure hides on.
///
/// The **encoding** level is the visible one: `simd_float4` is an
/// `ext_vector_type`, for which Clang deliberately emits no `@encode`
/// character, so `@encode(simd_float4x4)` is the literal `{?=[4]}` — an
/// unnamed struct wrapping an array of four elements whose type is
/// unwritable. objc2's debug-build verification compares that against
/// whatever `Encode` impl a caller supplies, and a mismatch is a Rust
/// panic at the send.
///
/// The **ABI** level is the one that matters, and no `Encode` impl
/// reaches it. On arm64 `simd_float4x4` is a homogeneous vector
/// aggregate of four 128-bit vectors, returned in v0-v3. rustc's
/// `extern "C"` returns a vector aggregate in registers only when the
/// *whole aggregate* is 64 or 128 bits; at 512 it falls back to the x8
/// hidden pointer. Measured, not assumed: for a `#[repr(C)]` struct of
/// four `float32x4_t`, rustc emits `mov x8, sp` before the call and
/// reads the buffer afterwards, while Clang emits `stp q0, q1` /
/// `stp q2, q3` from the returned registers. The callee writes v0-v3;
/// Rust reads a stack buffer nobody wrote.
///
/// That is not a hypothetical: it was live in this crate. Every 3-D
/// joint carried stale stack bytes reinterpreted as metres — values
/// near `1e26`, and *finite*, so every finiteness check passed them.
///
/// There is no Rust type that fixes it. `#[repr(simd)]` is unstable,
/// and would not help: the aggregate-size rule rejects the shape, not
/// the element. A float HFA reaches only s0-s3 or d0-d3 — the low half
/// of each register — which can recover a translation's `x` and `y` and
/// can never recover its `z`.
///
/// # Why the selector is not a parameter
///
/// Because x86_64 returns this matrix by a different convention *and
/// through a different dispatcher*. It is MEMORY class there, which the
/// runtime reaches by `objc_msgSend_stret` — a symbol that does not
/// exist on arm64 at all. Which dispatcher is correct depends on the
/// target and the method's return type together, so the shim types the
/// send rather than choosing one: Clang then emits
/// `objc_msgSend_stret` for x86_64 and `objc_msgSend` for arm64, and
/// this crate holds no architecture table whose wrong cells would be
/// memory corruption.
///
/// Typing the send means fixing the selector, which also deletes the
/// sharpest edge this seam could have had — a caller can no longer name
/// a method whose return type disagrees with the convention being used
/// to read it.
///
/// # Safety
///
/// `receiver` must be a live object responding to `position` with a
/// method whose Objective-C return type is `simd_float4x4`; every
/// `VNPoint3D` and its subclasses do. A receiver that does not respond
/// raises, and callers are inside [`guard_vision_ffi`], which catches
/// it.
pub(crate) unsafe fn vn_point3d_position<T: Message>(receiver: &T) -> [f32; 16] {
  let mut matrix = [0f32; 16];
  // SAFETY: the shim reads `receiver` only by sending it `position`,
  // and writes exactly the 16 floats of a `simd_float4x4` into `out`,
  // which is the length of `matrix`. The receiver is borrowed for the
  // call, so it outlives it; the caller upholds the receiver contract.
  unsafe {
    avanalyze_0_6_vn_point3d_position(
      core::ptr::from_ref(receiver).cast_mut().cast(),
      matrix.as_mut_ptr(),
    );
  }
  matrix
}

// ----- the shared request run -----------------------------------------------

/// Refuse an oversized payload BEFORE Foundation copies it into an
/// `NSData` and doubles peak memory. Surfaces as a structured error so
/// the orchestrator can decide whether to retry, log, or escalate.
#[inline]
fn check_input_len(jpeg: &[u8]) -> Result<(), AnalyzeError> {
  if jpeg.len() > MAX_INPUT_IMAGE_BYTES {
    return Err(AnalyzeError::new(
      AnalyzeErrorKind::RequestFailed,
      "input image exceeds MAX_INPUT_IMAGE_BYTES",
    ));
  }
  Ok(())
}

// ----- the decoded-dimension SOF preflight -----------------------------------
//
// A JPEG stream is SOI (`FF D8`), then a sequence of marker segments, each
// `FF <code>` optionally followed by a big-endian u16 length (counting
// itself) and that many bytes of payload. This walk reads only that
// structure — never the entropy-coded scan data a real decoder would need —
// to reach the first SOF (Start Of Frame) marker and its declared
// dimensions. It is allocation-free (every read comes straight out of
// `jpeg`) and total: every exit is a `Result`, never a panic or an
// out-of-bounds read, no matter how the input is truncated or how its
// length fields lie.
//
// Two primitives do every bounds-checked read in the walk: [`next_byte`]
// (one byte, `pos` advanced by one) and [`bounded_slice`] (a range, its end
// computed with checked addition). Every other helper is built from those
// two, so there is exactly one place that indexes a single byte and one
// place that slices a range.

/// Bytes ImageIO's decode allocates per pixel in the worst case, as a
/// function of the SOF's sample `precision` (bits per component) — its
/// worst-case buffer is 4 channels (RGBA-family) at either 1 or 2 bytes
/// per channel. Channel count is fixed at 4 rather than read from the
/// JPEG's own `Nf` component count: a hostile SOF could under-report `Nf`
/// (claim grayscale) to shrink the declared budget while Vision still
/// allocates the wider buffer, so the bound has to assume the worst case
/// regardless of what the file claims.
///
/// Precision, unlike `Nf`, does change the multiplier: baseline JPEGs
/// declare 8-bit precision and ImageIO decodes them at 4 bytes/pixel
/// (8-bit RGBA), but SOF1 and SOF3 (and the other extended/lossless SOF
/// markers) permit 9-16 bit precision, and ImageIO decodes those at
/// 16-bit/component — 8 bytes/pixel, double the baseline case. Charging
/// only 4 bytes/pixel for a 12- or 16-bit-precision frame would let a
/// nominal at-cap frame's real decode land at roughly twice
/// [`MAX_DECODED_IMAGE_BYTES`], so any precision above 8 bits charges the
/// wider rate.
fn decoded_bytes_per_pixel(precision: u8) -> u64 {
  const CHANNELS: u64 = 4;
  let bytes_per_channel: u64 = if precision > 8 { 2 } else { 1 };
  CHANNELS * bytes_per_channel
}

/// SOF-family marker codes: 0xC0..=0xCF minus the three reserved for other
/// purposes — 0xC4 (DHT, Define Huffman Table), 0xC8 (JPG, reserved and
/// never emitted), 0xCC (DAC, Define Arithmetic Coding). The remaining
/// thirteen (SOF0-3, SOF5-7, SOF9-11, SOF13-15) cover every real frame type
/// — baseline, extended, progressive, lossless, their differential
/// variants, and both entropy codings — and all thirteen share the same
/// header shape: precision, height, width, `Nf`. That shape is all this
/// preflight reads.
fn is_sof_marker(code: u8) -> bool {
  matches!(code, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF)
}

fn no_sof_marker() -> AnalyzeError {
  AnalyzeError::new(
    AnalyzeErrorKind::RequestFailed,
    "input image has no valid JPEG SOF marker",
  )
}

fn malformed_marker_length() -> AnalyzeError {
  AnalyzeError::new(
    AnalyzeErrorKind::RequestFailed,
    "input image has a malformed JPEG marker length",
  )
}

fn deferred_sof_dimension() -> AnalyzeError {
  AnalyzeError::new(
    AnalyzeErrorKind::RequestFailed,
    "input image's SOF marker declares a dimension of zero",
  )
}

fn hierarchical_jpeg_unsupported() -> AnalyzeError {
  AnalyzeError::new(
    AnalyzeErrorKind::RequestFailed,
    "input image uses hierarchical JPEG (DHP), which this preflight does not support",
  )
}

/// Reads the byte at `*pos` and advances it by one. The only place in this
/// module that indexes a single byte — every walk step that is not a
/// checked-range read ([`bounded_slice`]) goes through this, so a
/// truncated buffer always surfaces as [`no_sof_marker`] here rather than a
/// panic.
fn next_byte(jpeg: &[u8], pos: &mut usize) -> Result<u8, AnalyzeError> {
  let byte = *jpeg.get(*pos).ok_or_else(no_sof_marker)?;
  *pos = pos.checked_add(1).ok_or_else(malformed_marker_length)?;
  Ok(byte)
}

/// A bounds-checked subslice: `start` and `len` are combined with checked
/// addition before the range ever reaches indexing, so a forged or
/// oversized length can only ever produce [`malformed_marker_length`],
/// never an out-of-bounds read or an overflow panic.
fn bounded_slice(jpeg: &[u8], start: usize, len: usize) -> Result<&[u8], AnalyzeError> {
  let end = start.checked_add(len).ok_or_else(malformed_marker_length)?;
  jpeg.get(start..end).ok_or_else(malformed_marker_length)
}

fn read_u16_be(jpeg: &[u8], pos: usize) -> Result<u16, AnalyzeError> {
  let bytes = bounded_slice(jpeg, pos, 2)?;
  Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
}

/// Reads one marker's code, absorbing any `0xFF` fill bytes the JPEG spec
/// permits before it (`FF FF FF D8` is SOI preceded by two fill bytes).
/// `*pos` must point at the marker's leading `0xFF`; on return it points
/// just past the code byte.
///
/// Terminates because every branch either returns or calls [`next_byte`],
/// which advances `*pos` by exactly one and errors the moment the buffer is
/// exhausted — so this can read at most `jpeg.len() - *pos` bytes before
/// one of those two things happens.
fn read_marker_code(jpeg: &[u8], pos: &mut usize) -> Result<u8, AnalyzeError> {
  if next_byte(jpeg, pos)? != 0xFF {
    return Err(no_sof_marker());
  }
  loop {
    let byte = next_byte(jpeg, pos)?;
    if byte != 0xFF {
      return Ok(byte);
    }
  }
}

/// Skips a generic length-prefixed marker segment: `pos` points at its
/// 2-byte length field (which counts itself), so a well-formed segment
/// always reports `seg_len >= 2`. Returns the position just past it.
fn skip_segment(jpeg: &[u8], pos: usize) -> Result<usize, AnalyzeError> {
  let seg_len = usize::from(read_u16_be(jpeg, pos)?);
  if seg_len < 2 {
    return Err(malformed_marker_length());
  }
  let next = pos
    .checked_add(seg_len)
    .ok_or_else(malformed_marker_length)?;
  if next > jpeg.len() {
    return Err(malformed_marker_length());
  }
  Ok(next)
}

/// The fields of a SOF payload this preflight needs: enough to compute a
/// worst-case decoded byte count, and nothing else — no component
/// specifiers, no quantization or Huffman tables, no scan/entropy data.
struct SofFrame {
  width: u16,
  height: u16,
  /// Sample precision in bits per component, straight from the SOF
  /// payload. Feeds [`decoded_bytes_per_pixel`]; never validated against
  /// a fixed vocabulary here, because any value this preflight has not
  /// specifically accounted for must be treated as the untrusted worst
  /// case, not silently normalized to a common one.
  precision: u8,
}

/// Reads a SOF payload's fixed-offset fields: `pos` points at its 2-byte
/// length field. Returns as soon as they are read — the component
/// specifiers that follow are validated for length only, never read,
/// because this is a dimension preflight, not a decoder.
///
/// Refuses a `width` or `height` of zero. Width is always fully specified
/// in a JPEG SOF, so zero is simply degenerate. Height is not: JPEG
/// permits a baseline SOF to declare `height = 0` and defer the real
/// value to a DNL marker after the first scan (ITU-T T.81 §B.2.5). This
/// preflight stops at the first SOF and never reads that far — reading
/// forward would mean skipping entropy-coded scan data, which is decoder
/// territory this preflight deliberately stays out of — so a deferred
/// height is a height this preflight cannot establish, and a preflight
/// that cannot establish a bound must refuse rather than default to
/// treating an unknown value as zero.
fn read_sof_dimensions(jpeg: &[u8], pos: usize) -> Result<SofFrame, AnalyzeError> {
  // length(2) + precision(1) + height(2) + width(2) + Nf(1).
  const FIXED_HEADER_LEN: usize = 8;

  let seg_len = usize::from(read_u16_be(jpeg, pos)?);
  if seg_len < FIXED_HEADER_LEN {
    return Err(malformed_marker_length());
  }
  let payload_start = pos.checked_add(2).ok_or_else(malformed_marker_length)?;
  let payload = bounded_slice(jpeg, payload_start, seg_len - 2)?;

  let precision = payload[0];
  let height = u16::from_be_bytes([payload[1], payload[2]]);
  let width = u16::from_be_bytes([payload[3], payload[4]]);

  // Read only to validate the segment's own declared length against its
  // component count (`Nf` is capped at 255 by its one-byte width, so
  // `3 * nf` cannot overflow `usize` on any target this crate builds for).
  let nf = usize::from(payload[5]);
  if seg_len < FIXED_HEADER_LEN + 3 * nf {
    return Err(malformed_marker_length());
  }

  if width == 0 || height == 0 {
    return Err(deferred_sof_dimension());
  }

  Ok(SofFrame {
    width,
    height,
    precision,
  })
}

/// Walks marker segments from the leading SOI onward until the first SOF
/// ([`is_sof_marker`]), returning its declared [`SofFrame`]. Refuses the
/// input — never panics — when: the buffer does not start `FF D8`; the
/// walk runs off the end of the buffer before a SOF appears; SOS or EOI is
/// reached first (real encoders always place SOF before both); a DHP
/// marker appears (hierarchical JPEG, see below); a segment's length does
/// not fit the remaining buffer; or the SOF itself declares a zero
/// width/height (see [`read_sof_dimensions`]).
///
/// # Hierarchical JPEG (DHP) is refused, not parsed
///
/// A DHP marker (`0xDE`) introduces JPEG's hierarchical mode (ITU-T T.81
/// Annex J): a sequence of multiple SOF-delimited frames at increasing
/// resolution, with DHP itself — in the *same* precision/height/width/Nf
/// shape as a SOF payload — declaring the dimensions of the *completed*
/// (largest) image. A conforming hierarchical stream can carry an
/// over-cap DHP ahead of an under-cap first SOF, so stopping at the first
/// SOF as usual would budget the small frame and miss the real one.
/// Correctly handling this means either trusting DHP's own declared
/// completed size or walking every frame in the sequence for its
/// maximum — both are hierarchical-mode-specific decoder knowledge this
/// preflight deliberately does not carry (it is a SOF preflight, not a
/// JPEG decoder). Refusing outright on DHP is the same choice already
/// made for a deferred DNL height: when this walk cannot itself establish
/// a trustworthy bound, it refuses rather than guesses.
///
/// Every step advances `pos` — one byte at a time through [`next_byte`], or
/// by a checked, buffer-fitting segment length through [`skip_segment`] /
/// [`read_sof_dimensions`] — so the walk cannot loop: each iteration either
/// returns or strictly progresses toward `jpeg.len()`, bounding the total
/// work at one pass over the input.
fn find_sof_dimensions(jpeg: &[u8]) -> Result<SofFrame, AnalyzeError> {
  if !jpeg.starts_with(&[0xFF, 0xD8]) {
    return Err(no_sof_marker());
  }
  let mut pos = 2usize;
  loop {
    let code = read_marker_code(jpeg, &mut pos)?;
    match code {
      0xD8 => continue,                                    // stray SOI: no payload
      0x01 | 0xD0..=0xD7 => continue,                      // TEM / RSTn: no payload
      0xD9 | 0xDA => return Err(no_sof_marker()),          // EOI / SOS: no SOF before it
      0xDE => return Err(hierarchical_jpeg_unsupported()), // DHP: see doc comment above
      _ if is_sof_marker(code) => return read_sof_dimensions(jpeg, pos),
      _ => pos = skip_segment(jpeg, pos)?,
    }
  }
}

/// Refuse a JPEG whose SOF marker declares decoded dimensions above
/// [`MAX_DECODED_IMAGE_BYTES`], before `NSData::with_bytes` copies the
/// compressed bytes and Vision/ImageIO allocates buffers proportional to
/// the declared size.
///
/// [`MAX_INPUT_IMAGE_BYTES`] bounds the *compressed* input, and that is
/// not the same bound: a small JPEG can declare gigantic *decoded*
/// dimensions in its SOF marker, and Vision/ImageIO allocates for
/// `width × height` the moment it decodes, before any downstream cap in
/// this crate runs. This is the only place that allocation can be
/// bounded, and it has to be bounded without performing it — hence a
/// marker walk and the worst-case rate [`decoded_bytes_per_pixel`]
/// charges. The pixel door needs none of this: it is handed the decoded
/// bytes, so it measures them directly at [`PixelPlane::new`].
#[inline]
pub(crate) fn check_decoded_dimensions(jpeg: &[u8]) -> Result<(), AnalyzeError> {
  let frame = find_sof_dimensions(jpeg)?;
  let bytes_per_pixel = decoded_bytes_per_pixel(frame.precision);
  let decoded_bytes = u64::from(frame.width) * u64::from(frame.height) * bytes_per_pixel;
  if decoded_bytes > MAX_DECODED_IMAGE_BYTES {
    return Err(AnalyzeError::new(
      AnalyzeErrorKind::RequestFailed,
      "input image declares decoded dimensions above MAX_DECODED_IMAGE_BYTES",
    ));
  }
  Ok(())
}

/// Whether the perform this call asked for actually ran.
///
/// # The invariant
///
/// **A request's `results` may be read only after a `Completed`
/// perform of THIS call.** Every caller of [`perform`] upholds it.
///
/// # Why it has to exist
///
/// A `VNRequest` is RETAINED across calls — that is what lets an entry
/// point own its requests and pay for its models once. The cost is
/// that `results` is not a value the perform returns; it is state
/// sitting on an object that outlived the previous frame. Vision
/// defines it only once the current request has been processed, and
/// gives no clearing postcondition for a request that was not, so
/// after a caught Objective-C exception it may still hold the last
/// SUCCESSFUL call's observations.
///
/// Reading it there would let one frame emit another's geometry. In
/// the face fusion the consequence is sharper than a stale reading:
/// stale observations would become the new spine and be handed to the
/// annotating passes verbatim, which preserve their uuids — so the
/// identity check would see a perfectly valid bijection and pass. The
/// whole identity universe is stale together, and no downstream
/// verification can catch that. Only refusing to read at all can.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Performed {
  /// Vision processed the requests; their `results` describe THIS call.
  Completed,
  /// An Objective-C exception was caught. The requests were not
  /// processed, and — because a request object is retained across
  /// calls — its `results` may still describe a PREVIOUS call. Nothing
  /// on it may be read.
  Raised,
}

/// Perform `requests` against `image` under the Objective-C exception
/// barrier.
///
/// A detector can fail three ways, and each is a different return:
///
/// - a recoverable `NSError` — `Err`, mapped to
///   [`AnalyzeErrorKind::RequestFailed`];
/// - a fatal Objective-C `NSException` that unwinds across the FFI
///   boundary and would otherwise abort the whole process —
///   `Ok(`[`Performed::Raised`]`)`, the barrier's doing;
/// - nothing at all — `Ok(`[`Performed::Completed`]`)`.
///
/// The second is why this returns a [`Performed`] rather than `()`.
/// Reporting a caught exception as a bare `Ok` would make it
/// indistinguishable from a completed perform, and the caller would go
/// on to read `results` off a request that Vision never processed on
/// this call — see [`Performed`] for what is in there.
pub(crate) fn perform(
  handler: &VNSequenceRequestHandler,
  image: &VisionImage,
  requests: &[Retained<VNRequest>],
) -> Result<Performed, AnalyzeError> {
  guard_vision_ffi("performRequests", Ok(Performed::Raised), || unsafe {
    // Inside the barrier, not before it: building the array is a
    // retained clone per request and a Foundation allocation, which is
    // FFI like everything else here.
    let array = NSArray::from_retained_slice(requests);
    match image {
      VisionImage::Encoded(data) => handler.performRequests_onImageData_error(&array, data),
      VisionImage::Decoded(image) => handler.performRequests_onCGImage_error(&array, image),
    }
    .map(|()| Performed::Completed)
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

/// The per-call preamble every entry point runs, without the perform:
/// bound the input, open an autorelease pool, build the image object and
/// the sequence handler, then hand both to `body`.
///
/// Holding the perform out is what lets a caller perform TWICE on one
/// image — [`FaceDetector`](crate::FaceDetector) needs a first pass's
/// observations before it can feed the other two — while every other
/// entry point keeps paying for exactly one, through
/// [`run_requests`].
///
/// The pre-flight runs BEFORE the pool and before anything is copied:
/// [`ImageSource::preflight`] is what stops an over-ceiling JPEG from
/// reaching `NSData::with_bytes`.
///
/// # The preamble is guarded too
///
/// Building the image object and the sequence handler are Objective-C
/// allocations, and `body` is where every perform and every extraction
/// runs, so the whole of it sits inside [`guard_native`]. A native
/// exception from any of it refuses the call with an
/// [`AnalyzeErrorKind::Environment`] error rather than unwinding into a
/// caller that has no barrier of its own. The autorelease pool is
/// inside the guard, so an exception caught by the trampoline has
/// already run the pool's drain on its way out.
pub(crate) fn with_image<R>(
  source: ImageSource<'_>,
  body: impl FnOnce(&VNSequenceRequestHandler, &VisionImage) -> Result<R, AnalyzeError>,
) -> Result<R, AnalyzeError> {
  source.preflight()?;
  guard_native("with_image", || {
    objc2::rc::autoreleasepool(|_| {
      let image = source.prepare()?;
      let handler = unsafe { VNSequenceRequestHandler::new() };
      body(&handler, &image)
    })
  })?
}

/// The whole per-call preamble every single-perform entry point runs:
/// [`with_image`], then perform exactly `requests` (and nothing else —
/// this is what makes a one-capability consumer pay for one
/// capability), then extract.
///
/// `extract` is called only after a [`Performed::Completed`] perform,
/// and reads the results off the very request objects the caller
/// passed in. On [`Performed::Raised`] it is not called at all and
/// `fallback` is returned instead: the requests were not processed on
/// this call, so what sits on them is a previous call's — see
/// [`Performed`].
///
/// That is the degradation [`guard_vision_ffi`] always documented ("an
/// empty result for that detector"); what changes is that the result
/// is now genuinely empty rather than whatever the retained request
/// happened to still hold. `fallback` is therefore the caller's own
/// empty value — the same one its inner extraction guard already
/// passes.
///
/// `source` is the ONLY thing that differs between an entry point's two
/// doors: same requests, same fallback, same extraction, same caps.
pub(crate) fn run_requests<R>(
  source: ImageSource<'_>,
  requests: &[Retained<VNRequest>],
  fallback: R,
  extract: impl FnOnce() -> R,
) -> Result<R, AnalyzeError> {
  with_image(source, |handler, image| {
    match perform(handler, image, requests)? {
      Performed::Completed => Ok(extract()),
      Performed::Raised => Ok(fallback),
    }
  })
}

// ----- the two doors ---------------------------------------------------------

/// What a caller handed an entry point — the whole difference between
/// this crate's two doors.
///
/// The doors diverge here and nowhere else. Both variants become a
/// [`VisionImage`] and go through the same [`perform`], the same
/// exception barrier, the same [`Performed`] discipline, and the same
/// extractor: an entry point's two public methods are one private body
/// that this enum is passed to.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ImageSource<'a> {
  /// Encoded bytes. Vision hands them to ImageIO, which decodes.
  Jpeg(&'a [u8]),
  /// Pixels the caller already decoded. Nothing decodes them again.
  Plane(&'a PixelPlane<'a>),
}

impl ImageSource<'_> {
  /// Refuse an input the engine will not put in front of Vision, before
  /// anything is allocated or copied.
  ///
  /// A JPEG is untrusted bytes and gets both checks — the compressed
  /// ceiling and the SOF decoded-size walk. A plane has nothing left to
  /// check: [`PixelPlane::new`] enforced the geometry, the extent and
  /// the same [`MAX_DECODED_IMAGE_BYTES`] ceiling when the caller
  /// constructed it, and a constructed plane is immutable.
  fn preflight(&self) -> Result<(), AnalyzeError> {
    match self {
      Self::Jpeg(jpeg) => {
        check_input_len(jpeg)?;
        check_decoded_dimensions(jpeg)
      }
      Self::Plane(_) => Ok(()),
    }
  }

  /// Build the Objective-C / Core Graphics object Vision performs
  /// against. Call inside the autorelease pool.
  fn prepare(&self) -> Result<VisionImage, AnalyzeError> {
    match self {
      Self::Jpeg(jpeg) => Ok(VisionImage::Encoded(NSData::with_bytes(jpeg))),
      Self::Plane(plane) => cg_image_from_plane(plane).map(VisionImage::Decoded),
    }
  }
}

/// The prepared image one perform runs against.
#[derive(Debug)]
pub(crate) enum VisionImage {
  /// `performRequests:onImageData:` — Vision decodes.
  Encoded(Retained<NSData>),
  /// `performRequests:onCGImage:` — nothing decodes.
  Decoded(CFRetained<CGImage>),
}

/// The `(bits per pixel, colour space, bitmap info)` Core Graphics needs
/// to read a plane of this format.
///
/// Every mapping is asserted bit-exactly in the test suite by rendering
/// a plane back out through a Core Graphics bitmap context and comparing
/// channels, so a wrong constant here is a failing test rather than a
/// silently colour-swapped detection.
fn cg_layout(format: PixelFormat) -> (usize, Option<CFRetained<CGColorSpace>>, CGBitmapInfo) {
  match format {
    PixelFormat::Rgb8 => (
      24,
      CGColorSpace::new_device_rgb(),
      CGBitmapInfo(CGImageAlphaInfo::None.0),
    ),
    PixelFormat::Rgba8 => (
      32,
      CGColorSpace::new_device_rgb(),
      // "Skip last", not "alpha last": the fourth byte is never read,
      // so no premultiplication is assumed and none is undone.
      CGBitmapInfo(CGImageAlphaInfo::NoneSkipLast.0),
    ),
    PixelFormat::Bgra8 => (
      32,
      CGColorSpace::new_device_rgb(),
      // A little-endian 32-bit word over the bytes `B G R A` reads as
      // `0xAARRGGBB`, so skipping the FIRST component skips the alpha
      // byte and leaves the colour bytes in place.
      CGBitmapInfo(CGImageAlphaInfo::NoneSkipFirst.0 | CGImageByteOrderInfo::Order32Little.0),
    ),
    PixelFormat::Gray8 => (
      8,
      CGColorSpace::new_device_gray(),
      CGBitmapInfo(CGImageAlphaInfo::None.0),
    ),
  }
}

/// A refusal from the pixel door's own allocation path.
#[inline]
fn plane_alloc_refused(message: &'static str) -> AnalyzeError {
  AnalyzeError::new(AnalyzeErrorKind::RequestFailed, message)
}

/// A `CGDataProvider` over ONE Core Foundation copy of the plane's
/// pixels, row padding removed.
///
/// # One copy, and only one
///
/// The engine's addition to the caller's peak memory is exactly one copy
/// of the image, whatever shape the plane arrives in — the same doubling
/// the JPEG door's own [`MAX_INPUT_IMAGE_BYTES`] ceiling is written
/// against ("cannot double the worker's peak memory"). That is the floor
/// for a door that must own its bytes, and reaching it takes two
/// different routes:
///
/// - a **tight** plane — every `PixelPlane::packed` caller — is already
///   the tight buffer, so `CFDataCreate` copies straight out of the
///   caller's slice and nothing of ours exists in between;
/// - a **padded** plane is appended row by row into a `CFMutableData`
///   sized up front, so the padding is dropped during the one copy
///   rather than by a compacting pass that would exist alongside it.
///
/// The intermediate `Vec` those two replace was the bug: for a padded
/// plane at the ceiling it held a third 512 MiB image live beside the
/// caller's and Core Foundation's, and `Vec::with_capacity` is
/// infallible, so failing to get it aborted the process instead of
/// refusing the frame.
///
/// # Every allocation here is fallible
///
/// `CFDataCreate` and `CFDataCreateMutable` report failure as null,
/// which becomes an [`AnalyzeErrorKind::RequestFailed`] like any other
/// refusal. No Rust-side infallible allocation is on this path at all,
/// so a plane the machine cannot hold costs the frame, never the worker.
/// The appended length is verified against the length asked for, so an
/// append that did not land is refused rather than shipped as a
/// truncated image.
///
/// # The arithmetic is bounded by construction
///
/// `PixelPlane::new` established `stride * (height - 1) + row_bytes <=
/// 512 MiB` and `stride >= row_bytes`, so `row_bytes * height <=
/// stride * (height - 1) + row_bytes` — the product cannot overflow and
/// cannot exceed the ceiling. The same statement is what puts every row
/// slice in bounds: the last one ends at exactly that extent, and the
/// plane's buffer is at least that long.
fn plane_provider(plane: &PixelPlane<'_>) -> Result<CFRetained<CGDataProvider>, AnalyzeError> {
  let row_bytes = plane.row_bytes();
  let height = plane.height() as usize;
  let tight_len = row_bytes * height;
  // Bounded by the plane's own ceiling, well inside `CFIndex`, but
  // converted rather than cast so a future ceiling cannot turn a length
  // into a negative index.
  let (Ok(length), Ok(row_length)) = (CFIndex::try_from(tight_len), CFIndex::try_from(row_bytes))
  else {
    return Err(plane_alloc_refused(
      "pixel plane is too large to describe to core foundation",
    ));
  };

  let provider = if plane.stride() == row_bytes {
    // SAFETY: the plane's own buffer is a live slice of at least
    // `tight_len` bytes — `PixelPlane::new` established exactly that —
    // and `CFDataCreate` copies out of it before returning, so nothing
    // of ours is borrowed past this call. `None` is the default
    // allocator, which `CFDataCreate` accepts.
    let data = unsafe { CFData::new(None, plane.data().as_ptr(), length) };
    let Some(data) = data else {
      return Err(plane_alloc_refused(
        "core foundation declined to copy the pixel plane",
      ));
    };
    CGDataProvider::with_cf_data(Some(&data))
  } else {
    let Some(data) = CFMutableData::new(None, length) else {
      return Err(plane_alloc_refused(
        "core foundation declined to allocate for the pixel plane",
      ));
    };
    for row in 0..height {
      let start = row * plane.stride();
      let source = &plane.data()[start..start + row_bytes];
      // SAFETY: `source` is a live slice of exactly `row_length` bytes,
      // read and copied before this returns. The appends total
      // `tight_len`, the capacity the buffer was created with, so none
      // of them asks it to grow past it.
      unsafe { CFMutableData::append_bytes(Some(&data), source.as_ptr(), row_length) };
    }
    if data.length() != length {
      return Err(plane_alloc_refused(
        "core foundation did not take the whole pixel plane",
      ));
    }
    CGDataProvider::with_cf_data(Some(&data))
  };

  provider.ok_or_else(|| {
    plane_alloc_refused("core graphics declined to create a data provider for the pixel plane")
  })
}

/// Wrap a plane in a `CGImage` for `performRequests:onCGImage:`.
///
/// # Why the pixels are copied
///
/// A `CGDataProvider` can be built over borrowed memory, and that would
/// make this door allocation-free. It would also stake the caller's
/// buffer on a lifetime this crate cannot establish: the `CGImage` is
/// handed to Vision, and nothing in Apple's contract says Vision does
/// not retain it past `performRequests`. Rust would have no way to
/// notice, and a read after the borrow ended is undefined behaviour, not
/// a wrong answer.
///
/// So the provider is backed by Core Foundation, which COPIES what it is
/// given and then owns it for as long as anything holds it — Core
/// Graphics, Vision, or this function. The caller's slice is untouched
/// and free the moment this returns, whoever is still holding the image.
/// [`plane_provider`] is where that one copy is made, and why it is
/// exactly one.
///
/// Core Foundation also removes the alternative's whole failure surface:
/// a buffer handed over raw with a release callback would have to be
/// freed by hand on every path where a create returned null, and getting
/// that wrong is a leak in one direction and a double free in the other.
/// Here every object is reference-counted from the moment it exists, so
/// an early return drops what it holds and nothing else.
///
/// # Refusals
///
/// [`AnalyzeErrorKind::RequestFailed`], for a colour space, a pixel
/// copy, a provider, or an image that the frameworks decline to
/// create — never an abort. See [`plane_provider`] for the allocation
/// half.
pub(crate) fn cg_image_from_plane(
  plane: &PixelPlane<'_>,
) -> Result<CFRetained<CGImage>, AnalyzeError> {
  let (bits_per_pixel, colour_space, bitmap_info) = cg_layout(plane.format());
  let Some(colour_space) = colour_space else {
    return Err(AnalyzeError::new(
      AnalyzeErrorKind::RequestFailed,
      "core graphics declined to create the plane's colour space",
    ));
  };

  let provider = plane_provider(plane)?;

  // SAFETY: `decode` is null, which the binding documents as the one
  // alternative to a valid pointer. Every other argument describes the
  // buffer the provider now owns: `row_bytes` bytes per row for
  // `height` rows is exactly `pixels.len()`, so the image cannot
  // describe a byte the provider does not hold.
  let image = unsafe {
    CGImage::new(
      plane.width() as usize,
      plane.height() as usize,
      8,
      bits_per_pixel,
      plane.row_bytes(),
      Some(&colour_space),
      bitmap_info,
      Some(&provider),
      core::ptr::null(),
      false,
      CGColorRenderingIntent::RenderingIntentDefault,
    )
  };
  image.ok_or_else(|| {
    AnalyzeError::new(
      AnalyzeErrorKind::RequestFailed,
      "core graphics declined to create an image for the pixel plane",
    )
  })
}
