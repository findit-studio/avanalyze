//! The Vision FFI boundary every entry point shares.
//!
//! Each entry point in this crate owns its own `VNRequest` objects —
//! that is the whole point of the split. What they cannot own
//! separately is the boundary itself: Apple's lower-left coordinates,
//! its unbounded FFI-reported arrays, and the Objective-C exceptions
//! that must never reach a Rust unwind. Those live here, once.
//!
//! Nothing in this module is public API.

use std::{borrow::Cow, panic::AssertUnwindSafe};

use objc2::{Message, exception::catch as catch_objc_exception, rc::Retained};
use objc2_core_foundation::{CGPoint, CGRect};
use objc2_foundation::{NSArray, NSData, NSDictionary};
use objc2_vision::{VNRequest, VNSequenceRequestHandler};
use smol_str::{SmolStr, ToSmolStr};

use crate::{AnalyzeError, AnalyzeErrorKind, BoundingBox};

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

// ----- the exception barrier ------------------------------------------------

/// Run `f` under an Objective-C exception barrier, returning `fallback`
/// (the empty/degraded result for that detector) if Apple's Vision
/// framework raises an `NSException` that unwinds across the FFI
/// boundary.
///
/// Rust's [`std::panic::catch_unwind`] (used in the 3-D body-pose path
/// for a *Rust*-panic quirk) explicitly **cannot** catch a foreign
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
  }
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

/// Perform `requests` against `data` under the Objective-C exception
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
  data: &NSData,
  requests: &[Retained<VNRequest>],
) -> Result<Performed, AnalyzeError> {
  let array = NSArray::from_retained_slice(requests);
  guard_vision_ffi("performRequests", Ok(Performed::Raised), || unsafe {
    handler
      .performRequests_onImageData_error(&array, data)
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
/// bound the input, open an autorelease pool, wrap the JPEG, build the
/// sequence handler, then hand both to `body`.
///
/// Holding the perform out is what lets a caller perform TWICE on one
/// image — [`FaceDetector`](crate::FaceDetector) needs a first pass's
/// observations before it can feed the other two — while every other
/// entry point keeps paying for exactly one, through
/// [`run_requests`].
pub(crate) fn with_image<R>(
  jpeg: &[u8],
  body: impl FnOnce(&VNSequenceRequestHandler, &NSData) -> Result<R, AnalyzeError>,
) -> Result<R, AnalyzeError> {
  check_input_len(jpeg)?;
  objc2::rc::autoreleasepool(|_| {
    let ns_data = NSData::with_bytes(jpeg);
    let handler = unsafe { VNSequenceRequestHandler::new() };
    body(&handler, &ns_data)
  })
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
pub(crate) fn run_requests<R>(
  jpeg: &[u8],
  requests: &[Retained<VNRequest>],
  fallback: R,
  extract: impl FnOnce() -> R,
) -> Result<R, AnalyzeError> {
  with_image(jpeg, |handler, data| {
    match perform(handler, data, requests)? {
      Performed::Completed => Ok(extract()),
      Performed::Raised => Ok(fallback),
    }
  })
}
