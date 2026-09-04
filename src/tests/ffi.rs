//! The shared Vision FFI boundary: coordinate conversions, the
//! bounded-read gates, and the exception barrier.
//!
//! These run against the reference vocabulary (`mediaschema`) so the
//! assertions exercise a real validating implementation rather than a
//! permissive stub.

use core::ptr::NonNull;
use mediaschema::domain::aggregates::video::BoundingBox as DomainBoundingBox;

use objc2::{AnyThread, DefinedClass, Message, msg_send, rc::Retained, runtime::AnyObject};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_foundation::{
  NSArray, NSDictionary, NSEnumerator, NSFastEnumeration, NSFastEnumerationState, NSString,
  ns_string,
};

use crate::{
  AnalyzeErrorKind,
  face_landmarks::MAX_LANDMARK_POINTS,
  ffi::{
    ImageSource, MAX_POSE_JOINT_ATTEMPTS_PER_CALL, MAX_POSE_JOINT_NAME_BYTES_PER_CALL,
    MAX_POSE_JOINTS, MAX_POSE_JOINTS_PER_CALL, MAX_VISION_RESULTS_PER_FRAME, Performed, PoseBudget,
    PoseJoints, check_decoded_dimensions, collect_dictionary_pairs, finite_f32, guard_vision_ffi,
    pose_bbox_from_joint_bounds, project_landmark_to_image, read_pose_joints, run_requests,
    validate_raw_slice_bytes, validate_raw_slice_elems, vision_point_to_normalized,
    vision_rect_to_bbox, vn_point3d_position, with_image,
  },
  person_mask::MAX_MASK_BYTES,
  plane::MAX_DECODED_IMAGE_BYTES,
};

/// `vision_rect_to_bbox` must flip y. A Vision rect of
/// `(0.1, 0.2, 0.3, 0.4)` (lower-left origin) maps to
/// `(0.1, 1.0 - (0.2 + 0.4), 0.3, 0.4)` = `(0.1, 0.4, 0.3, 0.4)`
/// in the contract's top-left convention.
#[test]
fn vision_rect_to_bbox_flips_y() {
  let rect = CGRect::new(CGPoint::new(0.1, 0.2), CGSize::new(0.3, 0.4));
  let bbox =
    vision_rect_to_bbox::<DomainBoundingBox>(rect).expect("in-range rect must clamp to itself");
  assert!((bbox.x() - 0.1).abs() < 1e-6, "x: {}", bbox.x());
  assert!((bbox.y() - 0.4).abs() < 1e-6, "y: {}", bbox.y());
  assert!((bbox.width() - 0.3).abs() < 1e-6, "w: {}", bbox.width());
  assert!((bbox.height() - 0.4).abs() < 1e-6, "h: {}", bbox.height());
}

/// Lock the flipped full-image result against the validating domain
/// `BoundingBox::try_new` to ensure the components still satisfy the
/// `[0, 1]` invariant after the flip.
#[test]
fn vision_rect_to_bbox_full_image_round_trip() {
  let rect = CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(1.0, 1.0));
  let bbox =
    vision_rect_to_bbox::<DomainBoundingBox>(rect).expect("unit rect must clamp to itself");
  assert_eq!(bbox.x(), 0.0);
  assert_eq!(bbox.y(), 0.0);
  assert_eq!(bbox.width(), 1.0);
  assert_eq!(bbox.height(), 1.0);
  DomainBoundingBox::try_new(bbox.x(), bbox.y(), bbox.width(), bbox.height())
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
  let bbox =
    vision_rect_to_bbox::<DomainBoundingBox>(rect).expect("partial overlap must produce a bbox");
  // Clamped right edge is 1.0 → width = 0.2 (1.0 - 0.8).
  assert!((bbox.x() - 0.8).abs() < 1e-6, "x: {}", bbox.x());
  assert!((bbox.width() - 0.2).abs() < 1e-6, "w: {}", bbox.width());
  // y in schema space: 1.0 - (0.4 + 0.2) = 0.4 (in-range, no clamp).
  assert!((bbox.y() - 0.4).abs() < 1e-6, "y: {}", bbox.y());
  assert!((bbox.height() - 0.2).abs() < 1e-6, "h: {}", bbox.height());
  DomainBoundingBox::try_new(bbox.x(), bbox.y(), bbox.width(), bbox.height())
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
  let bbox =
    vision_rect_to_bbox::<DomainBoundingBox>(rect).expect("partial overlap must produce a bbox");
  // Bottom clamped to 1.0 → height = 1.0 - 0.7 = 0.3.
  assert!((bbox.x() - 0.1).abs() < 1e-6, "x: {}", bbox.x());
  assert!((bbox.y() - 0.7).abs() < 1e-6, "y: {}", bbox.y());
  assert!((bbox.width() - 0.3).abs() < 1e-6, "w: {}", bbox.width());
  assert!((bbox.height() - 0.3).abs() < 1e-6, "h: {}", bbox.height());
  DomainBoundingBox::try_new(bbox.x(), bbox.y(), bbox.width(), bbox.height())
    .expect("clamped bbox satisfies the [0,1] invariant");
}

/// A Vision rect entirely outside the unit square must yield `None`
/// so the detection is skipped rather than producing a degenerate
/// wire bbox.
#[test]
fn vision_bbox_fully_offscreen_yields_none() {
  let rect = CGRect::new(CGPoint::new(1.5, 0.5), CGSize::new(0.3, 0.4));
  assert!(vision_rect_to_bbox::<DomainBoundingBox>(rect).is_none());
}

/// A Vision rect that intersects the unit square only at a single
/// edge must yield `None` (the intersection has zero width).
#[test]
fn vision_bbox_edge_only_yields_none() {
  // Right edge at exactly x = 1.0, left edge at x = 1.0 — zero width.
  let rect = CGRect::new(CGPoint::new(1.0, 0.5), CGSize::new(0.0, 0.4));
  assert!(vision_rect_to_bbox::<DomainBoundingBox>(rect).is_none());
}

/// `NaN` from Vision (occasionally seen for off-image rects) must
/// not propagate: the rectangle is dropped rather than sanitised into
/// an edge-aligned box that downstream validation would accept.
#[test]
fn vision_bbox_handles_nan_origin() {
  let rect = CGRect::new(CGPoint::new(f64::NAN, 0.0), CGSize::new(0.3, 0.4));
  assert!(vision_rect_to_bbox::<DomainBoundingBox>(rect).is_none());
}

/// The same holds for a `NaN` in the y origin.
#[test]
fn vision_bbox_handles_nan_y_origin() {
  let rect = CGRect::new(CGPoint::new(0.1, f64::NAN), CGSize::new(0.3, 0.4));
  assert!(vision_rect_to_bbox::<DomainBoundingBox>(rect).is_none());
}

/// 2D points flip y AND clamp to `[0, 1]`. A Vision point that lands
/// outside `[0, 1]` after the flip is clamped to the unit edge so
/// downstream validation accepts it.
#[test]
fn vision_point_to_normalized_flips_y_only() {
  let (x, y) = vision_point_to_normalized(0.25, 0.75).expect("finite point");
  assert!((x - 0.25).abs() < 1e-6);
  assert!((y - 0.25).abs() < 1e-6);
}

/// Out-of-range Vision points clamp to `[0, 1]`.
#[test]
fn vision_point_to_normalized_clamps_out_of_range() {
  let (x, y) = vision_point_to_normalized(1.2, -0.3).expect("finite point");
  assert_eq!(x, 1.0);
  // `y = 1.0 - (-0.3) = 1.3` → clamped to 1.0.
  assert_eq!(y, 1.0);
}

/// Non-finite Vision points are rejected at the source: a `NaN`,
/// `+Inf`, or `-Inf` in either component returns `None` so the
/// caller can decide whether to drop the point or the whole
/// detection. Sanitising the bad component to `0.0` would fabricate
/// edge-aligned coordinates the domain validator could not
/// distinguish from real detections.
#[test]
fn vision_point_to_normalized_rejects_non_finite() {
  assert!(vision_point_to_normalized(f64::NAN, 0.5).is_none());
  assert!(vision_point_to_normalized(0.5, f64::NAN).is_none());
  assert!(vision_point_to_normalized(f64::INFINITY, 0.5).is_none());
  assert!(vision_point_to_normalized(0.5, f64::INFINITY).is_none());
  assert!(vision_point_to_normalized(f64::NEG_INFINITY, 0.5).is_none());
  assert!(vision_point_to_normalized(0.5, f64::NEG_INFINITY).is_none());
  // Finite path still works.
  assert!(vision_point_to_normalized(0.1, 0.2).is_some());
}

/// A document quad with even one non-finite corner must be dropped
/// in its entirety — a quad is geometrically meaningless without
/// all four corners. This mirrors the per-detection pattern the
/// extractor uses (`let (Some(tl), Some(tr), Some(bl), Some(br)) =
/// (...) else { continue; }`): if any corner returns `None`, the
/// whole quad is rejected. Partial-corner emission would be a
/// regression.
#[test]
fn document_quad_with_non_finite_corner_is_dropped() {
  let good = (0.1_f64, 0.1_f64);
  let bad = (f64::NAN, 0.5_f64);

  for (tl, tr, bl, br) in [
    (bad, good, good, good),
    (good, bad, good, good),
    (good, good, bad, good),
    (good, good, good, bad),
  ] {
    let result = (
      vision_point_to_normalized(tl.0, tl.1),
      vision_point_to_normalized(tr.0, tr.1),
      vision_point_to_normalized(bl.0, bl.1),
      vision_point_to_normalized(br.0, br.1),
    );
    assert!(
      !matches!(result, (Some(_), Some(_), Some(_), Some(_))),
      "quad with non-finite corner survived: {result:?}",
    );
  }
}

/// A document quad whose corners survive per-coord clamp but
/// collapse to a degenerate shape (e.g. all four corners coincident)
/// must be rejected by the domain validator, which the extractor runs
/// pre-emission.
#[test]
fn document_quad_with_collapsed_corners_is_rejected_by_domain() {
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
    mediaschema::domain::aggregates::video::DocumentSegment::try_new(tl, tr, br, bl, 0.9).is_err()
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
  let face = CGRect::new(CGPoint::new(0.2, 0.3), CGSize::new(0.4, 0.2));
  let projected = project_landmark_to_image(CGPoint::new(0.0, 0.0), face);
  let (sx, sy) =
    vision_point_to_normalized(projected.x, projected.y).expect("projected lower-left is finite");
  assert!((sx - 0.2).abs() < 1e-6, "schema-x: {sx}");
  // Vision lower-left at face y = 0.3 → schema-y = 1.0 - 0.3 = 0.7.
  assert!((sy - 0.7).abs() < 1e-6, "schema-y: {sy}");
}

/// A non-finite landmark component drops the offending point at
/// the schema-flip stage even when the face bbox is well-formed:
/// `project_landmark_to_image` propagates the non-finite component
/// (`0.2 + NaN * 0.4 = NaN`) and `vision_point_to_normalized`
/// rejects it.
#[test]
fn projected_non_finite_landmark_is_rejected() {
  let face = CGRect::new(CGPoint::new(0.2, 0.3), CGSize::new(0.4, 0.2));
  let projected = project_landmark_to_image(CGPoint::new(f64::NAN, 0.5), face);
  assert!(vision_point_to_normalized(projected.x, projected.y).is_none());
}

/// A pose with only one surviving joint cannot derive a non-degenerate
/// bbox. The helper must report `None` so the pose extractor skips
/// it instead of emitting a zero-extent box that the domain
/// validator would reject.
#[test]
fn pose_bbox_from_single_joint_yields_none() {
  assert!(pose_bbox_from_joint_bounds::<DomainBoundingBox>(0.5, 0.5, 0.5, 0.5).is_none());
}

/// A pose where every joint shares the same x (perfectly vertical
/// limbs) has zero-width bbox and must be reported as `None`.
#[test]
fn pose_bbox_from_vertical_joints_yields_none() {
  assert!(pose_bbox_from_joint_bounds::<DomainBoundingBox>(0.5, 0.1, 0.5, 0.9).is_none());
}

/// A pose where every joint shares the same y has zero-height bbox
/// and must be reported as `None`.
#[test]
fn pose_bbox_from_horizontal_joints_yields_none() {
  assert!(pose_bbox_from_joint_bounds::<DomainBoundingBox>(0.1, 0.5, 0.9, 0.5).is_none());
}

/// A pose with at least one joint per axis produces a valid bbox.
#[test]
fn pose_bbox_from_diagonal_joints_is_valid() {
  let bbox = pose_bbox_from_joint_bounds::<DomainBoundingBox>(0.1, 0.2, 0.4, 0.6)
    .expect("non-degenerate joints yield Some");
  assert!((bbox.x() - 0.1).abs() < 1e-6);
  assert!((bbox.y() - 0.2).abs() < 1e-6);
  assert!((bbox.width() - 0.3).abs() < 1e-6);
  assert!((bbox.height() - 0.4).abs() < 1e-6);
  DomainBoundingBox::try_new(bbox.x(), bbox.y(), bbox.width(), bbox.height())
    .expect("pose-derived bbox satisfies domain invariants");
}

/// Non-finite joint coordinates (NaN/Inf from a glitched Vision
/// observation) must short-circuit before reaching the
/// `BoundingBox::try_new` constructor.
#[test]
fn pose_bbox_from_nan_joints_yields_none() {
  assert!(pose_bbox_from_joint_bounds::<DomainBoundingBox>(f32::NAN, 0.5, 0.5, 0.5).is_none());
  assert!(pose_bbox_from_joint_bounds::<DomainBoundingBox>(0.1, 0.1, f32::INFINITY, 0.5).is_none());
}

/// `validate_raw_slice_bytes` rejects payloads above the cap and
/// above `isize::MAX`, in either order. Re-uses `MAX_MASK_BYTES`
/// as a representative caller-side ceiling; the helper is generic
/// and the cap value itself is not load-bearing for this test.
#[test]
fn validate_raw_slice_bytes_rejects_over_cap() {
  assert!(validate_raw_slice_bytes(0, MAX_MASK_BYTES).is_some());
  assert!(validate_raw_slice_bytes(MAX_MASK_BYTES, MAX_MASK_BYTES).is_some());
  assert!(validate_raw_slice_bytes(MAX_MASK_BYTES + 1, MAX_MASK_BYTES).is_none());
}

/// `validate_raw_slice_bytes` rejects `byte_len > isize::MAX` even
/// when the caller's cap is `usize::MAX` (i.e. no cap). This pins
/// the FFI-side `from_raw_parts` contract independently of the
/// caller-side ceiling.
#[test]
fn validate_raw_slice_bytes_rejects_isize_overflow() {
  assert!(validate_raw_slice_bytes(isize::MAX as usize, usize::MAX).is_some());
  assert!(validate_raw_slice_bytes((isize::MAX as usize).wrapping_add(1), usize::MAX).is_none());
}

/// `validate_raw_slice_elems::<CGPoint>` rejects element counts
/// above the caller-provided max regardless of the size_of math.
#[test]
fn validate_raw_slice_elems_rejects_over_cap() {
  assert!(validate_raw_slice_elems::<CGPoint>(MAX_LANDMARK_POINTS, MAX_LANDMARK_POINTS).is_some());
  assert!(
    validate_raw_slice_elems::<CGPoint>(MAX_LANDMARK_POINTS + 1, MAX_LANDMARK_POINTS).is_none()
  );
}

/// `validate_raw_slice_elems` rejects when `elem_count *
/// size_of::<T>()` overflows usize.
#[test]
fn validate_raw_slice_elems_rejects_byte_overflow() {
  assert!(validate_raw_slice_elems::<CGPoint>(usize::MAX, usize::MAX).is_none());
}

// ----- the bounded pose-dictionary reader -----------------------------------
//
// `collect_dictionary_pairs` takes the dictionary's self-reported count
// as an argument precisely so a test can LIE about it. That is not a
// convenience: the count is the untrusted input — the one value the
// reader exists to disbelieve — so injecting it drives a genuine
// count-versus-enumeration mismatch at the real trust boundary, over a
// real Foundation dictionary, with no unsound subclass in the way.
//
// What the mismatch stands for: the replaced `NSDictionary::to_vecs()`
// allocated two vectors of exactly the reported count, filled them with
// the deprecated unbounded `getObjects:andKeys:`, and then `set_len`'d
// both to that same count. A count too LOW is a write past the
// allocation; a count too HIGH is `set_len` over uninitialised
// pointers. Each direction below is one of those two halves.

/// A real three-entry `NSDictionary<NSString, NSString>` standing in
/// for a Vision joint dictionary. Every value is its key with a `v:`
/// prefix, so the pairing is checkable from the returned pair alone —
/// independently of what order the dictionary chooses to enumerate in.
fn three_joint_dictionary() -> Retained<NSDictionary<NSString, NSString>> {
  NSDictionary::from_slices(
    &[ns_string!("neck"), ns_string!("nose"), ns_string!("root")],
    &[
      ns_string!("v:neck"),
      ns_string!("v:nose"),
      ns_string!("v:root"),
    ],
  )
}

/// The same shape at any size: `entries` distinct joint names, each
/// carrying its own `v:`-prefixed value. The budget cases need a
/// dictionary long enough for one observation's walk to be worth
/// bounding.
fn joint_dictionary_of(entries: usize) -> Retained<NSDictionary<NSString, NSString>> {
  let keys: Vec<Retained<NSString>> = (0..entries)
    .map(|index| NSString::from_str(&format!("joint{index}")))
    .collect();
  let values: Vec<Retained<NSString>> = (0..entries)
    .map(|index| NSString::from_str(&format!("v:joint{index}")))
    .collect();
  let key_refs: Vec<&NSString> = keys.iter().map(|key| &**key).collect();
  let value_refs: Vec<&NSString> = values.iter().map(|value| &**value).collect();
  NSDictionary::from_slices(&key_refs, &value_refs)
}

/// Which of the three outcomes came back, as a name an assertion can
/// print. [`PoseJoints`] carries `Retained` payloads, so deriving
/// `Debug` on it would bind every key and value type in the engine to
/// `Debug` for the sake of the test suite.
fn outcome_name<K: Message, V: Message>(outcome: &PoseJoints<K, V>) -> &'static str {
  match outcome {
    PoseJoints::Read(_) => "Read",
    PoseJoints::Malformed => "Malformed",
    PoseJoints::Exhausted => "Exhausted",
  }
}

/// The pairs a read produced, or a panic naming the refusal that came
/// back instead of them.
fn expect_read<K: Message, V: Message>(
  outcome: PoseJoints<K, V>,
  why: &str,
) -> Vec<(Retained<K>, Retained<V>)> {
  let name = outcome_name(&outcome);
  match outcome {
    PoseJoints::Read(pairs) => pairs,
    _ => panic!("{why}: the read came back {name}"),
  }
}

/// How many joint visits `budget` has left — measured the only way a
/// caller can measure it, by spending them. Leaves the budget exhausted,
/// so it is the last thing a test asks of one.
///
/// This is what turns "the reader charged for what it walked" into an
/// exact number: a read that walked `n` entries leaves
/// `MAX_POSE_JOINT_ATTEMPTS_PER_CALL - n` here.
fn drain_visits(budget: &mut PoseBudget) -> usize {
  // One iteration past the ceiling, so a counter that refuses to be
  // exhausted — a wrapping subtraction, a charge that does not land —
  // fails here rather than spinning forever. `visits` is what the budget
  // has already paid out when the refusal arrives.
  for visits in 0..=MAX_POSE_JOINT_ATTEMPTS_PER_CALL {
    if !budget.charge_joint_visit() {
      return visits;
    }
  }
  panic!("the attempt budget admitted more than MAX_POSE_JOINT_ATTEMPTS_PER_CALL joint visits");
}

/// An honest count reads clean, and every key is paired with the value
/// actually stored under IT — not with whatever sat at the same index
/// in a parallel array. `to_vecs()` returned two vectors that the call
/// sites `zip`ped, i.e. they assumed `keys[i]` belonged with
/// `values[i]`; keyed lookup removes that assumption, so the pairing is
/// asserted here rather than just the count.
///
/// `read_pose_joints` — the entry point the four pose sites actually
/// call, which sources the count from the dictionary itself — is
/// asserted alongside, so the honest path is pinned end to end.
#[test]
fn collect_dictionary_pairs_reads_an_honest_count_and_pairs_by_key() {
  let dict = three_joint_dictionary();

  let pairs = expect_read(
    collect_dictionary_pairs(&dict, 3, MAX_POSE_JOINTS, &mut PoseBudget::new()),
    "a count that matches the enumeration is read, not refused",
  );
  assert_eq!(pairs.len(), 3);
  for (key, value) in &pairs {
    assert_eq!(
      value.to_string(),
      format!("v:{key}"),
      "each key must carry the value stored under it"
    );
  }

  let via_entry_point = expect_read(
    read_pose_joints(&dict, MAX_POSE_JOINTS, &mut PoseBudget::new()),
    "the real entry point reads a sound dictionary exactly as before",
  );
  assert_eq!(via_entry_point.len(), 3);
  for (key, value) in &via_entry_point {
    assert_eq!(value.to_string(), format!("v:{key}"));
  }
}

/// The count claims MORE than the dictionary enumerates: three entries,
/// `reported = 5`. This is the under-enumeration half — `to_vecs()`
/// would have `set_len(5)` over three initialised pointers and handed
/// two uninitialised ones to the joint loop. The reader refuses the
/// whole read instead, which at the call site drops the pose.
#[test]
fn collect_dictionary_pairs_refuses_a_count_claiming_more_than_it_enumerates() {
  let dict = three_joint_dictionary();
  assert_eq!(
    outcome_name(&collect_dictionary_pairs(
      &dict,
      5,
      MAX_POSE_JOINTS,
      &mut PoseBudget::new()
    )),
    "Malformed",
    "a count above what the dictionary enumerates must fail closed"
  );
}

/// The count claims FEWER than the dictionary enumerates: three
/// entries, `reported = 1`. This is the half that would have been a
/// buffer overrun — `to_vecs()` sized its allocations to 1 and let the
/// unbounded `getObjects:andKeys:` write three.
///
/// The refusal happens BEFORE any push past the allocation: the loop
/// tests `pairs.len() == reported` at the top of each iteration, so the
/// vector never grows beyond the capacity `reported` bought. That
/// ordering is not observable from the return value — the trailing
/// `pairs.len() != reported` check would refuse this dictionary too,
/// just after over-growing — which is exactly why it is asserted in
/// prose here rather than left to the assertion below. It is the
/// structural analogue of the overrun `to_vecs()` actually suffered,
/// where growing past the reported count was a write past a raw
/// allocation rather than a `Vec` realloc.
#[test]
fn collect_dictionary_pairs_refuses_a_count_claiming_fewer_than_it_enumerates() {
  let dict = three_joint_dictionary();
  assert_eq!(
    outcome_name(&collect_dictionary_pairs(
      &dict,
      1,
      MAX_POSE_JOINTS,
      &mut PoseBudget::new()
    )),
    "Malformed",
    "a count below what the dictionary enumerates must fail closed, before the over-push"
  );
}

/// Over the cap the pose is DROPPED, never truncated. This is the
/// pre-existing `points_by_joint.len() > MAX_POSE_JOINTS { continue; }`
/// contract, carried across unchanged — the reader returns `None` and
/// the call site's `continue` discards the whole observation rather
/// than emitting a pose built from the first `MAX_POSE_JOINTS` joints.
///
/// Two dictionaries, because the first alone would not isolate the cap.
/// At `reported = MAX_POSE_JOINTS + 1` over a three-entry dictionary
/// the count/enumeration reconciliation would refuse the read anyway,
/// so a second, HONEST dictionary is read against a cap one below its
/// own size: four entries, `reported = 4`, `max_entries = 3`. Nothing
/// there is inconsistent — only oversized — so the cap guard is the
/// only thing that can refuse it, and this is the assertion that fails
/// if the reader ever starts truncating to fit.
#[test]
fn collect_dictionary_pairs_refuses_a_count_over_the_cap() {
  let dict = three_joint_dictionary();
  assert_eq!(
    outcome_name(&collect_dictionary_pairs(
      &dict,
      MAX_POSE_JOINTS + 1,
      MAX_POSE_JOINTS,
      &mut PoseBudget::new()
    )),
    "Malformed",
    "over the cap the pose is dropped, not truncated"
  );

  let oversized = NSDictionary::from_slices(
    &[
      ns_string!("neck"),
      ns_string!("nose"),
      ns_string!("root"),
      ns_string!("tail"),
    ],
    &[
      ns_string!("v:neck"),
      ns_string!("v:nose"),
      ns_string!("v:root"),
      ns_string!("v:tail"),
    ],
  );
  assert_eq!(
    outcome_name(&collect_dictionary_pairs(
      &oversized,
      4,
      3,
      &mut PoseBudget::new()
    )),
    "Malformed",
    "a perfectly consistent dictionary one entry over the cap is still dropped whole"
  );
}

/// An empty dictionary is a legitimate read, not a refusal: an
/// observation with no recognised joints yields an empty pair list, and
/// the call site's own `joints.is_empty()` guard decides what that
/// means. Returning `None` here would conflate "nothing to report" with
/// "malformed".
#[test]
fn collect_dictionary_pairs_reads_an_empty_dictionary() {
  let dict = NSDictionary::<NSString, NSString>::new();
  let pairs = expect_read(
    collect_dictionary_pairs(&dict, 0, MAX_POSE_JOINTS, &mut PoseBudget::new()),
    "an empty dictionary reads empty",
  );
  assert!(pairs.is_empty());
}

/// The bound is INCLUSIVE: a dictionary whose size equals `max_entries`
/// exactly still reads in full. Only `reported > max_entries` refuses,
/// so tightening the cap to the joint count itself does not silently
/// start dropping the last joint — or the whole pose.
#[test]
fn collect_dictionary_pairs_bound_is_inclusive_at_the_cap() {
  let dict = three_joint_dictionary();
  let pairs = expect_read(
    collect_dictionary_pairs(&dict, 3, 3, &mut PoseBudget::new()),
    "a dictionary exactly at the cap is read, not refused",
  );
  assert_eq!(pairs.len(), 3, "the cap is inclusive and does not truncate");
}

// ----- a dictionary that genuinely lies -------------------------------------
//
// The injected-count cases above drive the decision directly. These drive
// `read_pose_joints` — the function the four pose sites actually
// call, which asks the OBJECT for its count — against a real Objective-C
// dictionary whose `count` disagrees with what it enumerates. Nothing is
// injected and nothing is faked: the lie is answered by the object, over
// the real message-send boundary, exactly as a malformed Vision
// dictionary would answer it.
//
// This is sound, not a trick. Apple documents `NSDictionary` as
// subclassable on three primitives — `count`, `objectForKey:` and
// `keyEnumerator` — and all three are implemented below, alongside a
// fast enumeration forwarded whole to a real dictionary (see
// `__count_by_enumerating` for why that fourth method is what lets the
// fixture lie downward at all). The subclass therefore IS an
// `NSDictionary`, holds only the `NSString` keys and values its objc2
// generic parameters promise, and writes no buffer of its own — every
// pointer the enumeration touches is filled by Foundation. objc2 reads
// the lying `count` only through `Iterator::size_hint`, which is a hint
// and cannot be relied on for safety, and the macro-generated `dealloc`
// releases the retained backing dictionary.
//
// The object is INCOHERENT — its count does not describe its contents —
// and that incoherence is the entire point: it is the malformed
// dictionary the finding describes, built without one unsound
// operation.

/// Instance state for [`LyingJointDictionary`]: a real dictionary to
/// enumerate and answer lookups from, and the count to lie with.
struct LyingJointDictionaryIvars {
  backing: Retained<NSDictionary<NSString, NSString>>,
  lied_count: usize,
}

objc2::define_class!(
  // SAFETY: `NSDictionary` is documented as subclassable provided the
  // subclass implements `count`, `objectForKey:` and `keyEnumerator`,
  // which the impl block below does — plus `countByEnumeratingWithState:
  // objects:count:`, forwarded whole rather than derived, justified at
  // its own definition. The class adds no state the superclass owns —
  // its backing dictionary is a private ivar, and the generated
  // `dealloc` drops it.
  #[unsafe(super(NSDictionary<NSString, NSString>))]
  #[name = "AvanalyzeLyingJointDictionary"]
  #[ivars = LyingJointDictionaryIvars]
  struct LyingJointDictionary;

  impl LyingJointDictionary {
    /// The lie: the FFI-reported count the reader refuses to trust,
    /// answered here by a real object rather than passed in beside one.
    #[unsafe(method(count))]
    fn __count(&self) -> usize {
      self.ivars().lied_count
    }

    /// Honest — keyed lookup resolves against the real backing
    /// dictionary, so only the COUNT is corrupt. That is the shape of
    /// the malformed dictionary the finding describes.
    #[unsafe(method_id(objectForKey:))]
    fn __object_for_key(&self, key: &NSString) -> Option<Retained<NSString>> {
      self.ivars().backing.objectForKey(key)
    }

    /// Also honest: the third primitive Apple requires of an
    /// `NSDictionary` subclass.
    #[unsafe(method_id(keyEnumerator))]
    fn __key_enumerator(&self) -> Retained<NSEnumerator<NSString>> {
      // SAFETY: the backing dictionary is immutable and privately owned
      // by this instance, so nothing can mutate it while the returned
      // enumerator is alive.
      unsafe { self.ivars().backing.keyEnumerator() }
    }

    /// Fast enumeration, forwarded WHOLE to the backing dictionary —
    /// and this override is what makes the fixture able to lie in both
    /// directions.
    ///
    /// Left to Foundation, an `NSDictionary` subclass's fast
    /// enumeration is DERIVED from `keyEnumerator` and clamped to the
    /// subclass's own `count`: measured on this host, a fixture with
    /// three real entries and `count` lying "1" enumerated its
    /// `keyEnumerator` three times but fast-enumerated exactly once.
    /// Foundation makes the count and the enumeration agree, so the
    /// downward lie — the dangerous one, the buffer overrun — could not
    /// be expressed at all.
    ///
    /// Forwarding hands the caller's state and buffer to the real
    /// three-entry dictionary, which fills them under its own contract
    /// and honours the caller's `len`. No buffer is written by this
    /// fixture, and every subsequent call in one enumeration forwards
    /// to the same object with the same state, so the enumeration is
    /// exactly "enumerate the backing dictionary" — while `count`
    /// still says whatever it was told to say.
    #[unsafe(method(countByEnumeratingWithState:objects:count:))]
    fn __count_by_enumerating(
      &self,
      state: NonNull<NSFastEnumerationState>,
      buffer: NonNull<*mut AnyObject>,
      len: usize,
    ) -> usize {
      // SAFETY: `state` and `buffer` are the caller's, valid for `len`
      // by the fast-enumeration contract, and are passed through
      // unmodified to a real `NSDictionary` that upholds that contract.
      unsafe {
        self
          .ivars()
          .backing
          .countByEnumeratingWithState_objects_count(state, buffer, len)
      }
    }
  }
);

impl LyingJointDictionary {
  /// A three-entry dictionary that reports `lied_count` as its size.
  fn with_lied_count(lied_count: usize) -> Retained<Self> {
    Self::over(three_joint_dictionary(), lied_count)
  }

  /// `backing`'s real entries, under a `count` of `lied_count`.
  fn over(
    backing: Retained<NSDictionary<NSString, NSString>>,
    lied_count: usize,
  ) -> Retained<Self> {
    let this = Self::alloc().set_ivars(LyingJointDictionaryIvars {
      backing,
      lied_count,
    });
    // SAFETY: `-[NSDictionary init]` is the superclass initialiser for a
    // subclass instance, and the ivars are set before it runs.
    unsafe { msg_send![super(this), init] }
  }
}

/// The subclass really does lie, and telling the truth still reads
/// clean. This pins the fixture before the two refusals lean on it:
/// were `count` honest, or the enumeration empty, the two tests below
/// would pass for the wrong reason.
#[test]
fn a_lying_dictionary_reads_clean_when_its_count_is_honest() {
  let honest = LyingJointDictionary::with_lied_count(3);
  let as_dict: &NSDictionary<NSString, NSString> = &honest;
  assert_eq!(as_dict.len(), 3, "the fixture answers count over the FFI");
  assert_eq!(
    as_dict.keys().count(),
    3,
    "and enumerates the three entries it really holds"
  );

  let pairs = expect_read(
    read_pose_joints(as_dict, MAX_POSE_JOINTS, &mut PoseBudget::new()),
    "a dictionary telling the truth about itself reads exactly as before",
  );
  assert_eq!(pairs.len(), 3);
  for (key, value) in &pairs {
    assert_eq!(value.to_string(), format!("v:{key}"));
  }
}

/// A real object claiming MORE than it enumerates — `count` says 5 over
/// three entries — is refused end to end, through the same
/// `read_pose_joints` the four pose sites call. The lie is kept
/// UNDER `MAX_POSE_JOINTS` deliberately: a lie above the cap would be
/// caught by the size guard, and this case exists to show the
/// count-versus-enumeration reconciliation catching what no size guard
/// could. `to_vecs()` on this object would have `set_len(5)` over three
/// initialised pointers and handed two uninitialised ones onward.
#[test]
fn a_lying_dictionary_claiming_more_than_it_enumerates_is_refused() {
  let lying = LyingJointDictionary::with_lied_count(5);
  let as_dict: &NSDictionary<NSString, NSString> = &lying;
  assert_eq!(as_dict.len(), 5, "the object really does over-report");
  assert_eq!(
    outcome_name(&read_pose_joints(
      as_dict,
      MAX_POSE_JOINTS,
      &mut PoseBudget::new()
    )),
    "Malformed",
    "an over-reporting dictionary must fail closed at the real entry point"
  );
}

/// The other direction, and the dangerous one: `count` says 1 over three
/// entries. `to_vecs()` would have sized both allocations to a single
/// pointer and let the unbounded `getObjects:andKeys:` write three —
/// the heap corruption the finding names. The reader stops at the second
/// entry instead and drops the pose.
#[test]
fn a_lying_dictionary_claiming_fewer_than_it_enumerates_is_refused() {
  let lying = LyingJointDictionary::with_lied_count(1);
  let as_dict: &NSDictionary<NSString, NSString> = &lying;
  assert_eq!(as_dict.len(), 1, "the object really does under-report");
  assert_eq!(
    outcome_name(&read_pose_joints(
      as_dict,
      MAX_POSE_JOINTS,
      &mut PoseBudget::new()
    )),
    "Malformed",
    "an under-reporting dictionary must fail closed at the real entry point"
  );
}

/// A refused read has still done the walking, and now pays for it.
///
/// This is the accounting the finding was about. The old reader
/// allocated, enumerated and keyed-looked-up its way through a
/// dictionary and, on any of the four refusals, returned `None` without
/// moving a counter; the caller `continue`d to the next observation with
/// the attempt ceiling exactly where it started. The charge now sits
/// inside the walk, so each of these reads leaves the budget short by
/// precisely the number of entries it reached.
///
/// Each direction walks a different distance and the budget shows it.
/// Over-reporting is only discovered when the enumeration runs dry, so
/// all three entries are walked; under-reporting is discovered on the
/// entry after the claimed count is already full, so two are. The
/// conforming read is pinned alongside them at three — the number the
/// removed bulk `charge_attempts(pairs.len())` charged — because the
/// fusion must not have become a new tax on the productive path.
#[test]
fn a_refused_read_pays_for_every_entry_it_walked() {
  let honest = three_joint_dictionary();
  let mut on_honest = PoseBudget::new();
  assert_eq!(
    outcome_name(&read_pose_joints(&honest, MAX_POSE_JOINTS, &mut on_honest)),
    "Read"
  );
  assert_eq!(
    drain_visits(&mut on_honest),
    MAX_POSE_JOINT_ATTEMPTS_PER_CALL - 3,
    "a conforming read costs exactly the entries it enumerates — no more than the bulk charge it \
     replaced"
  );

  let over = LyingJointDictionary::with_lied_count(5);
  let over_dict: &NSDictionary<NSString, NSString> = &over;
  let mut on_over = PoseBudget::new();
  assert_eq!(
    outcome_name(&read_pose_joints(over_dict, MAX_POSE_JOINTS, &mut on_over)),
    "Malformed",
    "an over-reporting dictionary is still refused"
  );
  assert_eq!(
    drain_visits(&mut on_over),
    MAX_POSE_JOINT_ATTEMPTS_PER_CALL - 3,
    "and it paid for all three entries it walked before the lie surfaced"
  );

  let under = LyingJointDictionary::with_lied_count(1);
  let under_dict: &NSDictionary<NSString, NSString> = &under;
  let mut on_under = PoseBudget::new();
  assert_eq!(
    outcome_name(&read_pose_joints(
      under_dict,
      MAX_POSE_JOINTS,
      &mut on_under
    )),
    "Malformed",
    "an under-reporting dictionary is still refused"
  );
  assert_eq!(
    drain_visits(&mut on_under),
    MAX_POSE_JOINT_ATTEMPTS_PER_CALL - 2,
    "and it paid for the two it walked — what the walk reached, not what the dictionary held"
  );

  // The one refusal that must stay free: `reported > max_entries` is
  // decided before the enumeration begins, so no entry is walked and
  // charging for one would charge for work never done.
  let mut on_over_cap = PoseBudget::new();
  assert_eq!(
    outcome_name(&collect_dictionary_pairs(
      &honest,
      MAX_POSE_JOINTS + 1,
      MAX_POSE_JOINTS,
      &mut on_over_cap
    )),
    "Malformed"
  );
  assert_eq!(
    drain_visits(&mut on_over_cap),
    MAX_POSE_JOINT_ATTEMPTS_PER_CALL,
    "a read refused before it enumerates anything charges nothing"
  );
}

/// The finding itself, pinned in arithmetic: no number of malformed
/// dictionaries can walk a call past its attempt ceiling.
///
/// The fixture is the shape the review named — a dictionary at the cap
/// that enumerates every one of its entries and only then turns out to
/// disagree with its own `count`. Under the old accounting each of those
/// observations was allocated for, enumerated and keyed-looked-up entry
/// by entry, and the refusal that followed cost nothing; at
/// `MAX_VISION_RESULTS_PER_FRAME` observations that is over a million
/// entry walks against an 8192-attempt budget that never moved.
///
/// Driven against ONE budget, as the four extractors drive it, the same
/// frame now stops dead: each read pays for the whole dictionary it
/// walked, the ceiling buys exactly `MAX_POSE_JOINT_ATTEMPTS_PER_CALL /
/// MAX_POSE_JOINTS` of them, and the next read comes back `Exhausted`
/// before it walks a single entry — which is the extractors' signal to
/// stop the frame, not to skip an observation.
///
/// A regression that drops the charge fails this test rather than
/// hanging: the loop's own hard stop is the frame's observation cap.
#[test]
fn malformed_dictionaries_cannot_walk_past_the_call_attempt_ceiling() {
  assert_eq!(
    MAX_POSE_JOINT_ATTEMPTS_PER_CALL % MAX_POSE_JOINTS,
    0,
    "this test reads the total walked off the read count, which needs the ceiling to be a whole \
     number of full-cap dictionaries"
  );
  let unbudgeted = MAX_VISION_RESULTS_PER_FRAME * MAX_POSE_JOINTS;
  assert!(
    unbudgeted > MAX_POSE_JOINT_ATTEMPTS_PER_CALL * 100,
    "the walk this test bounds must be able to overrun the ceiling by orders of magnitude: \
     {unbudgeted} entry walks against a {MAX_POSE_JOINT_ATTEMPTS_PER_CALL} ceiling"
  );

  let lying = LyingJointDictionary::over(
    joint_dictionary_of(MAX_POSE_JOINTS),
    // One below what it holds, so every entry is walked and the
    // over-enumeration branch is what finally refuses the read.
    MAX_POSE_JOINTS - 1,
  );
  let as_dict: &NSDictionary<NSString, NSString> = &lying;
  assert_eq!(
    as_dict.len(),
    MAX_POSE_JOINTS - 1,
    "the object under-reports by one"
  );
  assert_eq!(
    as_dict.keys().count(),
    MAX_POSE_JOINTS,
    "while really holding a full-cap dictionary to walk"
  );

  let mut budget = PoseBudget::new();
  let mut malformed = 0usize;
  let mut exhausted_at = None;
  for observation in 0..MAX_VISION_RESULTS_PER_FRAME {
    match read_pose_joints(as_dict, MAX_POSE_JOINTS, &mut budget) {
      PoseJoints::Read(_) => {
        panic!("observation {observation}: a dictionary that lies about its count must not read")
      }
      PoseJoints::Malformed => malformed += 1,
      PoseJoints::Exhausted => {
        exhausted_at = Some(observation);
        break;
      }
    }
  }

  let exhausted_at =
    exhausted_at.expect("the walk must run out of attempt budget, not run to the end of the frame");
  assert_eq!(
    malformed,
    MAX_POSE_JOINT_ATTEMPTS_PER_CALL / MAX_POSE_JOINTS,
    "each malformed read pays for the whole dictionary it walked, so the ceiling buys exactly \
     this many of them"
  );
  assert_eq!(
    malformed * MAX_POSE_JOINTS,
    MAX_POSE_JOINT_ATTEMPTS_PER_CALL,
    "and the total walked across every observation is the call-wide ceiling, not the \
     per-observation cap times the frame"
  );
  assert_eq!(
    exhausted_at, malformed,
    "the read after the last affordable one is the one refused"
  );
  assert_eq!(
    drain_visits(&mut budget),
    0,
    "an `Exhausted` read leaves nothing behind — it is a spent budget, not a skipped observation"
  );
}

/// A real frame's poses cost the budget almost nothing: twenty joints
/// with twenty-byte Apple identifiers, over and over, never approaches
/// any of the three ceilings.
#[test]
fn pose_budget_admits_ordinary_poses_in_succession() {
  let mut budget = PoseBudget::new();
  for pose in 0..64 {
    for joint in 0..20 {
      assert!(
        budget.charge_joint_visit(),
        "pose {pose}, joint {joint}: an ordinary pose's walk is nowhere near the attempt ceiling"
      );
    }
    assert!(
      budget.admit_pose(20, 400),
      "pose {pose}: an ordinary pose is nowhere near the joint or name-byte ceilings"
    );
  }
}

/// An exhausted attempt budget stays exhausted, and every visit up to
/// that point was admitted.
///
/// The unit charge makes both halves the same property: a refusal that
/// charged anyway — or a wrapping subtraction, which would have handed
/// the walk a near-`usize::MAX` budget on its first refusal — shows up
/// here as a `true` where a `false` belongs. The four extractors treat
/// that `false` as "stop the frame", so it has to be permanent.
#[test]
fn pose_budget_refuses_every_joint_visit_past_the_ceiling_without_wrapping() {
  let mut budget = PoseBudget::new();
  assert_eq!(
    drain_visits(&mut budget),
    MAX_POSE_JOINT_ATTEMPTS_PER_CALL,
    "every visit inside the ceiling is admitted, and the ceiling is where they stop"
  );
  for refusal in 0..3 {
    assert!(
      !budget.charge_joint_visit(),
      "refusal {refusal}: a spent attempt budget never comes back"
    );
  }
}

/// The joint ceiling refuses a pose it cannot cover whole, and charges
/// NEITHER counter on the way out.
///
/// Both halves are pinned to the unit. The joint half: the ten joints
/// left after the refusal are still spendable. The name half: exactly
/// two poses were admitted at a hundred name bytes each, so asking for
/// the remaining `MAX - 200` to the byte fails the moment a
/// joint-ceiling refusal also spends the name budget.
#[test]
fn pose_budget_refuses_a_pose_over_the_joint_ceiling_and_charges_nothing() {
  let mut budget = PoseBudget::new();
  assert!(budget.admit_pose(MAX_POSE_JOINTS_PER_CALL - 10, 100));
  assert!(
    !budget.admit_pose(11, 100),
    "eleven joints do not fit in the ten that remain"
  );
  assert!(
    budget.admit_pose(10, 100),
    "the refusal charged nothing — the ten joints that remained are still there"
  );
  assert!(
    budget.admit_pose(0, MAX_POSE_JOINT_NAME_BYTES_PER_CALL - 200),
    "a joint-ceiling refusal must leave the NAME budget untouched too"
  );
}

/// The name-byte ceiling refuses on its own terms — the joints fit and
/// the pose is dropped anyway — and charges NEITHER counter on the way
/// out.
///
/// The cross-counter half is the one worth pinning to the unit: exactly
/// two poses were admitted at one joint each, so asking for the
/// remaining `MAX - 2` fails the moment a name-byte refusal charges the
/// joint it was carrying before turning the pose away.
#[test]
fn pose_budget_refuses_a_pose_over_the_name_byte_ceiling_and_charges_nothing() {
  let mut budget = PoseBudget::new();
  assert!(budget.admit_pose(1, MAX_POSE_JOINT_NAME_BYTES_PER_CALL - 10));
  assert!(
    !budget.admit_pose(1, 11),
    "one joint fits; its eleven name bytes do not, and that alone refuses the pose"
  );
  assert!(
    budget.admit_pose(1, 10),
    "the refusal charged nothing — the ten name bytes that remained are still there"
  );
  assert!(
    budget.admit_pose(MAX_POSE_JOINTS_PER_CALL - 2, 0),
    "a name-byte refusal must leave the JOINT budget untouched too"
  );
}

/// Each ceiling is INCLUSIVE: a pose that exactly exhausts a budget is
/// admitted, and only the next one is refused.
#[test]
fn pose_budget_admits_an_exact_fit_at_every_ceiling() {
  let mut attempts = PoseBudget::new();
  assert_eq!(
    drain_visits(&mut attempts),
    MAX_POSE_JOINT_ATTEMPTS_PER_CALL,
    "the attempt ceiling is a bound, not an off-by-one"
  );
  assert!(!attempts.charge_joint_visit());

  let mut joints = PoseBudget::new();
  assert!(
    joints.admit_pose(MAX_POSE_JOINTS_PER_CALL, 0),
    "the joint ceiling is a bound, not an off-by-one"
  );
  assert!(!joints.admit_pose(1, 0));

  let mut names = PoseBudget::new();
  assert!(
    names.admit_pose(0, MAX_POSE_JOINT_NAME_BYTES_PER_CALL),
    "the name-byte ceiling is a bound, not an off-by-one"
  );
  assert!(!names.admit_pose(0, 1));
}

/// `usize::MAX` is what a corrupted joint count would look like arriving
/// at [`PoseBudget::admit_pose`] — the one method that still takes a
/// count from the walk. It refuses every combination, and — because the
/// subtractions are checked rather than wrapping or saturating — the
/// budget afterwards is still a whole one: a wrapping subtraction would
/// have left a near-`usize::MAX` remainder behind, a saturating one
/// zero. In a debug build an unchecked subtraction would have panicked
/// here instead.
#[test]
fn pose_budget_refuses_usize_max_without_panicking_or_wrapping() {
  let mut budget = PoseBudget::new();
  assert!(!budget.admit_pose(usize::MAX, 0));
  assert!(!budget.admit_pose(0, usize::MAX));
  assert!(!budget.admit_pose(usize::MAX, usize::MAX));

  assert!(
    budget.admit_pose(MAX_POSE_JOINTS_PER_CALL, MAX_POSE_JOINT_NAME_BYTES_PER_CALL),
    "the joint and name-byte budgets are untouched by three refusals"
  );
  assert_eq!(
    drain_visits(&mut budget),
    MAX_POSE_JOINT_ATTEMPTS_PER_CALL,
    "and the attempt budget — a separate counter, spent one visit at a time — is untouched by \
     them too"
  );
}

/// The finding this budget closes, pinned in arithmetic.
///
/// The outer `.take(MAX_VISION_RESULTS_PER_FRAME)` and the
/// per-observation [`MAX_POSE_JOINTS`] dictionary cap are each
/// respected by an adversarial result that still composes them into
/// 1,048,576 joints — every one of which may carry a
/// `MAX_FFI_STRING_BYTES` name. Per-observation caps bound the FACTORS;
/// only a cumulative per-call budget bounds the PRODUCT, and this
/// asserts it does: driving a fresh budget with back-to-back
/// maximum-size poses admits `MAX_POSE_JOINTS_PER_CALL /
/// MAX_POSE_JOINTS` of them and then stops, a 256-fold cut on what the
/// factors alone allow.
#[test]
fn pose_budget_bounds_the_product_that_the_per_observation_caps_leave_open() {
  let unbudgeted = MAX_VISION_RESULTS_PER_FRAME * MAX_POSE_JOINTS;
  assert_eq!(unbudgeted, 1_048_576);
  assert!(
    unbudgeted >= MAX_POSE_JOINTS_PER_CALL * 100,
    "the per-observation caps alone leave the per-call joint count two orders of magnitude above \
     the cumulative ceiling: {unbudgeted} against {MAX_POSE_JOINTS_PER_CALL}"
  );

  let mut budget = PoseBudget::new();
  let mut admitted = 0usize;
  'poses: for _ in 0..MAX_VISION_RESULTS_PER_FRAME {
    for _ in 0..MAX_POSE_JOINTS {
      if !budget.charge_joint_visit() {
        break 'poses;
      }
    }
    if !budget.admit_pose(MAX_POSE_JOINTS, 0) {
      break;
    }
    admitted += 1;
  }
  assert_eq!(
    admitted,
    MAX_POSE_JOINTS_PER_CALL / MAX_POSE_JOINTS,
    "one call admits only as many maximum-size poses as the cumulative joint ceiling pays for"
  );
  assert!(
    admitted * MAX_POSE_JOINTS <= MAX_POSE_JOINTS_PER_CALL,
    "and what it admitted is inside that ceiling, not merely inside the per-observation one"
  );
}

/// `guard_vision_ffi` passes a non-raising closure's value through
/// untouched (the common, no-exception path).
#[test]
fn guard_vision_ffi_returns_closure_value_when_no_exception() {
  let got = guard_vision_ffi("test_detector", Vec::<u8>::new(), || vec![1u8, 2, 3]);
  assert_eq!(got, vec![1u8, 2, 3]);
}

/// The core of the process-abort fix: a real Objective-C `NSException`
/// raised inside the guarded closure is caught and converted to the
/// `fallback`, NOT propagated. `std::panic::catch_unwind` cannot do
/// this — a foreign exception escaping it aborts the process with
/// `fatal runtime error: Rust cannot catch foreign exceptions`.
///
/// `-[NSArray objectAtIndex:]` on an empty array raises
/// `NSRangeException` — a genuine foreign exception via a *valid*
/// selector, so objc2's debug-build msg_send verification passes and
/// the runtime raises for real in BOTH debug and release builds
/// (unlike the encoding-mismatched `VNHumanBodyRecognizedPoint3D`
/// selector, which only raises in release). If `guard_vision_ffi`
/// did not wrap the call in `objc2::exception::catch`, this test
/// would abort the whole test binary instead of returning.
#[test]
fn guard_vision_ffi_catches_objc_exception_and_returns_fallback() {
  let empty: Retained<NSArray<objc2::runtime::NSObject>> = NSArray::new();
  let got = guard_vision_ffi("test_detector", 7u32, || {
    // Out-of-bounds access raises NSRangeException across the FFI.
    let _ = empty.objectAtIndex(0);
    0u32
  });
  assert_eq!(
    got, 7u32,
    "guard must return the fallback after catching the NSException"
  );
}

// This `#[link]` is the ONLY thing that names the test archive, and it
// exists only in a `cfg(test)` build — `build.rs` compiles that archive
// with `cargo_metadata(false)`, so no `rustc-link-lib` reaches a
// consumer. That separation is load-bearing rather than tidiness: a
// test-only Objective-C class sharing the production archive would be
// pulled into any consumer linking with `-ObjC`, `-all_load` or
// `-force_load`, which load every member defining a class whether or
// not anything references it (Apple QA1490). An archive nothing links
// cannot be force-loaded.
//
// `C`, not `C-unwind`: this one allocates and initialises, and neither
// can raise. The send that can is the shim's, next door.
#[link(name = "avanalyze_0_6_objc_simd_shim_test", kind = "static")]
unsafe extern "C" {
  /// `src/objc_simd_shim_test.m` — a class that answers `position` with
  /// a matrix fixed in its own source, at +1 for the caller to own.
  ///
  /// The `0_6` matches the shim's own version scope; see the extern
  /// block in `src/ffi.rs` for why the names carry one at all.
  fn avanalyze_0_6_test_point3d_new() -> *mut AnyObject;
}

/// The shim's ABI, proved against a matrix this crate wrote down.
///
/// This is the deterministic half of the 3-D read's coverage, and the
/// half that belongs in a unit test. The end-to-end assertion in
/// `src/tests/body_pose.rs` runs a photograph through a neural network:
/// it proves the whole road, but its numbers are inference, pinned to
/// this host, this macOS and this backend, so it cannot be asked for
/// exact floats. This test asks for exact floats, because the object on
/// the other side of the boundary is sixteen literals in
/// `src/objc_simd_shim_test.m` and nothing about it can drift.
///
/// The matrix is `columns[col][row] = col * 4 + row + 0.5`, which in
/// Apple's column-major memory order is `0.5, 1.5, ... 15.5`. Every
/// entry differs from every other and from its transpose partner, so
/// each way this read has failed or could fail is caught by value:
///
/// - the shipped ABI defect returned a stack buffer the callee never
///   wrote, so the floats were unrelated garbage near `1e26`;
/// - a float HFA would reach only the low half of each register, taking
///   the first two elements of each column and dropping the rest;
/// - a transposed read would return `0.5, 4.5, 8.5, 12.5, ...`;
/// - a short read would leave the tail at its initial zero.
///
/// The matrix is deliberately not affine — its bottom row is
/// `(3.5, 7.5, 11.5, 15.5)`. This test is about sixteen floats crossing
/// a calling convention intact; deciding which of them is a translation
/// is `translation_if_affine`, which is pure Rust and is tested on its
/// own synthetic matrices.
#[test]
fn the_simd_shim_reads_a_known_matrix_exactly() {
  let point = unsafe { Retained::from_raw(avanalyze_0_6_test_point3d_new()) }
    .expect("the test object's initializer must not return nil");

  let got = unsafe { vn_point3d_position(&*point) };

  let want: [f32; 16] = core::array::from_fn(|i| i as f32 + 0.5);
  assert_eq!(
    got, want,
    "the sixteen floats of a simd_float4x4 must cross the shim unchanged and in memory order"
  );
}

/// The same contract, through the C shim — the one frame in this crate
/// an exception crosses that Rust did not compile.
///
/// `vn_point3d_position` dispatches through `src/objc_simd_shim.m`, so
/// a raise unwinds a frame Clang compiled before it reaches the
/// `objc2::exception::catch` that catches it. That works only if both
/// halves of the boundary admit unwinding: the Rust declaration is
/// `extern "C-unwind"`, and `build.rs` compiles the shim with exception
/// support. Get either wrong and the exception meets a frame declared
/// not to unwind, which is undefined behaviour rather than a catch.
///
/// An `NSString` does not respond to `position`, so this raises
/// `doesNotRecognizeSelector:` — the very exception the 3-D pose path
/// used to raise for real, from its own missing selector. A regression
/// here aborts the test binary instead of failing it, which is the
/// honest signal for this class of defect.
#[test]
fn a_raise_unwinds_out_of_the_simd_shim_and_is_caught() {
  let not_a_point = NSString::from_str("not a VNPoint3D");
  let got = guard_vision_ffi("test_detector", [f32::NAN; 16], || unsafe {
    vn_point3d_position(&*not_a_point)
  });
  assert!(
    got[0].is_nan(),
    "the guard must return the fallback after catching the raise out of the shim"
  );
}

/// The same raise, under the barrier order `extract_3d` actually uses.
///
/// The test above guards the shim with `guard_vision_ffi` alone, which
/// is not the shape the 3-D path runs: there, `catch_unwind` wraps the
/// Objective-C barrier, and that ordering is the whole point. An
/// Objective-C exception reaching the outer `catch_unwind` would abort
/// the process — `fatal runtime error: Rust cannot catch foreign
/// exceptions` — rather than fail a test, so the flat version above
/// cannot prove the production shape.
///
/// This pins it: the raise must be consumed by the INNER barrier, and
/// the outer `catch_unwind` must come back `Ok`. `Err` would mean the
/// exception had been turned into a Rust panic somewhere it should not
/// have been; an abort would mean it crossed the Objective-C barrier
/// untouched.
#[test]
fn a_raise_on_the_joint_read_path_never_reaches_the_outer_catch_unwind() {
  use std::panic::{AssertUnwindSafe, catch_unwind};

  let not_a_point = NSString::from_str("not a VNPoint3D");
  let outcome = catch_unwind(AssertUnwindSafe(|| {
    guard_vision_ffi("body_pose_3d", [f32::NAN; 16], || unsafe {
      vn_point3d_position(&*not_a_point)
    })
  }));

  let got = outcome.expect("the Objective-C barrier must consume the raise, not the Rust one");
  assert!(
    got[0].is_nan(),
    "the inner guard must return the fallback after catching the raise"
  );
}

/// The same production nesting, against the exception class the
/// Objective-C barrier does NOT catch.
///
/// The test above pins the order for an Objective-C raise, which
/// `objc2::exception::catch` consumes. A C++ throw walks straight
/// through that `@catch (id)` — deliberately: it matches Objective-C
/// objects and lets everything else keep unwinding — and until this
/// crate grew a C++-aware barrier there was nothing further out to stop
/// it. It reached `extract_3d`'s `catch_unwind` and aborted the
/// process.
///
/// So this asserts the same shape one layer deeper: the throw must be
/// consumed by the NATIVE barrier that now sits outside the
/// Objective-C one, and the outer `catch_unwind` must still come back
/// `Ok`. An abort here means the C++ exception crossed both barriers
/// untouched, which is the defect itself.
#[test]
fn a_cxx_throw_on_the_joint_read_path_never_reaches_the_outer_catch_unwind() {
  use std::panic::{AssertUnwindSafe, catch_unwind};

  use crate::tests::native_barrier::{SyntheticThrow, synthetic_throw};

  let outcome = catch_unwind(AssertUnwindSafe(|| {
    guard_vision_ffi("body_pose_3d", [f32::NAN; 16], || {
      // SAFETY: inside `guard_vision_ffi`, which is how the native
      // barrier is reached.
      unsafe { synthetic_throw(SyntheticThrow::StdException) };
      [0f32; 16]
    })
  }));

  let got = outcome.expect("the native barrier must consume the throw, not the Rust one");
  assert!(
    got[0].is_nan(),
    "the guard must return the fallback after catching the C++ throw"
  );
}

/// A perform has exactly two outcomes and they must never collapse
/// into one another. `Completed` licenses a caller to read the
/// requests' `results`; `Raised` forbids it, because a retained
/// request that Vision did not process on this call may still hold the
/// PREVIOUS call's observations. Reporting a caught exception as a
/// bare `Ok` — which is what this type replaced — made the two
/// indistinguishable at every call site.
///
/// `Copy` is load-bearing rather than cosmetic: `FaceDetector::detect`
/// holds the stage-two outcome across the input clear it must run
/// before propagating, so the outcome cannot be a value that moving
/// consumes.
#[test]
fn performed_never_confuses_a_caught_perform_with_a_completed_one() {
  assert_ne!(Performed::Completed, Performed::Raised);
  assert_eq!(Performed::Completed, Performed::Completed);
  assert_eq!(Performed::Raised, Performed::Raised);

  let outcome = Performed::Raised;
  let held = outcome;
  assert_eq!(held, outcome, "the outcome survives being held and reread");
}

/// The `Completed` half of `run_requests`: a perform that really ran
/// calls `extract` and returns ITS value, leaving `fallback` untouched.
///
/// The `Raised` half — `extract` not called at all, `fallback`
/// returned — is not covered here and cannot honestly be: reaching it
/// requires Apple's Vision to raise a genuine Objective-C
/// `NSException` from inside `performRequests:onImageData:error:`, and
/// every misuse this crate can construct without a mock Vision
/// framework (an abstract `VNRequest`, an undecodable payload) is
/// reported as an `NSError` instead, which is the `Err` path. Faking a
/// raise would test the fake. The property is instead carried by the
/// type: `extract` is reachable from exactly one match arm.
#[test]
fn run_requests_extracts_only_after_a_completed_perform() {
  use std::cell::Cell;

  use objc2_vision::{VNDetectFaceRectanglesRequest, VNRequest};

  const JPEG: &[u8] = include_bytes!("../../tests/fixtures/airport_keyframe.jpg");

  let requests = unsafe {
    [Retained::cast_unchecked::<VNRequest>(
      VNDetectFaceRectanglesRequest::new(),
    )]
  };
  let extracted = Cell::new(false);
  let got = run_requests(ImageSource::Jpeg(JPEG), &requests, vec![9u8], || {
    extracted.set(true);
    vec![1u8, 2, 3]
  })
  .expect("a real request performed on a real image completes");

  assert!(
    extracted.get(),
    "a completed perform is the one condition under which request state may be read"
  );
  assert_eq!(
    got,
    vec![1u8, 2, 3],
    "the completed path returns the extraction, never the fallback"
  );
}

/// Every entry point is documented as "not safe to share across
/// threads; build one per worker". That is not a convention — the
/// retained `VNRequest` handles make each holder `!Sync`, so the
/// compiler already refuses `Arc<TextRecognizer>` and every sibling.
/// Nothing pinned it, though, and a future objc2-vision that marks the
/// request classes thread-safe would let the holders become `Sync`
/// silently, turning a compile error into a data race across two
/// workers performing on one stateful request object. This fails to
/// compile the moment that happens.
///
/// The two impls are the ambiguity trick: when the self type is `Sync`
/// both apply and `Marker` cannot be inferred, so the call site stops
/// compiling. `Marker` MUST stay a parameter of `assert_not_sync`
/// rather than being inferred inside its body — the ambiguity only
/// arises where the self type is concrete. Given
/// `fn assert_not_sync<T: ?Sized>() { <T as NotSyncProof<_>>::proof() }`
/// the second impl's `T: Sync` obligation is unprovable for the
/// unbounded parameter, so it is winnowed out, `Marker` resolves to
/// `()` for every `T`, and the assertion silently proves nothing.
trait NotSyncProof<Marker> {
  fn proof() {}
}
impl<T: ?Sized> NotSyncProof<()> for T {}
impl<T: ?Sized + Sync> NotSyncProof<u8> for T {}

fn assert_not_sync<Marker, T: ?Sized + NotSyncProof<Marker>>() {
  <T as NotSyncProof<Marker>>::proof();
}

#[test]
fn entry_points_are_not_shareable_across_workers() {
  assert_not_sync::<_, crate::VisionAnalyzer>();
  assert_not_sync::<_, crate::TextRecognizer>();
  assert_not_sync::<_, crate::BarcodeDetector>();
  assert_not_sync::<_, crate::FaceDetector>();
  assert_not_sync::<_, crate::FaceLandmarker>();
  assert_not_sync::<_, crate::BodyPoser>();
  assert_not_sync::<_, crate::HandPoser>();
  assert_not_sync::<_, crate::AnimalPoser>();
  assert_not_sync::<_, crate::PersonMasker>();
}

// ----- the decoded-dimension SOF preflight (issue #2) ------------------------

/// Builds the smallest well-formed JPEG prefix [`check_decoded_dimensions`]
/// needs: SOI followed by one SOF0 segment declaring `width` × `height` at
/// 8-bit precision, with a single component. No entropy-coded data
/// follows — the walk returns as soon as it reads the SOF, so none is
/// needed.
fn crafted_sof0(width: u16, height: u16) -> Vec<u8> {
  crafted_sof0_with_precision(width, height, 0x08)
}

/// As [`crafted_sof0`], with an explicit sample precision byte.
fn crafted_sof0_with_precision(width: u16, height: u16, precision: u8) -> Vec<u8> {
  let [h0, h1] = height.to_be_bytes();
  let [w0, w1] = width.to_be_bytes();
  vec![
    0xFF, 0xD8, // SOI
    0xFF, 0xC0, // SOF0
    0x00,
    0x0B, // length = 11: itself(2) + precision(1) + height(2) + width(2) + Nf(1) + one component(3)
    precision, h0, h1, // height, big-endian
    w0, w1,   // width, big-endian
    0x01, // Nf = 1
    0x01, 0x11, 0x00, // component: id=1, sampling=0x11, quant table=0
  ]
}

/// Ticket test 1/5: a truncated or empty input fails cleanly — refused,
/// not panicked on — before any Vision call is even attempted.
#[test]
fn sof_preflight_rejects_truncated_or_empty_input() {
  let cases: [&[u8]; 4] = [&[], &[0xFF, 0xD8], &[0xFF], &[0x00, 0x01, 0x02]];
  for jpeg in cases {
    let err = check_decoded_dimensions(jpeg).expect_err("truncated/empty input must be refused");
    assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
  }
}

/// Ticket test 2/5: a valid JPEG under the cap passes.
#[test]
fn sof_preflight_passes_a_real_keyframe_under_the_cap() {
  const JPEG: &[u8] = include_bytes!("../../tests/fixtures/airport_keyframe.jpg");
  check_decoded_dimensions(JPEG).expect("a real keyframe under the cap must pass preflight");
}

/// Ticket test 3/5: a valid JPEG declaring dimensions over the cap is
/// rejected before `NSData::with_bytes` — proven here by driving the
/// refusal through [`with_image`] itself, the one function that calls it,
/// and asserting its body closure never runs.
#[test]
fn sof_preflight_rejects_over_cap_dimensions_before_ns_data() {
  use std::cell::Cell;

  // 65535 × 65535 × 4 bytes/pixel ≈ 16 GiB, far past the 512 MiB cap —
  // the issue's own credible hostile case.
  let jpeg = crafted_sof0(u16::MAX, u16::MAX);
  let body_ran = Cell::new(false);

  let err = with_image(ImageSource::Jpeg(&jpeg), |_handler, _image| {
    body_ran.set(true);
    Ok(())
  })
  .expect_err("over-cap decoded dimensions must be refused");

  assert!(
    !body_ran.get(),
    "the preflight must refuse before NSData::with_bytes hands data to the request body"
  );
  assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
  assert!(err.message().contains("MAX_DECODED_IMAGE_BYTES"));
}

/// Ticket test 4/5: a malformed JPEG with no SOF marker at all — SOI, one
/// harmless APP0 segment, then straight to EOI — is rejected.
#[test]
fn sof_preflight_rejects_input_with_no_sof_marker() {
  #[rustfmt::skip]
  let jpeg: [u8; 10] = [
    0xFF, 0xD8,                          // SOI
    0xFF, 0xE0, 0x00, 0x04, 0x4A, 0x46,  // APP0, length 4, 2-byte payload
    0xFF, 0xD9,                          // EOI -- no SOF ever appeared
  ];
  let err = check_decoded_dimensions(&jpeg).expect_err("input with no SOF marker must be refused");
  assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
  assert!(err.message().contains("no valid JPEG SOF marker"));
}

/// Ticket test 5/5: a forged SOF length field is rejected without a panic
/// or an out-of-bounds read, in both lying directions — a length that
/// reaches past the end of the buffer, and one shorter than the fixed
/// header it claims to hold. `catch_unwind` makes "never panics" an
/// explicit assertion here rather than an implicit one.
#[test]
fn sof_preflight_rejects_forged_sof_length_without_panic_or_oob() {
  use std::panic::{AssertUnwindSafe, catch_unwind};

  #[rustfmt::skip]
  let length_past_buffer_end: [u8; 12] = [
    0xFF, 0xD8,             // SOI
    0xFF, 0xC0, 0xFF, 0xFF, // SOF0, length = 0xFFFF -- nowhere near the 6 bytes actually here
    0x08, 0x00, 0x10, 0x00, 0x10, 0x01,
  ];
  #[rustfmt::skip]
  let length_shorter_than_header: [u8; 8] = [
    0xFF, 0xD8,             // SOI
    0xFF, 0xC0, 0x00, 0x04, // SOF0, length = 4 -- covers only itself + 2 bytes, not the 8-byte header
    0x08, 0x00,
  ];

  for jpeg in [&length_past_buffer_end[..], &length_shorter_than_header[..]] {
    let result = catch_unwind(AssertUnwindSafe(|| check_decoded_dimensions(jpeg)));
    let err = result
      .expect("a forged SOF length must never panic or read out of bounds")
      .expect_err("a forged SOF length must be refused");
    assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
    assert!(err.message().contains("malformed JPEG marker length"));
  }
}

/// Beyond the five: the marker walk must terminate on a pathological input
/// that never presents a real marker code — an SOI followed entirely by
/// `0xFF` fill bytes — rather than loop. If it looped, this test would
/// hang instead of failing an assertion.
#[test]
fn sof_preflight_terminates_on_a_long_run_of_fill_bytes() {
  let mut jpeg = vec![0xFFu8; 4098];
  jpeg[1] = 0xD8; // SOI; every other byte stays a 0xFF fill byte

  let err = check_decoded_dimensions(&jpeg)
    .expect_err("a buffer with no real marker code must be refused, not hang");
  assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
}

/// Beyond the five: the cap compares strictly-greater-than, matching the
/// issue's own wording ("above `MAX_DECODED_IMAGE_BYTES`") — dimensions
/// landing exactly on the cap pass, and one pixel row more is refused.
#[test]
fn sof_preflight_boundary_is_at_and_just_over_the_cap() {
  assert_eq!(MAX_DECODED_IMAGE_BYTES, 512 * 1024 * 1024);

  // 16384 × 8192 × 4 bytes/pixel = exactly 512 MiB.
  let at_cap = crafted_sof0(16384, 8192);
  check_decoded_dimensions(&at_cap).expect("dimensions landing exactly on the cap must pass");

  // One more pixel row pushes the decoded size one row past the cap.
  let over_cap = crafted_sof0(16384, 8193);
  let err =
    check_decoded_dimensions(&over_cap).expect_err("dimensions just over the cap must be refused");
  assert!(err.message().contains("MAX_DECODED_IMAGE_BYTES"));
}

/// Codex R1 finding (high): a SOF declaring sample precision above 8 bits
/// decodes through ImageIO at double the baseline byte rate (16-bit/component
/// RGBA, confirmed against real ImageIO) — 16384 × 8192 lands exactly on the
/// cap at the 8-bit rate but must be refused at 12-bit precision, where the
/// real decode is twice the declared budget.
#[test]
fn sof_preflight_charges_the_wider_rate_for_high_precision_frames() {
  let baseline = crafted_sof0_with_precision(16384, 8192, 8);
  check_decoded_dimensions(&baseline)
    .expect("8-bit precision at exactly the cap still passes at the 4 bytes/pixel rate");

  for precision in [9u8, 12, 16, 255] {
    let high_precision = crafted_sof0_with_precision(16384, 8192, precision);
    let err = check_decoded_dimensions(&high_precision).unwrap_err();
    assert!(
      err.message().contains("MAX_DECODED_IMAGE_BYTES"),
      "precision {precision} must be charged at the wider byte rate and refused at this cap-boundary size"
    );
  }
}

/// Codex R1 finding (medium): JPEG permits a baseline SOF to declare
/// `height = 0` and defer the real value to a DNL marker after the first
/// scan (ITU-T T.81 §B.2.5). This preflight never reads that far, so a
/// deferred height must be refused outright rather than silently treated
/// as zero decoded bytes — the DNL-deferral bypass this closes: a hostile
/// SOF claiming height zero while the real, DNL-supplied height is huge.
#[test]
fn sof_preflight_rejects_a_deferred_zero_height() {
  let deferred_height = crafted_sof0(16384, 0);
  let err = check_decoded_dimensions(&deferred_height)
    .expect_err("a zero-height SOF (deferred to a later DNL marker) must be refused");
  assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);

  // Width has no analogous deferral mechanism in the JPEG spec, but zero
  // is refused for the same reason: a preflight that cannot establish a
  // dimension must refuse rather than compute a budget of zero from it.
  let zero_width = crafted_sof0(0, 16384);
  check_decoded_dimensions(&zero_width).expect_err("a zero-width SOF must be refused");
}

/// Codex R2 finding (medium): a DHP marker (ITU-T T.81 Annex J,
/// hierarchical JPEG) declares the *completed* image's dimensions in the
/// same precision/height/width/Nf shape as a SOF payload, but a
/// conforming hierarchical stream's first actual SOF can be small — a
/// walk that stops at the first SOF alone would budget the small frame
/// and miss the oversized completed image DHP declared. This SOF0(16, 16)
/// would pass trivially on its own, proving the refusal comes from
/// seeing DHP first, not from anything about the SOF that follows it.
#[test]
fn sof_preflight_rejects_hierarchical_jpeg_via_dhp() {
  #[rustfmt::skip]
  let mut jpeg: Vec<u8> = vec![
    0xFF, 0xD8,             // SOI
    0xFF, 0xDE, 0x00, 0x0B, // DHP, length 11 -- same shape as a SOF payload
    0x08,                   // precision
    0xFF, 0xFF,             // "completed" height = 65535
    0xFF, 0xFF,             // "completed" width  = 65535
    0x01,                   // Nf = 1
    0x01, 0x11, 0x00,       // one component
  ];
  // Everything after crafted_sof0's own SOI: a small, otherwise-valid SOF0.
  jpeg.extend_from_slice(&crafted_sof0(16, 16)[2..]);

  let err = check_decoded_dimensions(&jpeg)
    .expect_err("a DHP marker must be refused even though the small first SOF alone would pass");
  assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
  assert!(err.message().to_lowercase().contains("hierarchical"));
}
