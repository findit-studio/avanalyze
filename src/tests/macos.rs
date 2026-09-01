//! Engine internals that only exist on Apple targets.
//!
//! These pin the coordinate conversions, the resource guards, and the
//! two exception barriers. They run against the reference vocabulary
//! (`mediaschema`) so the assertions exercise a real validating
//! implementation rather than a permissive stub.

use mediaschema::domain::aggregates::video::{
  BoundingBox as DomainBoundingBox, FaceLandmarkRegion as DomainFaceLandmarkRegion,
};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};

use crate::*;

/// #48: IoU of a box with itself is 1.0 — exact-duplicate detections (the
/// common case across the two Vision face passes) match perfectly.
#[test]
fn bbox_iou_identical_is_one() {
  let a = DomainBoundingBox::try_new(0.1, 0.2, 0.3, 0.4).unwrap();
  assert!(
    (bbox_iou(&a, &a) - 1.0).abs() < 1e-6,
    "iou = {}",
    bbox_iou(&a, &a)
  );
}

/// #48: IoU of disjoint boxes is 0.0.
#[test]
fn bbox_iou_disjoint_is_zero() {
  let a = DomainBoundingBox::try_new(0.0, 0.0, 0.2, 0.2).unwrap();
  let b = DomainBoundingBox::try_new(0.5, 0.5, 0.2, 0.2).unwrap();
  assert_eq!(bbox_iou(&a, &b), 0.0);
}

/// #48: partial overlap. Two 0.2×0.2 boxes offset by 0.1 in x share a
/// 0.1×0.2 strip: inter = 0.02, union = 0.04 + 0.04 − 0.02 = 0.06 → IoU = 1/3.
#[test]
fn bbox_iou_partial_overlap() {
  let a = DomainBoundingBox::try_new(0.0, 0.0, 0.2, 0.2).unwrap();
  let b = DomainBoundingBox::try_new(0.1, 0.0, 0.2, 0.2).unwrap();
  assert!(
    (bbox_iou(&a, &b) - 1.0 / 3.0).abs() < 1e-5,
    "iou = {}",
    bbox_iou(&a, &b)
  );
}

/// #48: a detected face is annotated with the quality of the overlapping
/// capture-quality observation, and the highest-IoU match wins.
#[test]
fn matched_capture_quality_takes_overlapping_score() {
  let face = DomainBoundingBox::try_new(0.10, 0.10, 0.20, 0.20).unwrap();
  let scored = vec![
    (
      DomainBoundingBox::try_new(0.50, 0.50, 0.20, 0.20).unwrap(),
      0.2,
    ), // disjoint
    (
      DomainBoundingBox::try_new(0.10, 0.10, 0.20, 0.20).unwrap(),
      0.8,
    ), // exact overlap
  ];
  let got = matched_capture_quality(&face, &scored).expect("an overlapping observation must match");
  assert!((got - 0.8).abs() < 1e-6);
}

/// #20: a face the capture-quality pass did not cover annotates to `None`
/// — "unmatched" — never a real `Some(0.0)` measurement. (The
/// `min_capture_quality` threshold gate still fails closed on `None` at
/// the `extract_faces` call site: the default 0.1 drops it, while 0.0
/// keeps it — see the comment there. This function's own contract is
/// just the match outcome.)
#[test]
fn matched_capture_quality_none_without_overlap() {
  let face = DomainBoundingBox::try_new(0.10, 0.10, 0.20, 0.20).unwrap();
  let scored = vec![(
    DomainBoundingBox::try_new(0.60, 0.60, 0.20, 0.20).unwrap(),
    0.9,
  )];
  assert_eq!(matched_capture_quality(&face, &scored), None);
}

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
/// not propagate — the helper sanitises non-finite components to
/// `0.0`. A `NaN` `origin.x` collapses left and right to 0.0, so the
/// rectangle has zero width after clamping and is reported as
/// `None` (the detection is dropped).
#[test]
fn vision_bbox_handles_nan_origin() {
  let rect = CGRect::new(CGPoint::new(f64::NAN, 0.0), CGSize::new(0.3, 0.4));
  assert!(vision_rect_to_bbox::<DomainBoundingBox>(rect).is_none());
}

/// `NaN` in a single size component still produces a usable
/// rectangle iff the surviving edges enclose a non-zero area. A
/// finite `origin.x`/`width` keeps the horizontal extent live; a
/// `NaN` `origin.y` collapses the vertical extent to zero and the
/// rectangle is dropped.
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
/// detection. Previously the helper collapsed the bad component to
/// `0.0` via `clamp01`, which fabricated edge-aligned coordinates
/// that the domain validator could not distinguish from real
/// detections.
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

/// `normalized_bbox_from_pixel_bounds` must NOT flip the y axis —
/// `CVPixelBuffer` rows are top-to-bottom, so row 0 is the top edge
/// and the natural mapping `min_y / height` is already in top-left
/// convention.
#[test]
fn pixel_bounds_to_normalized_bbox_does_not_flip() {
  // A 100x100 mask with the foreground rectangle in rows 10..=29,
  // columns 5..=24. The expected normalized bbox is
  // `(5/100, 10/100, 20/100, 20/100)` in top-left convention.
  let bbox = normalized_bbox_from_pixel_bounds::<DomainBoundingBox>(5, 10, 24, 29, 100, 100)
    .expect("valid bbox");
  assert!((bbox.x() - 0.05).abs() < 1e-6);
  assert!((bbox.y() - 0.10).abs() < 1e-6);
  assert!((bbox.width() - 0.20).abs() < 1e-6);
  assert!((bbox.height() - 0.20).abs() < 1e-6);
}

/// An all-zero 8-bit mask must yield `None` so the caller skips the
/// detection. Previously the buffer returned `Some` with
/// `BoundingBox::default()` (a zero-extent box), which the domain
/// `BoundingBox::try_new` would later reject.
#[test]
fn empty_8bit_mask_yields_none() {
  let src = vec![0u8; 4 * 4]; // 4×4 all-zero mask, tight stride.
  assert!(process_mask_bytes_u8::<DomainBoundingBox>(4, 4, 4, &src).is_none());
}

/// An all-zero 32-bit-float mask must also yield `None`. Same
/// rationale as the 8-bit case.
#[test]
fn empty_32fp_mask_yields_none() {
  let src = vec![0u8; 4 * 4 * 4]; // 4×4 all-zero f32 mask.
  assert!(process_mask_bytes_f32::<DomainBoundingBox>(4, 4, 16, &src).is_none());
}

/// An 8-bit mask with one foreground pixel at row 1, col 2 of a
/// 4×4 buffer must round-trip the bbox and the packed bytes.
#[test]
fn single_pixel_8bit_mask_round_trip() {
  let mut src = vec![0u8; 16];
  // Row 1, column 2 — stride 4.
  src[6] = 0xFF;
  let (bbox, packed) =
    process_mask_bytes_u8::<DomainBoundingBox>(4, 4, 4, &src).expect("foreground produces Some");
  assert!((bbox.x() - 0.5).abs() < 1e-6, "x: {}", bbox.x());
  assert!((bbox.y() - 0.25).abs() < 1e-6, "y: {}", bbox.y());
  assert!((bbox.width() - 0.25).abs() < 1e-6, "w: {}", bbox.width());
  assert!((bbox.height() - 0.25).abs() < 1e-6, "h: {}", bbox.height());
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
  let (bbox, packed) =
    process_mask_bytes_f32::<DomainBoundingBox>(4, 4, 16, &src).expect("foreground produces Some");
  assert!((bbox.x() - 0.5).abs() < 1e-6, "x: {}", bbox.x());
  assert!((bbox.y() - 0.25).abs() < 1e-6, "y: {}", bbox.y());
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
  let (_, packed) =
    process_mask_bytes_f32::<DomainBoundingBox>(4, 1, 16, &src).expect("foreground present");
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
  let (_, packed) =
    process_mask_bytes_f32::<DomainBoundingBox>(4, 1, 16, &src).expect("foreground at col 1");
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
  let (bbox, packed) =
    process_mask_bytes_u8::<DomainBoundingBox>(3, 2, 8, &src).expect("foreground produces Some");
  assert_eq!(packed.len(), 3 * 2);
  assert_eq!(packed, [1, 0, 0, 0, 0, 1]);
  // Foreground spans cols 0..=2 and rows 0..=1 — bbox is the whole mask.
  assert!((bbox.x() - 0.0).abs() < 1e-6);
  assert!((bbox.y() - 0.0).abs() < 1e-6);
  assert!((bbox.width() - 1.0).abs() < 1e-6);
  assert!((bbox.height() - 1.0).abs() < 1e-6);
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
  mediaschema::domain::aggregates::video::BoundingBox::try_new(
    bbox.x(),
    bbox.y(),
    bbox.width(),
    bbox.height(),
  )
  .expect("pose-derived bbox satisfies domain invariants");
}

/// Non-finite joint coordinates (NaN/Inf from a glitched Vision
/// observation) must short-circuit before reaching the
/// `BoundingBox::new` constructor.
#[test]
fn pose_bbox_from_nan_joints_yields_none() {
  assert!(pose_bbox_from_joint_bounds::<DomainBoundingBox>(f32::NAN, 0.5, 0.5, 0.5).is_none());
  assert!(pose_bbox_from_joint_bounds::<DomainBoundingBox>(0.1, 0.1, f32::INFINITY, 0.5).is_none());
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
  assert!(process_mask_bytes_u8::<DomainBoundingBox>(width, height, width, &[]).is_none());
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
    vision_point_to_normalized(projected.x, projected.y).expect("projected lower-left is finite");
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
  assert!(vision_point_to_normalized(projected.x, projected.y).is_none());
}

// ──────────────── R7 fixes (codex round 7) ────────────────

/// #20: `sanitize_capture_quality` no longer collapses absent into
/// corrupt's old fallback — `None` (Vision did not provide a value)
/// now stays `None`, the same "never measured" state a non-finite
/// reading also collapses to (see
/// `sanitize_capture_quality_non_finite_returns_none` below). Before
/// this fix, `None` mapped to `Some(0.0)`, indistinguishable from a
/// face Vision genuinely measured and scored at zero — the angles
/// precedent (#18/#19) applied to this seat.
#[test]
fn sanitize_capture_quality_absent_maps_to_none() {
  assert_eq!(sanitize_capture_quality(None), None);
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
  let measured = HeightEstimation::Measured;
  let (h, e) = sanitize_body_height_pair(1.75, measured);
  assert!((h - 1.75).abs() < 1e-6);
  assert_eq!(e, measured);

  let reference = HeightEstimation::Reference;
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
    let (h, e) = sanitize_body_height_pair(raw, HeightEstimation::Measured);
    assert_eq!(h, 0.0, "non-finite must collapse to 0.0 (raw = {raw:?})");
    assert_eq!(
      e,
      HeightEstimation::Unknown,
      "non-finite must force UNKNOWN (raw = {raw:?})",
    );
    // Same for Reference.
    let (h, e) = sanitize_body_height_pair(raw, HeightEstimation::Reference);
    assert_eq!(h, 0.0);
    assert_eq!(e, HeightEstimation::Unknown);
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

// ──────────────── R8 fixes (codex round 8) ────────────────

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

/// `validate_raw_slice_elems::<u8>` rejects when `elem_count *
/// size_of::<T>()` overflows usize. For `T = u8` size_of is 1 so
/// the overflow surfaces only on the isize::MAX check.
#[test]
fn validate_raw_slice_elems_rejects_byte_overflow() {
  // `usize::MAX / 2 + 2` * 16 (size_of CGPoint with two f64) overflows.
  assert!(validate_raw_slice_elems::<CGPoint>(usize::MAX, usize::MAX).is_none());
}

/// The key R8 regression: a 2^24+1-pixel-wide mask with foreground
/// in the rightmost column previously produced `x = 1.0` with
/// positive width (because `f32` cannot distinguish `2^24` from
/// `2^24 + 1`). The f64-intermediate fix should now produce a
/// `[0, 1]`-valid bbox OR drop the detection — never emit
/// `x + width > 1.0`.
#[test]
fn normalized_bbox_handles_2pow24_plus_one_width() {
  // 2^24 = 16,777,216. Pick a width slightly above the f32
  // mantissa exhaustion point. Foreground = rightmost column.
  let width: usize = (1 << 24) + 1;
  let height: usize = 1;
  let right_col = width - 1;
  let bbox = normalized_bbox_from_pixel_bounds::<DomainBoundingBox>(
    right_col, 0, right_col, 0, width, height,
  )
  .expect("valid bbox at right edge");
  // Without the f64 fix this would have been `x = 1.0`. With the
  // fix `x = (2^24) / (2^24 + 1)` ≈ 0.99999994 (f32).
  assert!(
    bbox.x() < 1.0,
    "x must remain strictly less than 1.0: {}",
    bbox.x()
  );
  assert!(
    bbox.width() > 0.0,
    "positive foreground width: {}",
    bbox.width()
  );
  // `x + width` MUST satisfy the schema `<= 1.0` invariant
  // (in fact equals 1.0 modulo f32 representation).
  let right_edge = bbox.x() + bbox.width();
  assert!(
    right_edge <= 1.0 + 1e-6,
    "right edge exceeds image: {right_edge}"
  );
}

/// The normalizer rejects degenerate input (width or height zero,
/// or max < min) by returning `None` rather than emitting a wire
/// bbox the domain validator would reject.
#[test]
fn normalized_bbox_rejects_degenerate_input() {
  assert!(normalized_bbox_from_pixel_bounds::<DomainBoundingBox>(0, 0, 10, 10, 0, 100).is_none());
  assert!(normalized_bbox_from_pixel_bounds::<DomainBoundingBox>(0, 0, 10, 10, 100, 0).is_none());
  // max < min (corrupted input)
  assert!(
    normalized_bbox_from_pixel_bounds::<DomainBoundingBox>(20, 0, 10, 10, 100, 100).is_none()
  );
}

// ──────────────── R9 fixes (codex round 9) ────────────────

/// R8's f64 intermediate fixed the canonical 2^24+1 case but
/// codex round 9 surfaced that the SAME class returns at
/// 2^25+1 — `left = 2^25 / (2^25 + 1)` narrows to `1.0` in f32
/// while `width = 1 / (2^25 + 1)` remains positive. R9's
/// edge-based fix derives width as `right - left` AFTER both
/// narrow to f32, AND explicitly rejects `left >= 1.0` after
/// the narrow.
///
/// Test inputs intentionally span the f32 mantissa-exhaustion
/// power-of-two boundaries (2^24, 2^25, 2^26) plus the cap
/// edge — every one must either emit a valid `[0, 1]` bbox OR
/// return `None`, never `x = 1.0` with positive width.
#[test]
fn normalized_bbox_handles_mantissa_exhaustion_boundaries() {
  // Span the boundaries the codex finding called out.
  for shift in 24u32..=25 {
    let width: usize = (1 << shift) + 1;
    let height: usize = 1;
    let right_col = width - 1;
    let result = normalized_bbox_from_pixel_bounds::<DomainBoundingBox>(
      right_col, 0, right_col, 0, width, height,
    );
    match result {
      None => {
        // Acceptable: the rounding pushed `left` to >= 1.0 and
        // the explicit guard caught it. The detection is dropped,
        // which is the safe semantic.
      }
      Some(bbox) => {
        // If we DO emit a bbox, every invariant must hold —
        // a `[0, 1]`-valid box with positive extent and a right
        // edge that does not exceed the image.
        assert!(
          bbox.x() < 1.0,
          "shift={shift}: x must be < 1.0, got {}",
          bbox.x()
        );
        assert!(
          bbox.width() > 0.0,
          "shift={shift}: width must be > 0.0, got {}",
          bbox.width()
        );
        // f32-safe right-edge check: edge computed directly,
        // not as left + width.
        let right_edge = bbox.x() + bbox.width();
        assert!(
          right_edge <= 1.0 + 1e-6,
          "shift={shift}: right edge exceeds image: {right_edge}",
        );
      }
    }
  }
}

/// Same intent at width close to the 64 MiB cap (the largest
/// allowed width / height combination, where f32 precision
/// is most degraded).
#[test]
fn normalized_bbox_handles_max_mask_bytes_boundary() {
  let width = MAX_MASK_BYTES; // 64 MiB worth of 1-row mask.
  let height = 1usize;
  let right_col = width - 1;
  let result = normalized_bbox_from_pixel_bounds::<DomainBoundingBox>(
    right_col, 0, right_col, 0, width, height,
  );
  if let Some(bbox) = result {
    assert!(
      bbox.x() < 1.0,
      "x must remain strictly less than 1.0: {}",
      bbox.x()
    );
    assert!(
      bbox.width() > 0.0,
      "positive foreground width: {}",
      bbox.width()
    );
    let right_edge = bbox.x() + bbox.width();
    assert!(
      right_edge <= 1.0 + 1e-6,
      "right edge exceeds image: {right_edge}"
    );
  }
  // `None` is also acceptable — see the previous test's rationale.
}

/// `max_x + 1 > width` (corrupted input) must return `None`.
#[test]
fn normalized_bbox_rejects_max_above_dimensions() {
  // max_x = width - 1 is OK (right edge); max_x = width is corrupt.
  assert!(normalized_bbox_from_pixel_bounds::<DomainBoundingBox>(0, 0, 100, 0, 100, 1).is_none());
  assert!(normalized_bbox_from_pixel_bounds::<DomainBoundingBox>(0, 0, 0, 100, 1, 100).is_none());
}

/// Regression pin: `SimdFloat4x4::ENCODING` must format as
/// `{?=[4]}` to match Clang's `@encode(simd_float4x4)` and the
/// runtime metadata of `-[VNHumanBodyRecognizedPoint3D position]`.
/// The previous `Encoding::Unknown` element rendered as `{?=[4?]}`
/// and silently broke every msg_send for that selector under
/// `catch_unwind`. Pinning the string here so a future objc2
/// upgrade or accidental edit surfaces as a test failure.
#[test]
fn simd_float4x4_encoding_matches_clang_at_encode() {
  assert_eq!(SimdFloat4::ENCODING.to_string(), "");
  assert_eq!(SimdFloat4x4::ENCODING.to_string(), "{?=[4]}");
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

// ----- attempt accounting precedes every rejection branch --------------------
//
// Each test below is shaped as an adversarial walk: an input that reaches a
// rejection branch on EVERY step and emits nothing at all. Under the previous
// order every such step was free, so the walk ran to its structural cap
// instead of its ceiling. The assertions pin the ceiling as the bound.

/// A mask walk that emits nothing must still terminate at
/// [`MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME`].
///
/// This is the shape of an `NSIndexSet` whose every index is above
/// `u32::MAX`: `u32::try_from` rejects each one, the emission counters
/// (count, bytes) never move, and the walk advances by pure
/// `indexGreaterThanIndex` traversal. The charge used to sit AFTER that
/// early `continue`, so [`MAX_NESTED_INSTANCES_PER_OBSERVATION`] (64)
/// indices × [`MAX_VISION_RESULTS_PER_FRAME`] (4096) observations =
/// 262,144 index visits ran uncharged against a 1,024 ceiling. Charging at
/// the step's entry bounds the same walk at 1,024.
///
/// The loop's own hard stop is that 262,144 — the reach the ceiling failed
/// to bound — so a regression that drops the charge fails this test rather
/// than hanging.
#[test]
fn mask_walk_that_emits_nothing_stops_at_the_attempt_ceiling() {
  let old_order_reach = MAX_NESTED_INSTANCES_PER_OBSERVATION * MAX_VISION_RESULTS_PER_FRAME;
  assert!(
    old_order_reach > MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME,
    "the walk this test pins must be able to overrun the ceiling: {old_order_reach} visits vs a \
     {MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME} ceiling"
  );

  let mut total_mask_attempts = 0usize;
  let mut steps = 0usize;
  for _ in 0..old_order_reach {
    // Emission counters stay at zero: nothing this walk visits is ever
    // pushed, which is exactly why an emission budget cannot bound it.
    if !charge_mask_walk_step(0, 0, &mut total_mask_attempts) {
      break;
    }
    steps += 1;
  }

  assert_eq!(
    steps, MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME,
    "an all-rejecting mask walk is bounded by the attempt ceiling, not by its structural cap"
  );
  assert_eq!(total_mask_attempts, MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME);
}

/// Every one of the three mask ceilings refuses the step, and a refusal
/// charges nothing — the budget a caller reads after a `false` is the one
/// it had before.
#[test]
fn mask_walk_refusal_charges_nothing() {
  let mut on_count = 0usize;
  assert!(!charge_mask_walk_step(
    0,
    MAX_TOTAL_MASKS_PER_FRAME,
    &mut on_count
  ));
  assert_eq!(on_count, 0, "the emitted-count ceiling charges nothing");

  let mut on_bytes = 0usize;
  assert!(!charge_mask_walk_step(
    MAX_TOTAL_MASK_BYTES_PER_FRAME,
    0,
    &mut on_bytes
  ));
  assert_eq!(on_bytes, 0, "the emitted-bytes ceiling charges nothing");

  let mut on_attempts = MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME;
  assert!(!charge_mask_walk_step(0, 0, &mut on_attempts));
  assert_eq!(
    on_attempts, MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME,
    "the attempt ceiling charges nothing"
  );
}

/// The last step below the ceiling is admitted and lands exactly on it;
/// the next is refused. An off-by-one here would either lose a legitimate
/// mask or overrun the cap.
#[test]
fn mask_walk_admits_an_exact_fit_at_the_attempt_ceiling() {
  let mut attempts = MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME - 1;
  assert!(charge_mask_walk_step(0, 0, &mut attempts));
  assert_eq!(attempts, MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME);
  assert!(!charge_mask_walk_step(0, 0, &mut attempts));
  assert_eq!(attempts, MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME);
}

/// A region Vision did not report costs the frame one attempt unit — and
/// nothing else. Under the previous order the absent region returned before
/// any charge, so it was free.
///
/// This drives the fixed function itself: `None` is precisely what
/// `landmarks.leftPupil()` and its twelve siblings return for a region
/// Vision declined to compute, and it is one of the four rejection branches
/// the visit charge now precedes (the others — an empty region, an over-cap
/// `pointCount`, a null point buffer — need a lying `VNFaceLandmarkRegion2D`
/// to reach, and share this branch's accounting).
#[test]
fn absent_landmark_region_still_charges_its_visit() {
  let face = CGRect::new(CGPoint::new(0.1, 0.1), CGSize::new(0.5, 0.5));
  let mut regions: Vec<DomainFaceLandmarkRegion> = Vec::new();
  let mut total_points_remaining = MAX_FACE_LANDMARK_POINTS_PER_FRAME;
  let mut total_landmark_attempts = 0usize;

  push_face_landmark_region(
    &mut regions,
    "leftPupil",
    None,
    face,
    &mut total_points_remaining,
    &mut total_landmark_attempts,
  );

  assert_eq!(
    total_landmark_attempts, 1,
    "the visit is charged before the absent-region branch can return"
  );
  assert_eq!(
    total_points_remaining, MAX_FACE_LANDMARK_POINTS_PER_FRAME,
    "the EMISSION budget is untouched — a refused region emits no points, so it spends none"
  );
  assert!(regions.is_empty());
}

/// A face set whose every named region is refused must stop at
/// [`MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME`] rather than run to its
/// structural cap.
///
/// The structural cap is 13 named regions × [`MAX_VISION_RESULTS_PER_FRAME`]
/// = 53,248 region visits, every one of them free under the previous order.
/// That total sat below this ceiling only by arithmetic accident
/// (13 × 4096 < 4 × 16,384); charging the visit makes the ceiling the bound
/// by construction, so raising the results cap or lowering the landmark
/// budget cannot silently reopen the gap.
#[test]
fn landmark_walk_whose_regions_are_all_refused_stops_at_the_attempt_ceiling() {
  let face = CGRect::new(CGPoint::new(0.1, 0.1), CGSize::new(0.5, 0.5));
  let mut regions: Vec<DomainFaceLandmarkRegion> = Vec::new();
  let mut total_points_remaining = MAX_FACE_LANDMARK_POINTS_PER_FRAME;
  let mut total_landmark_attempts = 0usize;

  let mut visits = 0usize;
  for _ in 0..MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME + 1 {
    let before = total_landmark_attempts;
    push_face_landmark_region(
      &mut regions,
      "leftPupil",
      None,
      face,
      &mut total_points_remaining,
      &mut total_landmark_attempts,
    );
    if total_landmark_attempts == before {
      break;
    }
    visits += 1;
  }

  assert_eq!(
    visits, MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME,
    "an all-refusing landmark walk is bounded by the attempt ceiling"
  );
  assert_eq!(
    total_landmark_attempts, MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME,
    "and never past it"
  );
  assert!(regions.is_empty());
  assert_eq!(
    total_points_remaining, MAX_FACE_LANDMARK_POINTS_PER_FRAME,
    "no points were emitted, so the emission budget never moved"
  );
}

/// Both landmark ceilings refuse the visit, and a refusal charges nothing.
#[test]
fn landmark_region_visit_refusal_charges_nothing() {
  let mut on_points = 0usize;
  assert!(charge_landmark_region_visit(0, &mut on_points).is_none());
  assert_eq!(
    on_points, 0,
    "an exhausted emission budget refuses the visit without charging"
  );

  let mut on_attempts = MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME;
  assert!(charge_landmark_region_visit(1, &mut on_attempts).is_none());
  assert_eq!(
    on_attempts, MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME,
    "the attempt ceiling refuses the visit without charging"
  );

  // An over-counted budget — the direction a caught Objective-C exception
  // can leave a counter in — reads as exhausted, never as capacity.
  let mut overcounted = MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME + 7;
  assert!(charge_landmark_region_visit(1, &mut overcounted).is_none());
  assert_eq!(overcounted, MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME + 7);
}

/// The visit unit is a FLOOR on a region refused before it walks anything,
/// never a SURCHARGE on one that walks. A region whose points exactly fit
/// the attempt budget available before the visit still walks every one of
/// them, and its total cost is exactly the points it walked — the same
/// total, and the same cap, as before the visit unit existed.
#[test]
fn a_region_that_walks_costs_exactly_the_points_it_walks() {
  const POINT_COUNT: usize = 76;
  for available in [POINT_COUNT + 1, POINT_COUNT, POINT_COUNT - 1] {
    let mut attempts = MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME - available;
    let before = attempts;
    let attempts_remaining =
      charge_landmark_region_visit(MAX_FACE_LANDMARK_POINTS_PER_FRAME, &mut attempts)
        .expect("the budget admits the visit");
    assert_eq!(
      attempts_remaining, available,
      "the walk is sized against the budget as it stood BEFORE the visit"
    );

    let region_cap = charge_landmark_points(
      POINT_COUNT,
      MAX_FACE_LANDMARK_POINTS_PER_FRAME,
      attempts_remaining,
      &mut attempts,
    )
    .expect("a positive cap walks");
    assert_eq!(
      region_cap,
      POINT_COUNT.min(available),
      "the cap falls exactly where it fell before the visit unit existed"
    );
    assert_eq!(
      attempts - before,
      region_cap,
      "the region's total cost is exactly the points it walks"
    );
    assert!(
      attempts <= MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME,
      "and the ceiling is never overshot: {attempts}"
    );
  }
}

/// The emission budget caps the walk the same way, and a frame that cannot
/// afford a single point drops the region whole rather than emitting an
/// empty one.
#[test]
fn landmark_point_charge_respects_the_emission_budget_and_drops_at_zero() {
  let mut attempts = 0usize;
  let capped = charge_landmark_points(500, 12, MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME, &mut attempts)
    .expect("a positive cap walks");
  assert_eq!(capped, 12, "the emission budget caps the walk");
  assert_eq!(attempts, 11, "minus the one unit the visit already paid");

  let mut none_left = 40usize;
  assert!(
    charge_landmark_points(500, 0, MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME, &mut none_left).is_none(),
    "no emittable points left drops the region whole"
  );
  assert_eq!(none_left, 40, "and charges nothing further");

  let mut no_attempts = 40usize;
  assert!(
    charge_landmark_points(500, 500, 0, &mut no_attempts).is_none(),
    "no attempt budget left drops the region whole"
  );
  assert_eq!(no_attempts, 40, "and charges nothing further");
}

/// A configured `max_instances_per_observation` of zero is reachable — the
/// knob is an unbounded `usize` with no lower clamp — which is why the
/// instance walk short-circuits before `allInstances` / `firstIndex` rather
/// than reading an index it would only reject.
#[test]
fn a_zero_instance_cap_is_a_configurable_state() {
  let opts = AppleVisionPersonInstanceMaskOptions::new().with_max_instances_per_observation(0);
  assert_eq!(opts.max_instances_per_observation(), 0);
  assert_eq!(
    opts
      .max_instances_per_observation()
      .min(MAX_NESTED_INSTANCES_PER_OBSERVATION),
    0,
    "the effective inner cap is zero, so the walk must fetch no index at all"
  );
}

/// The visit charge is small enough that it cannot displace a conforming
/// frame's points: 13 regions per face is a rounding error against a
/// ceiling sized at four times the point budget, so the binding constraint
/// on a real frame stays the emission budget it has always been.
#[test]
fn the_visit_charge_cannot_starve_a_conforming_frame() {
  // A generous conforming frame: every point of the emission budget spent,
  // one region visit charged per region that produced them.
  let regions_visited = 13 * MAX_FACE_LANDMARK_POINTS_PER_FRAME / MAX_LANDMARK_POINTS;
  let worst_case = MAX_FACE_LANDMARK_POINTS_PER_FRAME + regions_visited;
  assert!(
    worst_case < MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME,
    "a frame that spends every emittable point still has attempt budget left: {worst_case} vs \
     {MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME}"
  );
}
