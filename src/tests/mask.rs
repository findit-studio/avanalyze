//! The mask-buffer machinery: bounded allocation, the two pixel
//! formats, and the pixel-bounds → normalized-bbox narrowing.

use mediaschema::domain::aggregates::video::BoundingBox as DomainBoundingBox;

use crate::{
  AppleVisionPersonInstanceMaskOptions,
  ffi::MAX_VISION_RESULTS_PER_FRAME,
  person_mask::{
    MAX_MASK_BYTES, MAX_NESTED_INSTANCES_PER_OBSERVATION, MAX_TOTAL_MASK_ATTEMPTS_PER_CALL,
    MAX_TOTAL_MASK_BYTES_PER_CALL, MAX_TOTAL_MASKS_PER_CALL, MaskBudget,
    normalized_bbox_from_pixel_bounds, process_mask_bytes_f32, process_mask_bytes_u8,
    try_alloc_packed_mask, validate_mask_dims_for_slice,
  },
};

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
/// detection rather than emitting a zero-extent box the domain
/// `BoundingBox::try_new` would reject.
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
/// u8 wire payload.
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
  // `f32::round()` is half-away-from-zero: 127.5 → 128.
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

/// `process_mask_bytes_u8` propagates the bounded allocation:
/// feeding dimensions whose product exceeds the cap returns `None`
/// instead of attempting the alloc. The source slice need not be
/// filled — the function returns at the allocation step before
/// reading any pixel.
#[test]
fn process_mask_bytes_u8_caps_allocation() {
  let width = MAX_MASK_BYTES + 1;
  let height = 1;
  assert!(process_mask_bytes_u8::<DomainBoundingBox>(width, height, width, &[]).is_none());
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

/// A 2^24+1-pixel-wide mask with foreground in the rightmost column
/// must not produce `x = 1.0` with positive width (`f32` cannot
/// distinguish `2^24` from `2^24 + 1`). The f64-intermediate
/// narrowing produces a `[0, 1]`-valid bbox or drops the detection —
/// never `x + width > 1.0`.
#[test]
fn normalized_bbox_handles_2pow24_plus_one_width() {
  let width: usize = (1 << 24) + 1;
  let height: usize = 1;
  let right_col = width - 1;
  let bbox = normalized_bbox_from_pixel_bounds::<DomainBoundingBox>(
    right_col, 0, right_col, 0, width, height,
  )
  .expect("valid bbox at right edge");
  // Without the f64 intermediate this would have been `x = 1.0`.
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

/// The mantissa-exhaustion class returns at 2^25+1 as well as
/// 2^24+1: `left = 2^25 / (2^25 + 1)` narrows to `1.0` in f32 while
/// `width = 1 / (2^25 + 1)` remains positive. The edge-based
/// computation derives width as `right - left` AFTER both narrow to
/// f32 and explicitly rejects `left >= 1.0` after the narrow.
///
/// Inputs intentionally span the f32 mantissa-exhaustion
/// power-of-two boundaries — every one must either emit a valid
/// `[0, 1]` bbox OR return `None`, never `x = 1.0` with positive
/// width.
#[test]
fn normalized_bbox_handles_mantissa_exhaustion_boundaries() {
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
        // f32-safe right-edge check: edge computed directly, not as
        // left + width.
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

// ----- attempt accounting precedes every rejection branch --------------------
//
// Each test below is shaped as an adversarial walk: an input that reaches
// a rejection branch on EVERY step and emits nothing at all. Under the
// previous order every such step was free, so the walk ran to its
// structural cap instead of its ceiling. The assertions pin the ceiling
// as the bound.

/// A mask walk that emits nothing must still terminate at
/// [`MAX_TOTAL_MASK_ATTEMPTS_PER_CALL`].
///
/// This is the shape of an `NSIndexSet` whose every index is above
/// `u32::MAX`: `u32::try_from` rejects each one, the emission counters
/// (count, bytes) never move, and the walk advances by pure
/// `indexGreaterThanIndex` traversal. The charge used to sit AFTER that
/// early `continue`, so [`MAX_NESTED_INSTANCES_PER_OBSERVATION`] (64)
/// indices × [`MAX_VISION_RESULTS_PER_FRAME`] (4096) observations =
/// 262,144 index visits ran uncharged against a 1,024 ceiling. Charging
/// at the step's entry bounds the same walk at 1,024.
///
/// The loop's own hard stop is that 262,144 — the reach the ceiling
/// failed to bound — so a regression that drops the charge fails this
/// test rather than hanging.
#[test]
fn mask_walk_that_emits_nothing_stops_at_the_attempt_ceiling() {
  let old_order_reach = MAX_NESTED_INSTANCES_PER_OBSERVATION * MAX_VISION_RESULTS_PER_FRAME;
  assert!(
    old_order_reach > MAX_TOTAL_MASK_ATTEMPTS_PER_CALL,
    "the walk this test pins must be able to overrun the ceiling: {old_order_reach} visits vs a \
     {MAX_TOTAL_MASK_ATTEMPTS_PER_CALL} ceiling"
  );

  // The emission counters stay at zero for the whole walk: nothing it
  // visits is ever pushed, which is exactly why an emission budget
  // cannot bound it.
  let mut budget = MaskBudget::new();
  let mut steps = 0usize;
  for _ in 0..old_order_reach {
    if !budget.charge_walk_step() {
      break;
    }
    steps += 1;
  }

  assert_eq!(
    steps, MAX_TOTAL_MASK_ATTEMPTS_PER_CALL,
    "an all-rejecting mask walk is bounded by the attempt ceiling, not by its structural cap"
  );
}

/// Every one of the three mask ceilings refuses the step, and a refusal
/// charges nothing — the budget a caller reads after a `false` is the
/// one it had before, all three counters of it.
#[test]
fn mask_walk_refusal_charges_nothing() {
  // The emitted-count ceiling: 256 zero-byte emissions, no attempts.
  let mut on_count = MaskBudget::new();
  for _ in 0..MAX_TOTAL_MASKS_PER_CALL {
    on_count.charge_emission(0);
  }
  let before_count = on_count.clone();
  assert!(!on_count.charge_walk_step());
  assert_eq!(
    on_count, before_count,
    "the emitted-count ceiling charges nothing"
  );

  // The emitted-bytes ceiling, reached by a single maximal emission.
  let mut on_bytes = MaskBudget::new();
  on_bytes.charge_emission(MAX_TOTAL_MASK_BYTES_PER_CALL);
  let before_bytes = on_bytes.clone();
  assert!(!on_bytes.charge_walk_step());
  assert_eq!(
    on_bytes, before_bytes,
    "the emitted-bytes ceiling charges nothing"
  );

  // The attempt ceiling, reached by the charge under test itself.
  let mut on_attempts = MaskBudget::new();
  for _ in 0..MAX_TOTAL_MASK_ATTEMPTS_PER_CALL {
    assert!(on_attempts.charge_walk_step());
  }
  let before_attempts = on_attempts.clone();
  assert!(!on_attempts.charge_walk_step());
  assert_eq!(
    on_attempts, before_attempts,
    "the attempt ceiling charges nothing"
  );
}

/// The ceiling is a bound, not an off-by-one: every step below it is
/// admitted, the last lands exactly on it, and only the next is refused.
/// An error here would either lose a legitimate mask or overrun the cap.
#[test]
fn mask_walk_admits_an_exact_fit_at_the_attempt_ceiling() {
  let mut budget = MaskBudget::new();
  for step in 0..MAX_TOTAL_MASK_ATTEMPTS_PER_CALL {
    assert!(
      budget.charge_walk_step(),
      "step {step} is inside the ceiling and must be admitted"
    );
  }
  let full = budget.clone();
  assert!(
    !budget.charge_walk_step(),
    "and the step after the exact fit is refused"
  );
  assert_eq!(budget, full, "that refusal, too, charges nothing");
}

/// A configured `max_instances_per_observation` of zero is reachable —
/// the knob is an unbounded `usize` with no lower clamp — which is why
/// the instance walk short-circuits before `allInstances` / `firstIndex`
/// rather than reading an index it would only reject.
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
