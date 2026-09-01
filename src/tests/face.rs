//! The face fusion: the per-pass correspondence that seats an
//! annotating pass's readings on the detection spine by observation
//! identity, and the 76→5-point reduction.

use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use smol_str::SmolStr;

use crate::{
  FaceKeypoints,
  face::{centroid, farthest_from, mouth_corners, sanitize_capture_quality, spine_permutation},
  face_landmarks::{
    MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME, MAX_FACE_LANDMARK_POINTS_PER_FRAME, MAX_LANDMARK_POINTS,
    charge_landmark_points, charge_landmark_region_visit, landmark_region_points_complete,
    region_fits_budget,
  },
  ffi::MAX_VISION_RESULTS_PER_FRAME,
};

/// A face box shaped like Vision's: normalized, inside the frame, and
/// big enough that a projected landmark point lands inside `[0, 1]`.
fn face_box() -> CGRect {
  CGRect::new(CGPoint::new(0.1, 0.1), CGSize::new(0.5, 0.5))
}

/// Three uuid strings shaped like Vision's, distinct in their last
/// character so a mis-seating is legible in a failure message.
fn uuids(names: &[&str]) -> Vec<SmolStr> {
  names.iter().map(|n| SmolStr::new(*n)).collect()
}

const A: &str = "6B1E9C3A-0000-4000-8000-00000000000A";
const B: &str = "6B1E9C3A-0000-4000-8000-00000000000B";
const C: &str = "6B1E9C3A-0000-4000-8000-00000000000C";
const STRANGER: &str = "6B1E9C3A-0000-4000-8000-00000000000F";

/// A pass that happened to return its observations in the order it was
/// given resolves to the identity permutation — the no-op case, which
/// on a real 3-face frame is what Vision does roughly half the time.
#[test]
fn results_in_spine_order_resolve_to_the_identity_permutation() {
  assert_eq!(
    spine_permutation(&uuids(&[A, B, C]), &uuids(&[A, B, C])),
    Some(vec![0, 1, 2]),
    "a pass returned in spine order needs no re-ordering"
  );
}

/// The case the whole function exists for. Vision's
/// `VNFaceObservationAccepting` handoff preserves the SET of
/// observation identities but NOT their order: across two independent
/// 30-run measurements on this host, over one real 3-face frame, the
/// uuid set matched the spine 30/30 both times while the ORDER matched
/// only 14/30 and 15/30 (capture quality) and 12/30 and 15/30
/// (landmarks). So a permuted return is not an error path — it is the
/// ordinary path about half the time, and applying the permutation must
/// put every reading back on its own face.
#[test]
fn permuted_results_resolve_to_a_permutation_that_restores_spine_order() {
  let spine = uuids(&[A, B, C]);
  // Vision handed back C, A, B — a rotation, exactly the kind of
  // re-ordering measured above.
  let returned = uuids(&[C, A, B]);
  let permutation = spine_permutation(&spine, &returned).expect("the same three identities");
  assert_eq!(permutation, vec![1, 2, 0]);

  // The permutation is only meaningful applied to a PAYLOAD: these are
  // the capture-quality readings as the pass returned them, one per
  // returned observation, in the returned order.
  let readings_as_returned = [0.5088_f32, 0.4387, 0.3569];
  let seated: Vec<f32> = permutation
    .iter()
    .map(|&index| readings_as_returned[index])
    .collect();
  assert_eq!(
    seated,
    vec![0.4387, 0.3569, 0.5088],
    "face A wears A's reading, B wears B's, C wears C's — read positionally, every one of the \
     three would have worn a neighbour's"
  );
}

/// A pass returning FEWER observations than the spine annotates nothing
/// at all — every face absent, not merely the faces past the end. A
/// short list cannot say WHICH faces it dropped, and with attribution
/// by identity there is no geometry left to fall back on.
#[test]
fn a_short_pass_resolves_to_nothing() {
  assert_eq!(spine_permutation(&uuids(&[A, B, C]), &uuids(&[A, B])), None);
}

/// A pass returning MORE observations than the spine is refused the
/// same way, and for a sharper reason: an extra observation is one the
/// spine never handed over — a face the pass found on its own.
#[test]
fn a_long_pass_resolves_to_nothing() {
  assert_eq!(spine_permutation(&uuids(&[A, B]), &uuids(&[A, B, C])), None);
}

/// A pass whose results are the right LENGTH but name an identity the
/// spine never had. This is the check that array position could never
/// make: the lengths agree, so a positional read would seat all three
/// readings happily, one of them computed for a face this engine never
/// detected. Note the count is unchanged — a substitution, not an
/// addition.
#[test]
fn a_stranger_identity_resolves_to_nothing() {
  assert_eq!(
    spine_permutation(&uuids(&[A, B, C]), &uuids(&[A, STRANGER, C])),
    None,
    "an identity the spine never handed over refuses the whole pass"
  );
}

/// A pass that returned one observation twice. The lengths agree and
/// every spine uuid but one is present, so without the duplicate check
/// two spine faces would resolve to the SAME observation and wear one
/// reading between them — the mis-attribution class this design exists
/// to make impossible.
#[test]
fn a_duplicated_result_identity_resolves_to_nothing() {
  assert_eq!(
    spine_permutation(&uuids(&[A, B, C]), &uuids(&[A, B, B])),
    None,
    "one observation cannot annotate two faces"
  );
}

/// The mirror case, on the spine's side: a spine holding one identity
/// twice. Vision assigns every observation a fresh uuid, so this cannot
/// arise from a sound detection — but nothing in the SIGNATURE says so,
/// and the failure it would cause is the same one: two faces resolving
/// to one observation, each wearing a reading only one of them owns.
/// The lookup therefore CLAIMS each result rather than merely reading
/// it, so the second claim on an identity misses and refuses the pass.
#[test]
fn a_duplicated_spine_identity_resolves_to_nothing() {
  assert_eq!(
    spine_permutation(&uuids(&[A, B, B]), &uuids(&[A, B, C])),
    None,
    "a repeated spine identity cannot be a bijection onto distinct results"
  );
}

/// An empty spine and an empty pass resolve to an empty permutation —
/// `Some`, not `None`. Zero faces correspond perfectly to zero
/// observations, and the annotating readers must produce an empty
/// reading vector for it rather than refusing a pass that did nothing
/// wrong.
#[test]
fn an_empty_spine_resolves_to_an_empty_permutation() {
  assert_eq!(spine_permutation(&[], &[]), Some(Vec::new()));
}

/// `sanitize_capture_quality` does not collapse absent into a real
/// reading: `None` (Vision did not provide a value) stays `None`, the
/// same "never measured" state a non-finite reading also collapses to.
/// Mapping `None` to `Some(0.0)` would be indistinguishable from a
/// face Vision genuinely measured and scored at zero.
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

/// A non-finite captureQuality must NOT be substituted with a real
/// value. `unwrap_or(0.0)` would pass any `min_capture_quality = 0.0`
/// configuration and admit the detection; returning `None` lets the
/// caller's `Option` reach the contract seat as absence.
#[test]
fn sanitize_capture_quality_non_finite_returns_none() {
  assert_eq!(sanitize_capture_quality(Some(f32::NAN)), None);
  assert_eq!(sanitize_capture_quality(Some(f32::INFINITY)), None);
  assert_eq!(sanitize_capture_quality(Some(f32::NEG_INFINITY)), None);
}

/// An eye contour's centre is the mean of its points; a pupil region
/// (a single point) is its own centre.
#[test]
fn centroid_averages_the_contour() {
  assert_eq!(centroid(&[(0.4, 0.5)]), Some((0.4, 0.5)));
  let contour = [(0.2, 0.4), (0.4, 0.4), (0.4, 0.6), (0.2, 0.6)];
  let (x, y) = centroid(&contour).expect("non-empty contour");
  assert!((x - 0.3).abs() < 1e-6, "x: {x}");
  assert!((y - 0.5).abs() < 1e-6, "y: {y}");
}

/// An empty region has no centre — the reduction reports absence
/// rather than fabricating one.
#[test]
fn centroid_of_empty_region_is_none() {
  assert_eq!(centroid(&[]), None);
}

/// The nose tip is the crest point farthest from the eye midpoint.
/// The crest runs from between the eyes down to the tip, so this
/// picks the tip regardless of the order Vision reports the points
/// in — the property the reduction relies on instead of an
/// undocumented point index.
#[test]
fn farthest_from_picks_the_crest_tip() {
  let eye_midpoint = (0.5_f32, 0.40_f32);
  // Crest points from bridge (near the eyes) to tip (far), in order.
  let ordered = [(0.50, 0.43), (0.50, 0.48), (0.50, 0.53), (0.50, 0.58)];
  assert_eq!(farthest_from(&ordered, eye_midpoint), Some((0.50, 0.58)));
  // Reversed input must pick the same point.
  let mut reversed = ordered;
  reversed.reverse();
  assert_eq!(farthest_from(&reversed, eye_midpoint), Some((0.50, 0.58)));
}

/// An empty nose region yields no tip.
#[test]
fn farthest_from_empty_is_none() {
  assert_eq!(farthest_from(&[], (0.5, 0.5)), None);
}

/// The mouth corners are the lip contour's x-extremes.
#[test]
fn mouth_corners_are_the_x_extremes() {
  let lips = [(0.42, 0.70), (0.50, 0.68), (0.58, 0.70), (0.50, 0.73)];
  let (left, right) = mouth_corners(&lips).expect("a four-point contour has two corners");
  assert_eq!(left, (0.42, 0.70));
  assert_eq!(right, (0.58, 0.70));
}

/// A fully rolled (vertical) mouth still yields two distinct,
/// deterministically ordered corners: with every x equal the tie
/// breaks on y, so the corners are the contour's y-extremes rather
/// than the same point twice.
#[test]
fn mouth_corners_break_ties_on_y() {
  let vertical = [(0.50, 0.60), (0.50, 0.75), (0.50, 0.68)];
  let (left, right) = mouth_corners(&vertical).expect("three points are enough");
  assert_eq!(left, (0.50, 0.60));
  assert_eq!(right, (0.50, 0.75));
}

/// A lip contour shorter than two points cannot give two corners, so
/// the whole reduction reports absence rather than a partial set.
#[test]
fn mouth_corners_need_two_points() {
  assert_eq!(mouth_corners(&[]), None);
  assert_eq!(mouth_corners(&[(0.5, 0.7)]), None);
}

/// A contour both budgets can cover is walked end to end, which is
/// what makes the aggregate derived from it an aggregate over the
/// whole contour. Both exact fits are the edges that matter:
/// `landmark_region_points` caps its walk at
/// `point_count.min(points_remaining).min(attempts_remaining)`, so a
/// contour that exactly exhausts either budget is still walked in
/// full and must not be refused.
#[test]
fn region_fits_budget_admits_a_walk_both_budgets_cover() {
  assert!(region_fits_budget(10, 100, 0));
  // Exactly exhausts the emission budget.
  assert!(region_fits_budget(10, 10, 0));
  // Exactly exhausts the attempt budget.
  assert!(region_fits_budget(
    10,
    100,
    MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME - 10
  ));
}

/// A walk either budget would cut short is refused outright — the
/// whole reason the predicate exists. `landmark_region_points` would
/// return the affordable PREFIX instead, and an aggregate over a
/// prefix is a confident wrong answer where the caller needed an
/// honest absence.
#[test]
fn region_fits_budget_refuses_a_walk_either_budget_would_truncate() {
  // One point short on the emission budget.
  assert!(!region_fits_budget(10, 9, 0));
  // One point short on the attempt budget.
  assert!(!region_fits_budget(
    10,
    100,
    MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME - 10 + 1
  ));
  // No emission budget left at all, and a contour to walk.
  assert!(!region_fits_budget(1, 0, 0));
  // The attempt ceiling reached, and already passed.
  assert!(!region_fits_budget(
    1,
    100,
    MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME
  ));
  assert!(!region_fits_budget(
    1,
    100,
    MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME + 1
  ));
}

/// Why a truncated contour must be refused rather than used: the nose
/// tip is the crest point FARTHEST from the eye midpoint, and a crest
/// walked only part way does not contain it. The full crest picks the
/// tip; a two-point prefix of the same crest picks a bridge point —
/// and would report it through the same `Some` as the real tip.
#[test]
fn a_truncated_crest_would_pick_the_wrong_nose_tip() {
  let eye_midpoint = (0.50_f32, 0.40_f32);
  // Bridge (nearest the eyes) to tip (farthest), in Vision's order.
  let crest = [(0.50, 0.43), (0.50, 0.48), (0.50, 0.53), (0.50, 0.58)];
  let tip = farthest_from(&crest, eye_midpoint).expect("a full crest has a tip");
  assert_eq!(tip, (0.50, 0.58), "the full crest picks its far end");

  let prefix = farthest_from(&crest[..2], eye_midpoint).expect("a prefix is still non-empty");
  assert_eq!(prefix, (0.50, 0.48), "the prefix picks its own far end");
  assert_ne!(
    prefix, tip,
    "a budget-truncated crest names a bridge point as the nose tip"
  );
}

/// The mouth corners are the lip contour's x-extremes, so a contour
/// walked only part way reports the extremes OF THE PREFIX. Here the
/// true left corner is the last point Vision reports, so any prefix
/// that stops short names a different point as the left corner.
#[test]
fn a_truncated_lip_contour_would_report_the_wrong_mouth_corner() {
  let lips = [(0.50, 0.68), (0.58, 0.70), (0.50, 0.73), (0.42, 0.70)];
  let (left, right) = mouth_corners(&lips).expect("a full contour has two corners");
  assert_eq!(left, (0.42, 0.70));
  assert_eq!(right, (0.58, 0.70));

  let (prefix_left, prefix_right) =
    mouth_corners(&lips[..3]).expect("a three-point prefix still has two corners");
  assert_eq!(prefix_left, (0.50, 0.68));
  assert_ne!(
    (prefix_left, prefix_right),
    (left, right),
    "a budget-truncated lip contour reports the prefix's extremes as the mouth corners"
  );
}

/// An eye centre is the contour's centroid — an average over every
/// point — so a walk that stops early moves it. The prefix's centre is
/// a perfectly plausible point on the face; it just is not the eye's.
#[test]
fn a_truncated_eye_contour_would_move_the_eye_centre() {
  let eye = [(0.30, 0.40), (0.36, 0.38), (0.42, 0.40), (0.36, 0.42)];
  let centre = centroid(&eye).expect("a full contour has a centre");
  assert!((centre.0 - 0.36).abs() < 1e-6, "x: {}", centre.0);
  assert!((centre.1 - 0.40).abs() < 1e-6, "y: {}", centre.1);

  let prefix = centroid(&eye[..2]).expect("a prefix is still non-empty");
  assert_ne!(
    prefix, centre,
    "a budget-truncated eye contour centres on the points it managed to walk"
  );
}

/// The second axis of completeness, on the nose. A non-finite reading
/// does not shorten a contour from the end — it PUNCTURES it, removing
/// whichever point failed the finite check and leaving the rest. Here
/// the crest's far end (the true tip) is an interior element of
/// Vision's reported order, so no prefix test covers its loss: the
/// full crest picks the tip, the punctured one picks a nearer bridge
/// point, and both answer through the same `Some`. This is why
/// `landmark_region_points_complete` refuses a contour whose surviving
/// point count does not equal Vision's reported `pointCount`.
#[test]
fn a_punctured_crest_would_pick_the_wrong_nose_tip() {
  let eye_midpoint = (0.50_f32, 0.40_f32);
  // The tip sits at index 2, not at either end — the reduction leans
  // on distance precisely because Vision's ordering is undocumented.
  let crest = [(0.50, 0.43), (0.50, 0.48), (0.50, 0.58), (0.50, 0.53)];
  let tip = farthest_from(&crest, eye_midpoint).expect("a full crest has a tip");
  assert_eq!(tip, (0.50, 0.58), "the full crest picks its far end");

  let mut punctured = crest.to_vec();
  punctured.remove(2); // the tip itself read back non-finite
  let punctured_tip =
    farthest_from(&punctured, eye_midpoint).expect("a punctured crest is still non-empty");
  assert_eq!(
    punctured_tip,
    (0.50, 0.53),
    "the punctured crest picks the farthest SURVIVOR"
  );
  assert_ne!(
    punctured_tip, tip,
    "a punctured crest names a point short of the tip as the nose tip"
  );
}

/// The second axis of completeness, on the mouth. The corners are the
/// lip contour's x-extremes, so losing the extreme itself moves a
/// corner inwards while the aggregate reports it as confidently as
/// ever. The true left corner is an INTERIOR element of Vision's
/// reported order here, so a prefix test never reaches this case. This
/// is why `landmark_region_points_complete` refuses a contour whose
/// surviving point count does not equal Vision's reported
/// `pointCount`.
#[test]
fn a_punctured_lip_contour_would_report_the_wrong_mouth_corner() {
  let lips = [(0.50, 0.68), (0.42, 0.70), (0.58, 0.70), (0.50, 0.73)];
  let (left, right) = mouth_corners(&lips).expect("a full contour has two corners");
  assert_eq!(left, (0.42, 0.70));
  assert_eq!(right, (0.58, 0.70));

  let mut punctured = lips.to_vec();
  punctured.remove(1); // the left corner itself read back non-finite
  let (punctured_left, punctured_right) =
    mouth_corners(&punctured).expect("three points still have two corners");
  assert_eq!(
    punctured_left,
    (0.50, 0.68),
    "the punctured contour names a point on the upper lip as the left corner"
  );
  assert_ne!(
    (punctured_left, punctured_right),
    (left, right),
    "a punctured lip contour reports the survivors' extremes as the mouth corners"
  );
}

/// The second axis of completeness, on the eye. A centroid is an
/// average over every point, so dropping one from the middle of the
/// contour shifts the centre — a plausible point on the face that is
/// not the eye's. This is why `landmark_region_points_complete`
/// refuses a contour whose surviving point count does not equal
/// Vision's reported `pointCount`.
#[test]
fn a_punctured_eye_contour_would_move_the_eye_centre() {
  let eye = [(0.30, 0.40), (0.36, 0.38), (0.42, 0.40), (0.36, 0.42)];
  let centre = centroid(&eye).expect("a full contour has a centre");
  assert!((centre.0 - 0.36).abs() < 1e-6, "x: {}", centre.0);
  assert!((centre.1 - 0.40).abs() < 1e-6, "y: {}", centre.1);

  let mut punctured = eye.to_vec();
  punctured.remove(1); // an interior point read back non-finite
  let punctured_centre = centroid(&punctured).expect("a punctured contour is still non-empty");
  assert_ne!(
    punctured_centre, centre,
    "a punctured eye contour centres on the points that survived the finite check"
  );
}

// ----- attempt accounting precedes every rejection branch --------------------
//
// Each test below is shaped as an adversarial walk: an input that reaches
// a rejection branch on EVERY step and emits nothing at all. Under the
// previous order every such step was free, so the walk ran to its
// structural cap instead of its ceiling. The assertions pin the ceiling
// as the bound.

/// A region Vision did not report costs the frame one attempt unit —
/// and nothing else. Under the previous order the absent region
/// returned before any charge, so it was free.
///
/// This drives the fixed walker itself: `None` is precisely what
/// `landmarks.leftPupil()` and its siblings return for a region Vision
/// declined to compute, and it is one of the four rejection branches
/// the visit charge now precedes (the others — an empty region, an
/// over-cap `pointCount`, a null point buffer — need a lying
/// `VNFaceLandmarkRegion2D` to reach, and share this branch's
/// accounting).
///
/// The EMISSION budget is untouched by construction here: the walker
/// reads `points_remaining` by value and the caller decrements by the
/// length it returns, which for a refused region is zero.
#[test]
fn absent_landmark_region_still_charges_its_visit() {
  let mut attempts = 0usize;

  let points = landmark_region_points_complete(
    None,
    face_box(),
    MAX_FACE_LANDMARK_POINTS_PER_FRAME,
    &mut attempts,
  )
  .expect("an absent region is a complete read of nothing, not a refusal");

  assert_eq!(
    attempts, 1,
    "the visit is charged before the absent-region branch can return"
  );
  assert!(
    points.is_empty(),
    "a refused region emits no points, so it spends none of the emission budget"
  );
}

/// A face set whose every named region is refused must stop at
/// [`MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME`] rather than run to its
/// structural cap.
///
/// The structural cap is thirteen named regions ×
/// [`MAX_VISION_RESULTS_PER_FRAME`] = 53,248 region visits for the
/// landmarker, every one of them free under the previous order. That
/// total sat below this ceiling only by arithmetic accident
/// (13 × 4096 < 4 × 16,384); charging the visit makes the ceiling the
/// bound by construction, so raising the results cap or lowering the
/// landmark budget cannot silently reopen the gap.
#[test]
fn landmark_walk_whose_regions_are_all_refused_stops_at_the_attempt_ceiling() {
  let structural_cap = 13 * MAX_VISION_RESULTS_PER_FRAME;
  assert!(
    structural_cap > 0,
    "the walk this test pins is the one the region roster and the results cap compose"
  );

  let mut attempts = 0usize;
  let mut visits = 0usize;
  for _ in 0..MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME + 1 {
    let before = attempts;
    let read = landmark_region_points_complete(
      None,
      face_box(),
      MAX_FACE_LANDMARK_POINTS_PER_FRAME,
      &mut attempts,
    );
    if attempts == before {
      assert_eq!(read, None, "a visit that charged nothing is a refusal");
      break;
    }
    assert_eq!(read, Some(Vec::new()), "and one that charged is a read");
    visits += 1;
  }

  assert_eq!(
    visits, MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME,
    "an all-refusing landmark walk is bounded by the attempt ceiling"
  );
  assert_eq!(
    attempts, MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME,
    "and never past it"
  );
}

/// Both landmark ceilings refuse the visit, and a refusal charges
/// nothing.
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

  // An over-counted budget — the direction a caught Objective-C
  // exception can leave a counter in — reads as exhausted, never as
  // capacity.
  let mut overcounted = MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME + 7;
  assert!(charge_landmark_region_visit(1, &mut overcounted).is_none());
  assert_eq!(overcounted, MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME + 7);
}

/// The visit unit is a FLOOR on a region refused before it walks
/// anything, never a SURCHARGE on one that walks. A region whose points
/// exactly fit the attempt budget available before the visit still
/// walks every one of them, and its total cost is exactly the points it
/// walked — the same total, and the same cap, as before the visit unit
/// existed.
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

/// The emission budget caps the walk the same way, and a frame that
/// cannot afford a single point drops the region whole rather than
/// emitting an empty one.
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

/// The visit charge is small enough that it cannot displace a
/// conforming frame's points: thirteen regions per face is a rounding
/// error against a ceiling sized at four times the point budget, so the
/// binding constraint on a real frame stays the emission budget it has
/// always been.
#[test]
fn the_visit_charge_cannot_starve_a_conforming_frame() {
  // A generous conforming frame: every point of the emission budget
  // spent, one region visit charged per region that produced them.
  let regions_visited = 13 * MAX_FACE_LANDMARK_POINTS_PER_FRAME / MAX_LANDMARK_POINTS;
  let worst_case = MAX_FACE_LANDMARK_POINTS_PER_FRAME + regions_visited;
  assert!(
    worst_case < MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME,
    "a frame that spends every emittable point still has attempt budget left: {worst_case} vs \
     {MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME}"
  );
}

/// The split's SECOND walker, the one feeding this reduction, has no
/// counterpart upstream — and it carried the identical defect. Four
/// branches returned with zero charge: an absent region, an empty one,
/// an over-cap `pointCount`, and a null `normalizedPoints`. The
/// reduction reads up to eight regions per face (a pupil then its eye
/// contour, twice; the nose crest then the nostril contour; the two lip
/// contours) across up to [`MAX_VISION_RESULTS_PER_FRAME`] faces, so
/// those visits were free at scale.
///
/// Both halves of the fix are here. Every region the reduction visits
/// now costs one unit and only one — and the point walk is still sized
/// against the budget as it stood BEFORE that unit, so a contour that
/// exactly fills the frame's remaining attempts is still walked end to
/// end. Sizing the fit test after the unit would refuse that contour
/// and silently drop the whole face's reduction.
#[test]
fn the_second_walker_charges_every_visit_and_still_admits_an_exact_fit() {
  const READS_PER_FACE: usize = 8;
  let mut attempts = 0usize;
  for read in 0..READS_PER_FACE {
    let points = landmark_region_points_complete(
      None,
      face_box(),
      MAX_FACE_LANDMARK_POINTS_PER_FRAME,
      &mut attempts,
    )
    .expect("an absent region is a complete read of nothing");
    assert!(points.is_empty());
    assert_eq!(
      attempts,
      read + 1,
      "every region this reduction visits costs one unit, and only one"
    );
  }

  // The exact fit, in the shape the walker composes it: the fit test
  // sized against the pre-visit counter, then the point charge sized
  // against the pre-visit remainder.
  const CONTOUR: usize = 76;
  let mut attempts = MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME - CONTOUR;
  let before_visit = attempts;
  let attempts_remaining =
    charge_landmark_region_visit(MAX_FACE_LANDMARK_POINTS_PER_FRAME, &mut attempts)
      .expect("the budget admits the visit");
  assert!(
    region_fits_budget(CONTOUR, MAX_FACE_LANDMARK_POINTS_PER_FRAME, before_visit),
    "a contour that exactly fills the remaining attempts is still walked end to end"
  );
  let region_cap = charge_landmark_points(
    CONTOUR,
    MAX_FACE_LANDMARK_POINTS_PER_FRAME,
    attempts_remaining,
    &mut attempts,
  )
  .expect("a complete walk is affordable");
  assert_eq!(region_cap, CONTOUR, "and it is walked in full");
  assert_eq!(
    attempts, MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME,
    "costing exactly the points it walked, landing exactly on the ceiling"
  );
}

/// The five points read back in the canonical alignment order.
#[test]
fn face_keypoints_round_trip_in_canonical_order() {
  let k = FaceKeypoints::new(
    (0.35, 0.40),
    (0.65, 0.40),
    (0.50, 0.55),
    (0.38, 0.70),
    (0.62, 0.70),
  );
  assert_eq!(k.left_eye(), (0.35, 0.40));
  assert_eq!(k.right_eye(), (0.65, 0.40));
  assert_eq!(k.nose_tip(), (0.50, 0.55));
  assert_eq!(k.mouth_left(), (0.38, 0.70));
  assert_eq!(k.mouth_right(), (0.62, 0.70));
  assert_eq!(
    k.points(),
    [
      (0.35, 0.40),
      (0.65, 0.40),
      (0.50, 0.55),
      (0.38, 0.70),
      (0.62, 0.70)
    ]
  );
}
