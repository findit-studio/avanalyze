//! Full face landmarks: the thirteen named regions, their own entry
//! point, their own request.
//!
//! This is the heavy half of Apple's face landmarking. The cheap
//! five-point reduction lives on [`FaceDetector`](crate::FaceDetector);
//! come here when you want every point Vision computed.

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2_core_foundation::{CGPoint, CGRect};
#[cfg(target_vendor = "apple")]
use objc2_vision::*;

#[cfg(target_vendor = "apple")]
use crate::ffi::{
  MAX_VISION_RESULTS_PER_FRAME, guard_vision_ffi, project_landmark_to_image, run_requests,
  sanitize_confidence, validate_raw_slice_elems, vision_point_to_normalized, vision_rect_to_bbox,
};
use crate::{AnalyzeError, AppleVisionFaceLandmarkOptions, BoundingBox};

/// Upper bound on the number of landmark points per face-landmark
/// region. Vision's `allPoints` is ~76 points; per-feature regions
/// are smaller. 1024 leaves headroom against future API expansion
/// while still capping a corrupted/adversarial `pointCount`.
#[cfg(target_vendor = "apple")]
pub(crate) const MAX_LANDMARK_POINTS: usize = 1024;

/// Hard ceiling on the cumulative face-landmark points emitted per
/// frame across all detections × all named regions × all points.
/// Apple's typical output is at most a few faces × ~76 points each,
/// so 16384 is generous defence-in-depth against the worst-case
/// nested-emission product (4096 × 13 × MAX_LANDMARK_POINTS).
#[cfg(target_vendor = "apple")]
pub(crate) const MAX_FACE_LANDMARK_POINTS_PER_FRAME: usize = 16384;

/// Hard ceiling on the cumulative face-landmark walk attempted per
/// frame across all detections × all named regions: one unit per
/// region VISITED plus one per raw point walked. A corrupted
/// observation set where every region's points fail
/// [`vision_point_to_normalized`]'s finite check (or where the parent
/// detection later fails the bbox / min_region_count gates) would
/// otherwise let the helper walk up to
/// `4096 * 13 * MAX_LANDMARK_POINTS` raw points without the
/// per-frame emission budget ever decreasing. Sized as
/// `4 * MAX_FACE_LANDMARK_POINTS_PER_FRAME` so a successful frame
/// can tolerate non-finite/dropped points before the attempt cap
/// trips. The per-visit unit is what keeps a region Vision refused —
/// absent, empty, over-cap, or null-buffered — from costing nothing at
/// all.
#[cfg(target_vendor = "apple")]
pub(crate) const MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME: usize =
  4 * MAX_FACE_LANDMARK_POINTS_PER_FRAME;

// ----- attempt accounting precedes every rejection branch --------------------
//
// An emission counter rises only on success, so it bounds what a frame
// EMITS and nothing else. The failure paths — a region Vision did not
// report, an empty one, an over-cap `pointCount`, a null point buffer, a
// point that fails the finite check — each cost FFI traversal, and an
// adversarial result set can reach them once per named region per
// observation. Only an ATTEMPT counter bounds that, and only if it is
// charged BEFORE the walk can branch: a charge that sits after an early
// return is not a bound on the walk, it is a bound on the walk's
// productive steps.
//
// The two helpers below are the only places the landmark attempt budget
// is charged. Each fuses its ceiling test with its charge, so a region
// visit cannot reach a rejection branch without having paid, and a
// refusal charges nothing.

/// Gate one face-landmark region visit on the per-frame landmark
/// budgets and charge the attempt budget one unit for the visit itself,
/// indivisibly.
///
/// Without the visit charge, a region Vision did not report, an empty
/// region, an over-cap `pointCount`, and a null point buffer would each
/// be free — reachable once per named region per observation, which is
/// thirteen × [`MAX_VISION_RESULTS_PER_FRAME`] region visits that move
/// neither budget in [`FaceLandmarker`], and up to eight per face in
/// the keypoint reduction ([`landmark_region_points_complete`]).
///
/// Returns the attempt budget that was available BEFORE the visit,
/// which is what [`charge_landmark_points`] sizes the point walk
/// against. The visit unit is therefore a FLOOR on a region refused
/// before it walks anything, never a SURCHARGE on one that walks: a
/// region that walks its points costs exactly those points, and the
/// frame's point cap lands in exactly the place it did before the visit
/// unit existed.
///
/// `None` means a budget is exhausted and the caller stops; a refusal
/// charges nothing.
///
/// `points_remaining` is read, never charged, here — it is the emission
/// budget, spent only by points actually emitted.
#[cfg(target_vendor = "apple")]
#[inline]
pub(crate) fn charge_landmark_region_visit(
  points_remaining: usize,
  attempts: &mut usize,
) -> Option<usize> {
  if points_remaining == 0 {
    return None;
  }
  // `checked_sub` rather than `saturating_sub`: an over-counted budget
  // (the conservative direction a caught Objective-C exception can
  // leave behind) reads as exhausted, never as capacity.
  let attempts_remaining = MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME.checked_sub(*attempts)?;
  if attempts_remaining == 0 {
    return None;
  }
  *attempts = attempts.saturating_add(1);
  Some(attempts_remaining)
}

/// Size one region's point walk against the frame's budgets and charge
/// the balance of the attempt budget for it.
///
/// `attempts_remaining` is the value [`charge_landmark_region_visit`]
/// returned — the budget as it stood BEFORE the visit unit. The walk is
/// therefore capped exactly where it was capped before the visit unit
/// existed, and only `region_cap - 1` is charged here, so the region's
/// total cost is exactly `region_cap`: the points it walks, no more.
///
/// `None` means the frame cannot afford to walk a single point and the
/// region is dropped whole.
#[cfg(target_vendor = "apple")]
#[inline]
pub(crate) fn charge_landmark_points(
  point_count: usize,
  points_remaining: usize,
  attempts_remaining: usize,
  attempts: &mut usize,
) -> Option<usize> {
  let region_cap = point_count.min(points_remaining).min(attempts_remaining);
  if region_cap == 0 {
    return None;
  }
  *attempts = attempts.saturating_add(region_cap - 1);
  Some(region_cap)
}

/// Whether the frame's remaining landmark budgets can cover a
/// COMPLETE walk of `point_count` points.
///
/// [`landmark_region_points`] caps its walk at
/// `point_count.min(points_remaining).min(attempts_remaining)`, so a
/// region is walked end to end exactly when `point_count` fits under
/// both budgets. Callers that derive an aggregate over a whole
/// contour — a centroid, a farthest point, the x-extremes — must
/// consult this first: a prefix yields a confident wrong answer,
/// where the same budget merely shortens a point list for
/// [`FaceLandmarker`]. This is the BUDGET axis of completeness only;
/// [`landmark_region_points_complete`] enforces the data axis as
/// well, refusing a walk the frame could afford but which Vision's
/// own points punctured.
///
/// `attempts` is the counter as it stood BEFORE the caller's visit
/// unit, so this predicate answers exactly where it answered before
/// that unit existed — a visit never shifts the cap.
#[cfg(target_vendor = "apple")]
#[inline]
pub(crate) const fn region_fits_budget(
  point_count: usize,
  points_remaining: usize,
  attempts: usize,
) -> bool {
  point_count <= points_remaining
    && point_count <= MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME.saturating_sub(attempts)
}

/// One named group of face-landmark points.
///
/// The engine emits up to thirteen regions per face, named with
/// Apple's own identifiers (`allPoints`, `faceContour`, `leftEye`, …).
/// Points arrive already projected out of face-relative space into
/// image-normalized, top-left coordinates; a region with no surviving
/// point is never constructed.
pub trait FaceLandmarkRegion: Sized {
  /// Why a region was refused.
  type Error;

  /// Builds a region from its name and its non-empty point list.
  fn try_new(name: &str, points: &[(f32, f32)]) -> Result<Self, Self::Error>;
}

/// A face carrying its landmark regions.
///
/// The `confidence` is the landmark set's own score, not the parent
/// face observation's, and the region count has already been checked
/// against the configured minimum.
pub trait FaceLandmarksDetection: Sized {
  /// Why a landmark detection was refused.
  type Error;
  /// The geometry type this detection is built from.
  type BoundingBox: BoundingBox;
  /// The region type this detection collects.
  type Region: FaceLandmarkRegion;

  /// Builds a landmark detection.
  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    regions: Vec<Self::Region>,
  ) -> Result<Self, Self::Error>;
}

/// Apple Vision full face landmarking — one per worker thread.
///
/// Owns exactly one Vision request. This is the expensive landmark
/// pass; a consumer that only needs eyes, nose and mouth corners
/// should use [`FaceDetector`](crate::FaceDetector)'s
/// [`FaceKeypoints`](crate::FaceKeypoints) instead of paying for all
/// thirteen regions.
///
/// The retained `VNRequest` carries per-call state across
/// `performRequests` / `results()`, so a landmarker is not safe to
/// share across threads; build one per worker.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct FaceLandmarker {
  request: Retained<VNDetectFaceLandmarksRequest>,
}

#[cfg(target_vendor = "apple")]
impl FaceLandmarker {
  /// Creates a landmarker holding the face-landmarks request at its
  /// pinned revision.
  ///
  /// `_options` is unused: Apple bakes no knob this crate exposes into
  /// the request object, so every gate is read per call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionFaceLandmarkOptions) -> Self {
    let request = unsafe {
      let request = VNDetectFaceLandmarksRequest::new();
      request.setRevision(VNDetectFaceLandmarksRequestRevision3);
      request
    };
    Self { request }
  }

  /// Logs the pinned revision of the face-landmarks request.
  ///
  /// A revision drift changes the point roster **silently** — same
  /// API, different geometry.
  #[cfg(feature = "tracing")]
  pub fn log_request_revisions(&self) {
    unsafe {
      tracing::info!(
        face_landmarks_rev = self.request.revision(),
        "initialized pinned Apple Vision request revisions"
      );
    }
  }

  /// Detects faces in `jpeg_data` and returns each with its landmark
  /// regions.
  ///
  /// This is a separate Vision pass from
  /// [`FaceDetector::detect`](crate::FaceDetector::detect) and is not
  /// joined to it: a face may appear in one and not the other.
  pub fn detect<L: FaceLandmarksDetection>(
    &self,
    jpeg_data: &[u8],
    options: &AppleVisionFaceLandmarkOptions,
  ) -> Result<Vec<L>, AnalyzeError> {
    let requests = unsafe { [Retained::cast_unchecked::<VNRequest>(self.request.clone())] };
    run_requests(jpeg_data, &requests, Vec::new(), || {
      guard_vision_ffi("face_landmarks", Vec::new(), || self.extract::<L>(options))
    })
  }

  fn extract<L: FaceLandmarksDetection>(&self, opts: &AppleVisionFaceLandmarkOptions) -> Vec<L> {
    let Some(results) = (unsafe { self.request.results() }) else {
      return Vec::new();
    };

    let mut detections = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    // Per-frame budgets:
    // - `total_points_remaining` — emission budget; tentative-
    //   committed (decremented in a shadow during region extraction
    //   and applied to the master only when the detection survives
    //   every gate).
    // - `total_landmark_attempts` — attempt budget; immediately
    //   committed every time a region is VISITED (one unit, charged
    //   at entry) and again for every point that visit walks,
    //   regardless of whether the region yields anything or the parent
    //   detection ultimately survives. Bounds the FAILURE-PATH work
    //   that the emission budget alone could not catch.
    //
    // The observation-level gates below (absent landmarks, confidence,
    // an invalid face bbox) are deliberately NOT charged: refusing a
    // doomed observation before it can spend the landmark budget is
    // the point of putting them first, and the walk they bound is
    // `MAX_VISION_RESULTS_PER_FRAME` (4096) observation visits — one
    // sixteenth of this attempt ceiling, so no unmetered work escapes
    // a ceiling that claims to bound it.
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
      // face bbox (not the image).
      let face_rect_vision = unsafe { obs.boundingBox() }.standardize();
      // Validate the face bbox BEFORE walking landmarks so an
      // obviously-invalid observation does not spend the landmark
      // attempt budget. Re-using the same schema-conversion guard the
      // post-extraction commit uses.
      if vision_rect_to_bbox::<L::BoundingBox>(face_rect_vision).is_none() {
        continue;
      }

      // Tentative emission budget — commit point-budget consumption
      // ONLY after the detection survives every gate. Attempt budget
      // is committed immediately by the helper.
      let mut tentative_remaining = total_points_remaining;
      let regions = extract_face_landmark_regions::<L::Region>(
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
      let Ok(detection) = L::try_new(bbox, confidence, regions) else {
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
}

/// Read one named landmark region's points, projected out of
/// face-relative space into image-normalized, top-left coordinates.
///
/// Returns an empty vector when the region is absent, empty, fails the
/// raw-slice preconditions, cannot afford a single point, or has every
/// one of its points rejected as non-finite.
///
/// `points_remaining` is read (never written) to bound the slice
/// length; the caller decrements its own budget by the returned
/// length when it commits. `attempts` is charged **immediately** for
/// every point the helper is about to walk, whether or not the parent
/// detection later survives — that is the whole point of an attempt
/// budget.
///
/// The caller has already charged this region's VISIT through
/// [`charge_landmark_region_visit`] — that is what makes the four
/// rejection branches below cost something — and hands its return
/// value on as `attempts_remaining`: the budget as it stood before the
/// visit unit, which is what the point walk is sized against. Only the
/// balance is charged here, so a region that walks costs exactly the
/// points it walks.
#[cfg(target_vendor = "apple")]
pub(crate) fn landmark_region_points(
  region: Option<Retained<VNFaceLandmarkRegion2D>>,
  face_bbox_vision: CGRect,
  points_remaining: usize,
  attempts_remaining: usize,
  attempts: &mut usize,
) -> Vec<(f32, f32)> {
  let Some(region) = region else {
    return Vec::new();
  };

  let point_count = unsafe { region.pointCount() };
  if point_count == 0 {
    return Vec::new();
  }
  // Pre-validate the raw-slice preconditions
  // (count <= MAX_LANDMARK_POINTS AND count * size_of::<CGPoint>() <= isize::MAX)
  // BEFORE the unsafe slice construction. Every Vision boundary that
  // builds a Rust slice from an FFI pointer goes through a
  // `validate_raw_slice_*` gate.
  if validate_raw_slice_elems::<CGPoint>(point_count, MAX_LANDMARK_POINTS).is_none() {
    return Vec::new();
  }

  let points_ptr = unsafe { region.normalizedPoints() };
  if points_ptr.is_null() {
    return Vec::new();
  }

  // Construct the unsafe slice to the CAPPED element count, not
  // the FFI-reported point_count: when remaining < point_count,
  // exposing the full slice to subsequent code would let
  // from_raw_parts trust more elements than we'll read. The cap is
  // `point_count.min(remaining_budget)` — already validated against
  // MAX_LANDMARK_POINTS above. Also bound by the remaining attempt
  // budget so we never visit more points than the frame can afford
  // to walk.
  //
  // Charges the ATTEMPT budget for every point we're about to walk, up
  // front and unconditionally, minus the one unit the visit already
  // paid — so a region that walks costs exactly the points it walks,
  // and the visit unit never shifts where the cap falls. Whether the
  // points later survive finite-checks or the parent detection
  // survives its gates, the walk itself is bounded by this budget.
  let Some(region_cap) =
    charge_landmark_points(point_count, points_remaining, attempts_remaining, attempts)
  else {
    return Vec::new();
  };
  // SAFETY: `points_ptr` points at `point_count` valid `CGPoint`s
  // (Vision API contract). `region_cap <= point_count` and
  // `region_cap <= MAX_LANDMARK_POINTS` (verified via
  // `validate_raw_slice_elems::<CGPoint>` above), so the slice
  // length fits both the FFI buffer and the `isize::MAX` contract.
  let points = unsafe { std::slice::from_raw_parts(points_ptr, region_cap) };
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
  emitted_points
}

/// Read one named landmark region END TO END, or report that it could
/// not be read in full.
///
/// `Some(points)` is a contour the frame could afford to walk in full
/// AND every point of which survived projection, so `points.len()`
/// equals Vision's reported `pointCount`. `Some(empty)` means the
/// region is absent, empty, null, or fails the raw-slice
/// preconditions — the caller's documented fallback region applies.
/// `None` means EITHER the budget could not cover a complete walk OR
/// at least one reported point was not finite; in both cases the
/// caller must refuse the whole reduction rather than derive an
/// aggregate from an incomplete contour.
///
/// Every one of those exits costs one attempt unit, charged for the
/// VISIT before any of them can be taken — see
/// [`charge_landmark_region_visit`]. Only a visit the ceilings refuse
/// outright is free.
#[cfg(target_vendor = "apple")]
pub(crate) fn landmark_region_points_complete(
  region: Option<Retained<VNFaceLandmarkRegion2D>>,
  face_bbox_vision: CGRect,
  points_remaining: usize,
  attempts: &mut usize,
) -> Option<Vec<(f32, f32)>> {
  // Ceiling test AND this visit's own attempt charge, as one step,
  // before any of the four rejection branches below: a region Vision
  // did not report, an empty region, an over-cap `pointCount`, and a
  // null point buffer each cost an FFI read, and this walker's caller
  // reads up to eight regions per face across up to
  // `MAX_VISION_RESULTS_PER_FRAME` faces. A refusal charges nothing.
  //
  // The counter as it stood BEFORE that unit is what both the fit test
  // and the point charge below are sized against, so the cap falls
  // exactly where it fell before the visit unit existed.
  let attempts_before_visit = *attempts;
  let attempts_remaining = charge_landmark_region_visit(points_remaining, attempts)?;

  let Some(region) = region else {
    return Some(Vec::new());
  };

  let point_count = unsafe { region.pointCount() };
  if point_count == 0 {
    return Some(Vec::new());
  }
  // An absurd point count is an unusable region, not a budget
  // refusal: the caller's fallback region applies, exactly as it does
  // for an absent one.
  if validate_raw_slice_elems::<CGPoint>(point_count, MAX_LANDMARK_POINTS).is_none() {
    return Some(Vec::new());
  }

  // Consulted BEFORE the POINT budget is charged, and sized against
  // the attempt counter as it stood before the visit unit: a contour
  // the frame cannot walk end to end must leave the point budget
  // exactly as it found it, so refusing it is not itself a reason the
  // next region is refused. It has still spent its visit unit above —
  // the read that discovered it does not fit is work the frame did.
  if !region_fits_budget(point_count, points_remaining, attempts_before_visit) {
    return None;
  }

  let points_ptr = unsafe { region.normalizedPoints() };
  if points_ptr.is_null() {
    return Some(Vec::new());
  }

  // Charge the ATTEMPT budget for the whole walk, up front, minus the
  // one unit the visit already paid — the budget check above already
  // established the frame can afford every point, so `region_cap` is
  // `point_count` and the region's total cost is exactly that.
  let region_cap =
    charge_landmark_points(point_count, points_remaining, attempts_remaining, attempts)?;
  // SAFETY: `point_count` was validated by
  // `validate_raw_slice_elems::<CGPoint>` against
  // `MAX_LANDMARK_POINTS` and the `isize::MAX` contract above,
  // `region_cap <= point_count`, and Vision reports `point_count`
  // valid `CGPoint`s at `normalizedPoints`.
  let points = unsafe { std::slice::from_raw_parts(points_ptr, region_cap) };
  let mut emitted_points: Vec<(f32, f32)> = Vec::with_capacity(region_cap);
  for point in points.iter() {
    // The same projection `landmark_region_points` performs: out of
    // face-relative space into image-normalized Vision coordinates,
    // then through the top-left flip + clamp + finite check. A point
    // that fails it is not tolerated here — it is simply not pushed,
    // and the length check below turns that silence into a refusal.
    let projected = project_landmark_to_image(*point, face_bbox_vision);
    if let Some((x, y)) = vision_point_to_normalized(projected.x, projected.y) {
      emitted_points.push((x, y));
    }
  }
  // Completeness has two axes and this is the second. The budget
  // check above says the frame could afford the whole walk; this says
  // the whole walk actually produced points. `landmark_region_points`
  // drops a non-finite point and keeps going, which is right for a
  // region seat — a region IS the points it holds. It is wrong for an
  // aggregate: a non-finite value on the true nose tip or on a lip
  // corner removes precisely the point `farthest_from` or
  // `mouth_corners` is looking for, and they then answer confidently
  // over the survivors. A punctured contour is as unusable here as a
  // truncated one, and for the same reason.
  if emitted_points.len() != point_count {
    return None;
  }
  Some(emitted_points)
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
    // Ceiling test AND this region's own visit unit, as one step,
    // before `landmark_region_points` is called at all: an absent
    // region, an empty one, an over-cap `pointCount` and a null point
    // buffer each return from it for free, and each is reachable once
    // per named region per observation — thirteen × 4096 region visits
    // that would move neither budget. A refusal charges nothing and
    // ends the walk.
    //
    // `attempts_remaining` is the budget as it stood BEFORE the unit,
    // which is what the point walk below is sized against, so the
    // frame's point cap falls exactly where it fell before the visit
    // unit existed.
    let Some(attempts_remaining) =
      charge_landmark_region_visit(*total_points_remaining, total_landmark_attempts)
    else {
      break;
    };
    let points = landmark_region_points(
      region,
      face_bbox_vision,
      *total_points_remaining,
      attempts_remaining,
      total_landmark_attempts,
    );
    if points.is_empty() {
      continue;
    }
    let Ok(region) = R::try_new(name, &points) else {
      continue;
    };
    // Decrement the shared budget by the number of points actually
    // emitted (finite-rejected points don't consume budget).
    *total_points_remaining = total_points_remaining.saturating_sub(points.len());
    regions.push(region);
  }
  regions
}

/// Non-macOS stub for [`FaceLandmarker`].
#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct FaceLandmarker;

#[cfg(not(target_vendor = "apple"))]
impl FaceLandmarker {
  /// Constructs a non-macOS stub landmarker. The options are ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionFaceLandmarkOptions) -> Self {
    Self
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn detect<L: FaceLandmarksDetection>(
    &self,
    _jpeg_data: &[u8],
    _options: &AppleVisionFaceLandmarkOptions,
  ) -> Result<Vec<L>, AnalyzeError> {
    crate::error::unsupported()
  }
}
