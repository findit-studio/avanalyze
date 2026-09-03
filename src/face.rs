//! Face detection: three Vision passes fused into one record per
//! face, attributed by observation identity.

#[cfg(target_vendor = "apple")]
use std::collections::HashMap;

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2_core_foundation::CGRect;
#[cfg(target_vendor = "apple")]
use objc2_foundation::NSArray;
#[cfg(target_vendor = "apple")]
use objc2_vision::*;
#[cfg(target_vendor = "apple")]
use smol_str::SmolStr;

use crate::{AnalyzeError, AppleVisionFaceOptions, BoundingBox, PixelPlane};
#[cfg(target_vendor = "apple")]
use crate::{
  face_landmarks::{
    MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME, MAX_FACE_LANDMARK_POINTS_PER_FRAME,
    landmark_region_points_complete,
  },
  ffi::{
    ImageSource, MAX_VISION_RESULTS_PER_FRAME, Performed, ffi_nsstring_to_smolstr, finite_f32,
    guard_vision_ffi, perform, sanitize_confidence, vision_rect_to_bbox, with_image,
  },
};

/// The five canonical face keypoints, reduced from Vision's 76-point
/// landmark set.
///
/// Every coordinate is `(x, y)`, normalized to `0.0..=1.0` with a
/// top-left origin — the same convention as
/// [`BoundingBox`](crate::BoundingBox), and in **image** space, not
/// face-relative space: the engine has already projected them out of
/// Vision's face-bbox-relative landmark coordinates.
///
/// # How the five are derived
///
/// - `left_eye` / `right_eye` — the centroid of Vision's `leftPupil` /
///   `rightPupil` region when it reports one, else the centroid of the
///   `leftEye` / `rightEye` contour. The naming is Vision's own, which
///   labels the regions in image space.
/// - `nose_tip` — the point of Vision's `noseCrest` region (or `nose`,
///   when the crest is absent) **farthest from the midpoint of the two
///   eye centres**. The crest runs from between the eyes down to the
///   tip, so its far end is the tip regardless of the order Vision
///   reports the points in; nothing here depends on an undocumented
///   point index.
/// - `mouth_left` / `mouth_right` — the minimum-x and maximum-x points
///   of Vision's `outerLips` contour (or `innerLips`, when the outer
///   contour is absent), ties broken by `y`. These are **image-space**
///   extremes: for an upright face they agree with the eye naming; for
///   a strongly rolled one they need not, and the caller has
///   [`roll`](FaceDetection::try_new) if that matters.
///
/// All five must be derivable or the engine emits `None` rather than a
/// partial set — a four-point "five-point reduction" is not a thing any
/// alignment consumer can use. Every one of the five is an aggregate
/// over a WHOLE contour — a centroid, a farthest point, an x-extreme —
/// so a contour that is incomplete for EITHER reason yields `None` as
/// well: a landmark budget that could not cover the walk, or a point
/// Vision reported as non-finite. A prefix and a punctured contour
/// both produce a confident wrong point rather than an honest absence,
/// and the aggregate cannot tell either from the whole.
///
/// Cropping and alignment are the caller's business: this crate
/// produces geometry and never an identity, an embedding, or a cut
/// image.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FaceKeypoints {
  left_eye: (f32, f32),
  right_eye: (f32, f32),
  nose_tip: (f32, f32),
  mouth_left: (f32, f32),
  mouth_right: (f32, f32),
}

impl FaceKeypoints {
  /// Builds a keypoint set from the five normalized, top-left-origin
  /// points, in the canonical order.
  #[inline]
  pub const fn new(
    left_eye: (f32, f32),
    right_eye: (f32, f32),
    nose_tip: (f32, f32),
    mouth_left: (f32, f32),
    mouth_right: (f32, f32),
  ) -> Self {
    Self {
      left_eye,
      right_eye,
      nose_tip,
      mouth_left,
      mouth_right,
    }
  }

  /// The left eye centre.
  #[inline]
  pub const fn left_eye(&self) -> (f32, f32) {
    self.left_eye
  }

  /// The right eye centre.
  #[inline]
  pub const fn right_eye(&self) -> (f32, f32) {
    self.right_eye
  }

  /// The nose tip.
  #[inline]
  pub const fn nose_tip(&self) -> (f32, f32) {
    self.nose_tip
  }

  /// The left mouth corner.
  #[inline]
  pub const fn mouth_left(&self) -> (f32, f32) {
    self.mouth_left
  }

  /// The right mouth corner.
  #[inline]
  pub const fn mouth_right(&self) -> (f32, f32) {
    self.mouth_right
  }

  /// The five points in canonical order — left eye, right eye, nose
  /// tip, left mouth corner, right mouth corner.
  ///
  /// This is the order affine-alignment routines expect, which is why
  /// it is fixed and documented rather than left to field order.
  #[inline]
  pub const fn points(&self) -> [(f32, f32); 5] {
    [
      self.left_eye,
      self.right_eye,
      self.nose_tip,
      self.mouth_left,
      self.mouth_right,
    ]
  }
}

/// One detected face.
///
/// [`FaceDetector`] fuses three Vision passes into this one record, so
/// two of its seats — `capture_quality` and `keypoints` — are computed
/// by passes other than the one that produced `bbox`. They still reach
/// this face by **identity**, not by geometry: the annotating passes
/// are handed this face's own observation and return it enriched, so a
/// reading arrives at the face Vision computed it for because it IS
/// that face's observation coming back. There is no overlap join and no
/// way for one face to wear another's reading.
///
/// The three annotated seats are `Option` all the same, and absence now
/// says exactly one thing: **Vision did not compute that reading for
/// this face**. Never a join-miss, because there is no join.
///
/// `capture_quality` is three states: `Some(q)` — Vision measured this
/// face's capture quality and reported `q`; `Some(0.0)` — Vision
/// measured it and found it terrible, a real zero reading; `None` —
/// Vision never measured it, whether because the raw reading was nil or
/// because the capture-quality pass did not come back for this frame.
/// `Some(0.0)` and `None` are not interchangeable: collapsing `None` to
/// `Some(0.0)` tells every "quality below X" query that a face the
/// quality pass never scored was scored at the worst possible value.
///
/// `roll` / `yaw` / `pitch` are each an `Option<f32>` in radians.
/// Vision estimates the three independently, and which it reports
/// varies by OS version and by detection path — a face may arrive
/// with all three, some, or none. `Some(0.0)` is a head Vision
/// measured and found level; `None` is an angle Vision never
/// computed. The two are not interchangeable: collapsing `None` to
/// `0.0` tells every "pitched down more than 20°" query that a face
/// Vision never looked at was measured perfectly upright.
///
/// `keypoints` is the 76→5-point reduction, or `None` when Vision
/// returned no landmark set for this face, when the landmark reduction
/// refused it (a confidence below the gate, a contour that could not be
/// read end to end, or a per-frame budget already spent), or when the
/// landmarks pass did not come back. For every point Vision computed,
/// use [`FaceLandmarker`](crate::FaceLandmarker) instead.
pub trait FaceDetection: Sized {
  /// Why a face was refused.
  type Error;
  /// The geometry type this face is built from.
  type BoundingBox: BoundingBox;

  /// Builds a face detection. `capture_quality` is `None` where
  /// Vision never measured this face's capture quality; `roll` /
  /// `yaw` / `pitch` are `None` where Vision did not compute that
  /// angle; `keypoints` is `None` where the five-point reduction was
  /// not available for this face.
  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    capture_quality: Option<f32>,
    roll: Option<f32>,
    yaw: Option<f32>,
    pitch: Option<f32>,
    keypoints: Option<FaceKeypoints>,
  ) -> Result<Self, Self::Error>;
}

/// Apple Vision face detection — one per worker thread.
///
/// Owns **three** Vision requests and fuses them into one record per
/// face: the face-rectangles pass is the detection spine (every
/// detected face appears exactly once), annotated with the
/// capture-quality pass and with the landmarks pass reduced to
/// [`FaceKeypoints`]. No face is counted twice, and the two annotating
/// passes contribute no records of their own.
///
/// **Attribution is identity, not geometry — and not array position.**
/// The rectangles pass runs FIRST, and its observations are handed to
/// the capture-quality and landmarks requests through Vision's own
/// `VNFaceObservationAccepting` protocol. Those requests process
/// exactly those faces and return them enriched, each carrying the
/// `VNObservation.uuid` it was given. That uuid is the correspondence
/// token: a reading reaches the face Vision computed it for because the
/// observation carrying it names that face.
///
/// The ORDER of the returned observations is not part of the
/// correspondence, because Vision does not preserve it — and does not
/// even vary it deterministically: the same image handed to the same
/// detector twice comes back in two different orders about half the
/// time, while the uuid SET matches every time. Reading these passes by
/// position would therefore mis-attribute roughly half of all
/// multi-face frames. The engine resolves each pass's uuids to spine
/// positions instead (`spine_permutation`, which carries the
/// measurement) and refuses the pass outright if that resolution is not
/// a bijection.
///
/// There is no overlap join, no IoU floor, and no way for one face to
/// wear another's reading. The consumer's [`BoundingBox::try_new`] is
/// consulted exactly once per face, at emission, and a refusal there
/// drops that face alone.
///
/// The cost is one extra perform. The three requests can no longer run
/// in a single batch — the spine must exist before the other two can be
/// fed — so a face detection is **two** `performRequests` on one image
/// instead of one.
///
/// The retained `VNRequest`s carry per-call state across
/// `performRequests` / `results()` — and, for the two annotating
/// requests, the input observations set on them for the call — so a
/// detector is not safe to share across threads; build one per worker.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct FaceDetector {
  rectangles: Retained<VNDetectFaceRectanglesRequest>,
  quality: Retained<VNDetectFaceCaptureQualityRequest>,
  landmarks: Retained<VNDetectFaceLandmarksRequest>,
}

#[cfg(target_vendor = "apple")]
impl FaceDetector {
  /// Creates a detector holding the three face requests at their
  /// pinned revisions.
  ///
  /// `_options` is unused: Apple bakes no knob this crate exposes into
  /// these request objects, so every gate is read per call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionFaceOptions) -> Self {
    unsafe {
      let rectangles = VNDetectFaceRectanglesRequest::new();
      rectangles.setRevision(VNDetectFaceRectanglesRequestRevision3);

      let quality = VNDetectFaceCaptureQualityRequest::new();
      quality.setRevision(VNDetectFaceCaptureQualityRequestRevision3);

      let landmarks = VNDetectFaceLandmarksRequest::new();
      landmarks.setRevision(VNDetectFaceLandmarksRequestRevision3);

      Self {
        rectangles,
        quality,
        landmarks,
      }
    }
  }

  /// Logs the pinned revision of all three face requests.
  ///
  /// A revision drift changes which faces are found and how they are
  /// scored **silently** — same API, different numbers.
  #[cfg(feature = "tracing")]
  pub fn log_request_revisions(&self) {
    unsafe {
      tracing::info!(
        face_rectangles_rev = self.rectangles.revision(),
        face_quality_rev = self.quality.revision(),
        face_landmarks_rev = self.landmarks.revision(),
        "initialized pinned Apple Vision request revisions"
      );
    }
  }

  /// Detects every face in `jpeg_data`, one record per face.
  ///
  /// Two performs on one image. The face-rectangles pass runs alone
  /// first and becomes the detection spine; its observations are then
  /// set as the `inputFaceObservations` of the capture-quality and
  /// landmarks requests, which run together and hand those same
  /// observations back enriched. The annotating passes therefore never
  /// see a face the spine did not give them, and never return one the
  /// spine cannot place — but they DO return them in an order of their
  /// own choosing, so each pass's readings are seated by the
  /// `VNObservation.uuid` each returned observation carries, not by
  /// where it landed in the array.
  ///
  /// `min_capture_quality` drops faces whose quality compares below
  /// it, treating an unmeasured face as `0.0` **for that comparison
  /// only**: the default (0.1) drops quality-scored-low and unmeasured
  /// faces alike, while `min_capture_quality == 0.0` keeps every
  /// detected face. A face that passes unmeasured still carries `None`
  /// to [`FaceDetection::try_new`], never `Some(0.0)`.
  ///
  /// What can still cost a caller a face, stated in full: the
  /// rectangles pass's own confidence gate; a consumer's
  /// [`BoundingBox::try_new`] refusing that face's box at emission; a
  /// spine truncated past the per-frame results ceiling, which loses
  /// the faces past it and fabricates nothing for the ones it keeps;
  /// and an observation whose uuid string could not be read, which
  /// `FaceDetector::collect_spine` drops as unusable — a branch a
  /// 36-character uuid cannot reach in practice.
  /// What can cost a face its ANNOTATIONS: an annotating pass that did
  /// not come back, or one whose results do not correspond to the
  /// spine. Each of those reads as absence, at that face's own seat —
  /// never as another face's reading, which is the class of failure
  /// this fusion no longer has.
  ///
  /// # What a caught Objective-C exception costs
  ///
  /// Vision raising across the FFI boundary is caught rather than
  /// allowed to abort the worker, and every call reads only state this
  /// call produced. A caught exception therefore costs THIS frame's
  /// faces (when the rectangles pass raised) or THIS frame's
  /// annotations (when the annotating pass, or the setting of its
  /// inputs, raised) — an empty `Vec` or an all-absent seat. What it
  /// never costs is another frame's data leaking into this one: the
  /// three requests are retained across calls, so their `results` may
  /// still describe an earlier frame after a raise, and nothing here
  /// reads them unless this call's own perform completed.
  pub fn detect<F: FaceDetection>(
    &self,
    jpeg_data: &[u8],
    options: &AppleVisionFaceOptions,
  ) -> Result<Vec<F>, AnalyzeError> {
    self.detect_on::<F>(ImageSource::Jpeg(jpeg_data), options)
  }

  /// Detects every face in already-decoded `pixels`, one record per
  /// face.
  ///
  /// [`detect`](Self::detect) reached without the encode. Both performs
  /// run against the same one image, so the fusion, the spine, the
  /// identity seating and every refusal documented there are unchanged.
  pub fn detect_pixels<F: FaceDetection>(
    &self,
    pixels: &PixelPlane<'_>,
    options: &AppleVisionFaceOptions,
  ) -> Result<Vec<F>, AnalyzeError> {
    self.detect_on::<F>(ImageSource::Plane(pixels), options)
  }

  /// The one two-perform fusion both doors reach.
  fn detect_on<F: FaceDetection>(
    &self,
    source: ImageSource<'_>,
    options: &AppleVisionFaceOptions,
  ) -> Result<Vec<F>, AnalyzeError> {
    with_image(source, |handler, data| {
      // Stage one — the detection spine.
      let rectangles = unsafe {
        [Retained::cast_unchecked::<VNRequest>(
          self.rectangles.clone(),
        )]
      };
      if perform(handler, data, &rectangles)? == Performed::Raised {
        // The rectangles request was NOT processed on this call, so
        // its `results` may still hold the previous frame's faces.
        // Reading them would make a stale spine — and a stale spine
        // passes every check downstream of it, because the annotating
        // passes preserve the uuids they are given and the whole
        // identity universe would be stale together. So: no spine, no
        // stage two, no faces.
        return Ok(Vec::new());
      }
      let spine = guard_vision_ffi("face rectangles", Vec::new(), || {
        self.collect_spine(options)
      });
      if spine.is_empty() {
        // Nothing to feed the annotating passes, so they are not
        // performed at all — which is also what keeps them from running
        // their OWN face detection on an empty input.
        return Ok(Vec::new());
      }

      // Stage two — the two annotating passes, each fed the spine's OWN
      // observations, so what comes back is those same faces enriched.
      // It runs only when BOTH requests took this call's inputs: a face
      // request performed with a nil `inputFaceObservations` runs its
      // own face detection and returns faces the spine never saw.
      let staged = if self.set_input_observations(&spine) {
        let enriched = unsafe {
          [
            Retained::cast_unchecked::<VNRequest>(self.quality.clone()),
            Retained::cast_unchecked::<VNRequest>(self.landmarks.clone()),
          ]
        };
        Some(perform(handler, data, &enriched))
      } else {
        None
      };
      // Read the annotating requests on exactly one condition: this
      // call's stage-two perform completed. Every other outcome —
      // inputs not set, a raise, an `NSError` — is absence, and it is
      // absence of BOTH passes because the perform they share is what
      // did not happen.
      //
      // A pass whose own accessors raise is a different condition, and
      // it is refused alone: the FFI barrier for that sits inside
      // `read_annotations`, one per reader. There is deliberately no
      // barrier here, around the pair.
      let annotations = match &staged {
        Some(Ok(Performed::Completed)) => self.read_annotations(&spine, options),
        _ => Annotations::absent(spine.len()),
      };
      // ALWAYS, on every path past `set_input_observations` — including
      // the one where it failed and the one where the perform errored.
      // The stage-two `Result` is held until after the clear so that
      // `?` cannot skip it. See `clear_input_observations` for what a
      // request left holding this frame's observations would cost the
      // next one.
      self.clear_input_observations();
      if let Some(performed) = staged {
        performed?;
      }

      Ok(emit::<F>(&spine, &annotations, options))
    })
  }

  /// The detection spine: one [`SpineFace`] per face the rectangles
  /// pass reported and the confidence gate kept, in Vision's order.
  /// This order is the engine's OWN — every face is emitted in it, and
  /// each annotating pass is re-ordered back into it — but it is not
  /// the order the annotating passes return their results in.
  ///
  /// The confidence gate simply DROPS a face here. Nothing downstream
  /// is a pool it could have absorbed from — the annotating passes only
  /// ever see the observations this function collected — so a dropped
  /// face releases nothing to its neighbours.
  ///
  /// Each kept face's `VNObservation.uuid` string is read here, once,
  /// and it is the only thing the annotating passes are matched on.
  /// Reading it through [`ffi_nsstring_to_smolstr`] bounds it like every
  /// other FFI string this crate accepts; a refusal there drops the face
  /// exactly as an unusable reading would, rather than seating it with
  /// an identity the engine cannot compare. A uuid string is 36 ASCII
  /// characters, orders of magnitude inside that bound, so this branch
  /// cannot fire against a sound Vision — it exists so that no path
  /// reaches [`spine_permutation`] holding a placeholder identity.
  ///
  /// The walk is bounded by [`MAX_VISION_RESULTS_PER_FRAME`], the same
  /// bound every other extractor in this crate uses, and the capacity
  /// is sized by the same bound so an FFI-reported length cannot drive
  /// the allocator.
  fn collect_spine(&self, options: &AppleVisionFaceOptions) -> Vec<SpineFace> {
    let Some(results) = (unsafe { self.rectangles.results() }) else {
      return Vec::new();
    };
    let rect_opts = options.rectangles();

    let mut spine = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      let Some(confidence) =
        sanitize_confidence(unsafe { obs.confidence() }, rect_opts.min_confidence())
      else {
        continue;
      };
      // The correspondence token. Read before anything else this face
      // would carry, so a face the engine could not identify never
      // becomes a seat an annotating pass could be asked to fill.
      let Some(uuid) = ffi_nsstring_to_smolstr(&unsafe { obs.uuid() }.UUIDString()) else {
        continue;
      };
      let rect = unsafe { obs.boundingBox() }.standardize();
      // `None` at every stage means the same thing to a consumer: no
      // usable angle. Vision omitting the `NSNumber?` entirely and
      // Vision reporting a non-finite reading both collapse to `None`
      // here, and neither is defaulted to `0.0` — that default is
      // exactly the collapse this seat exists to stop making.
      let roll = unsafe { obs.roll() }
        .map(|v| v.floatValue())
        .and_then(finite_f32);
      let yaw = unsafe { obs.yaw() }
        .map(|v| v.floatValue())
        .and_then(finite_f32);
      let pitch = unsafe { obs.pitch() }
        .map(|v| v.floatValue())
        .and_then(finite_f32);
      spine.push(SpineFace {
        observation: obs.clone(),
        uuid,
        rect,
        confidence,
        roll,
        yaw,
        pitch,
      });
    }
    spine
  }

  /// Hand the spine's own observations to both annotating requests,
  /// through Vision's `VNFaceObservationAccepting` protocol.
  ///
  /// This is half the attribution mechanism: a request given input
  /// observations processes exactly those faces — no others, and no
  /// detection of its own — and returns copies of them with its own
  /// field populated, each copy carrying the `VNObservation.uuid` of
  /// the input it came from.
  ///
  /// The other half is [`spine_permutation`], because the protocol
  /// preserves the SET of identities but not their order — a 3-face
  /// frame came back in the order it was given about half the time, and
  /// the measurement is recorded there. The handoff is what makes the
  /// identities comparable; the resolution is what makes a reading land
  /// on the right face.
  ///
  /// Returns whether BOTH requests took THIS call's observations —
  /// which is the precondition for performing them at all. A request
  /// performed with a nil `inputFaceObservations` runs its own face
  /// detection and returns faces the spine never saw, so "not set" can
  /// never be allowed to read as "set to nothing".
  ///
  /// Three guarded steps, because there are three distinct FFI calls
  /// that can raise and each failure has to be visible on its own. The
  /// two setters get a guard EACH: neither request may be reported set
  /// because the other one was.
  fn set_input_observations(&self, spine: &[SpineFace]) -> bool {
    // Building the array is FFI too — a retained clone per face and an
    // `NSArray` allocation — so it sits inside the barrier as well.
    let Some(inputs) = guard_vision_ffi("face inputs", None, || {
      let observations: Vec<Retained<VNFaceObservation>> =
        spine.iter().map(|face| face.observation.clone()).collect();
      Some(NSArray::from_retained_slice(&observations))
    }) else {
      return false;
    };
    let quality = guard_vision_ffi("face quality inputs", false, || {
      unsafe { self.quality.setInputFaceObservations(Some(&inputs)) };
      true
    });
    let landmarks = guard_vision_ffi("face landmark inputs", false, || {
      unsafe { self.landmarks.setInputFaceObservations(Some(&inputs)) };
      true
    });
    quality && landmarks
  }

  /// Reset both annotating requests to an **empty** `NSArray` — never
  /// to `None`.
  ///
  /// The distinction is the safety property of this whole design.
  /// `nil` means "no observations provided", and a face request
  /// performed that way runs its OWN face detection and returns faces
  /// the spine never saw — measured on this host, three of them. An
  /// EMPTY array is honoured instead as "process these zero faces", so
  /// a request that was somehow performed without fresh inputs yields
  /// nothing rather than fabricating a face nothing on the spine can
  /// own.
  ///
  /// Called on every path past `set_input_observations`, so a request
  /// never carries one call's observations into the next.
  ///
  /// The two clears are INDEPENDENTLY guarded, for the same reason the
  /// two setters are: one raising must not skip the other, or that
  /// request keeps this frame's observations into the next call. Each
  /// builds its own empty array inside its own guard, so not even the
  /// allocation is shared between them.
  fn clear_input_observations(&self) {
    guard_vision_ffi("face quality inputs", (), || {
      let empty = NSArray::<VNFaceObservation>::from_retained_slice(&[]);
      unsafe { self.quality.setInputFaceObservations(Some(&empty)) };
    });
    guard_vision_ffi("face landmark inputs", (), || {
      let empty = NSArray::<VNFaceObservation>::from_retained_slice(&[]);
      unsafe { self.landmarks.setInputFaceObservations(Some(&empty)) };
    });
  }

  /// Read both annotating passes BY IDENTITY, one reading per spine
  /// face.
  ///
  /// The spine's uuids are collected once and handed to both readers:
  /// they are the only thing either pass is matched on, and neither
  /// reader needs anything else the spine holds. `SmolStr` clones are
  /// cheap, and the list is what [`spine_permutation`] compares
  /// against.
  ///
  /// Each pass is read independently and refused independently, through
  /// the same four conditions ([`results_in_spine_order`]): no results
  /// array at all, a length that differs from the spine's, a returned
  /// observation whose uuid could not be read, or a uuid set that does
  /// not resolve one-to-one onto the spine's. Any of the four makes
  /// that ONE pass all-absent; the other still annotates.
  ///
  /// # One barrier per reader, not one around the pair
  ///
  /// A raising Objective-C accessor is a fifth refusal condition, and
  /// it is per pass for the same reason the other four are: these are
  /// two independent readings of two separately performed requests, so
  /// a raise while reducing landmarks costs this frame its keypoints,
  /// not its capture quality. Each reader therefore gets its OWN
  /// [`guard_vision_ffi`], falling back to [`absent_readings`] at the
  /// spine's length for that pass alone, under that pass's own detector
  /// label so the warning names which one raised. Only the collection
  /// of the uuids is outside both, and it is pure Rust — no FFI to
  /// guard.
  ///
  /// A single barrier around the pair would not merely be coarse, it
  /// would be lossy in one direction: a landmarks raise after a
  /// completed quality read would discard that finished vector and
  /// return both passes absent. An absent quality reading compares as
  /// `0.0` against `min_capture_quality` in [`emit`], whose default
  /// minimum is positive — so conflating the two would silently drop
  /// every face in the frame rather than just their keypoints.
  fn read_annotations(&self, spine: &[SpineFace], options: &AppleVisionFaceOptions) -> Annotations {
    let spine_uuids: Vec<SmolStr> = spine.iter().map(|face| face.uuid.clone()).collect();
    Annotations {
      quality: guard_vision_ffi(
        "face capture quality",
        absent_readings(spine_uuids.len()),
        || self.read_capture_quality(&spine_uuids),
      ),
      keypoints: guard_vision_ffi("face landmarks", absent_readings(spine_uuids.len()), || {
        self.read_keypoints(&spine_uuids, options)
      }),
    }
  }

  /// The capture-quality pass, reduced to one `Option<f32>` per spine
  /// face. `None` at a seat is a reading Vision did not usably provide
  /// — see [`sanitize_capture_quality`].
  fn read_capture_quality(&self, spine_uuids: &[SmolStr]) -> Vec<Option<f32>> {
    let Some(ordered) = results_in_spine_order(
      spine_uuids,
      unsafe { self.quality.results() },
      "face capture quality",
    ) else {
      return absent_readings(spine_uuids.len());
    };
    // `ordered` is exactly `spine_uuids.len()` long and already in
    // spine order, so one push per element seats every reading at its
    // own face without an index in sight.
    let mut readings = Vec::with_capacity(ordered.len());
    for obs in &ordered {
      readings.push(sanitize_capture_quality(
        unsafe { obs.faceCaptureQuality() }.map(|q| q.floatValue()),
      ));
    }
    readings
  }

  /// The landmarks pass, reduced to one `Option<FaceKeypoints>` per
  /// spine face.
  ///
  /// The two per-frame budgets are the same ones the full landmarker
  /// uses: an emission budget on points actually kept, and an attempt
  /// budget charged for every point walked, so a corrupted observation
  /// set cannot drive unbounded FFI traversal on the failure path. A
  /// refusal of any kind — no landmark set, a confidence below the
  /// gate, a contour that could not be read end to end, or a budget
  /// already spent — is simply that face's `None`, at that face's own
  /// seat.
  ///
  /// The walk runs in SPINE order, which is why
  /// [`results_in_spine_order`] permutes before the reduction rather
  /// than after it. Both budgets are consumed in walk order, so under
  /// budget pressure it is the LATE faces in the walk that lose their
  /// reduction. Walking Vision's return order would make "which face
  /// goes unannotated" depend on an order that changes run to run on
  /// the same image; walking the spine's own order makes it depend only
  /// on the image, so the same frame yields the same faces every time.
  fn read_keypoints(
    &self,
    spine_uuids: &[SmolStr],
    options: &AppleVisionFaceOptions,
  ) -> Vec<Option<FaceKeypoints>> {
    let Some(ordered) = results_in_spine_order(
      spine_uuids,
      unsafe { self.landmarks.results() },
      "face landmarks",
    ) else {
      return absent_readings(spine_uuids.len());
    };
    let opts = options.keypoints();

    let mut points_remaining: usize = MAX_FACE_LANDMARK_POINTS_PER_FRAME;
    let mut attempts: usize = 0;
    let mut readings = Vec::with_capacity(ordered.len());
    for obs in &ordered {
      readings.push('reduce: {
        if points_remaining == 0 || attempts >= MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME {
          break 'reduce None;
        }
        // The RETURNED observation's own box, not the spine's copy of
        // it: the landmark points are normalized within the box of the
        // observation that carries them, so they must be projected out
        // of that one. The uuid says the two boxes describe the same
        // face; nothing here requires them to be bit-identical, which
        // leaves a future Vision free to refine a box it re-examined.
        let face_rect_vision = unsafe { obs.boundingBox() }.standardize();
        let Some(landmarks) = (unsafe { obs.landmarks() }) else {
          break 'reduce None;
        };
        if sanitize_confidence(unsafe { landmarks.confidence() }, opts.min_confidence()).is_none() {
          break 'reduce None;
        }
        let mut tentative_remaining = points_remaining;
        match reduce_to_keypoints(
          &landmarks,
          face_rect_vision,
          &mut tentative_remaining,
          &mut attempts,
        ) {
          Some(keypoints) => {
            points_remaining = tentative_remaining;
            Some(keypoints)
          }
          None => None,
        }
      });
    }
    readings
  }
}

/// Emit the spine as the consumer's records.
#[cfg(target_vendor = "apple")]
fn emit<F: FaceDetection>(
  spine: &[SpineFace],
  annotations: &Annotations,
  options: &AppleVisionFaceOptions,
) -> Vec<F> {
  let capture_opts = options.capture();
  // Both annotation vectors are exactly `spine.len()` long — a refused
  // pass is `absent_readings(spine.len())`, an accepted one is one push
  // per element of a list `results_in_spine_order` built at that same
  // length — and both are in SPINE order, so `i` is this face's own
  // seat in each. See `Annotations`.
  let mut faces = Vec::with_capacity(spine.len());
  for (i, face) in spine.iter().enumerate() {
    let capture_quality = annotations.quality[i];
    // The threshold gate compares an unmeasured (`None`) face as `0.0`,
    // so it fails any positive minimum exactly as a measured-and-zero
    // face would. Only the COMPARISON substitutes `0.0` — the `Option`
    // itself, not this local, is what reaches `try_new` below, so a
    // face that clears a `min_capture_quality == 0.0` gate while
    // unmeasured still arrives at the contract seat as `None`.
    if capture_quality.unwrap_or(0.0) < capture_opts.min_capture_quality() {
      continue;
    }
    // The ONLY `B::try_new` on a face's path. A vocabulary that refuses
    // this box drops this face and nothing else.
    let Some(bbox) = vision_rect_to_bbox::<F::BoundingBox>(face.rect) else {
      continue;
    };
    if let Ok(built) = F::try_new(
      bbox,
      face.confidence,
      capture_quality,
      face.roll,
      face.yaw,
      face.pitch,
      annotations.keypoints[i],
    ) {
      faces.push(built);
    }
  }
  faces
}

/// Reduce one Vision landmark set to the five canonical keypoints, or
/// `None` when any of the five is not derivable — which now includes
/// any contour that could not be read END TO END, whether because the
/// frame's landmark budget could not cover the walk or because a point
/// Vision reported was not finite. Each of the five is an aggregate
/// over a whole contour, so a prefix read and a punctured read alike
/// would yield a confident wrong point; the reduction refuses the
/// whole face instead.
///
/// `points_remaining` is decremented by the points actually consumed;
/// `attempts` is charged by [`landmark_region_points_complete`] one
/// unit for every region VISITED and again for every point walked. A
/// refused contour leaves the point budget untouched and has spent its
/// visit unit — reading far enough to discover it does not fit is work
/// the frame did.
#[cfg(target_vendor = "apple")]
fn reduce_to_keypoints(
  landmarks: &VNFaceLandmarks2D,
  face_bbox_vision: CGRect,
  points_remaining: &mut usize,
  attempts: &mut usize,
) -> Option<FaceKeypoints> {
  // A region read, charged against both budgets. `None` propagates:
  // the frame could not afford a complete walk of that contour.
  let mut read = |region: Option<Retained<VNFaceLandmarkRegion2D>>| -> Option<Vec<(f32, f32)>> {
    let points =
      landmark_region_points_complete(region, face_bbox_vision, *points_remaining, attempts)?;
    *points_remaining = points_remaining.saturating_sub(points.len());
    Some(points)
  };

  // Pupil first: it IS the eye centre, one point, no averaging.
  // The eye contour is the fallback, and its centroid is the centre.
  // An empty read here is a genuinely absent region — a budget that
  // could not cover the walk has already propagated `None`, so the
  // fallback fires only on a complete read that found nothing.
  let mut left = read(unsafe { landmarks.leftPupil() })?;
  if left.is_empty() {
    left = read(unsafe { landmarks.leftEye() })?;
  }
  let left_eye = centroid(&left)?;
  let mut right = read(unsafe { landmarks.rightPupil() })?;
  if right.is_empty() {
    right = read(unsafe { landmarks.rightEye() })?;
  }
  let right_eye = centroid(&right)?;

  // The nose crest runs from between the eyes down to the tip, so the
  // crest point farthest from the eye midpoint IS the tip — no
  // dependence on Vision's point ordering. `nose` (the nostril
  // contour) is the fallback under the same rule.
  let eye_midpoint = (
    (left_eye.0 + right_eye.0) / 2.0,
    (left_eye.1 + right_eye.1) / 2.0,
  );
  let mut nose_points = read(unsafe { landmarks.noseCrest() })?;
  if nose_points.is_empty() {
    nose_points = read(unsafe { landmarks.nose() })?;
  }
  let nose_tip = farthest_from(&nose_points, eye_midpoint)?;

  // The lip contour's x-extremes are the mouth corners. Ties break on
  // `y` so a perfectly vertical (fully rolled) mouth still yields two
  // distinct, deterministically ordered corners.
  let mut lip_points = read(unsafe { landmarks.outerLips() })?;
  if lip_points.is_empty() {
    lip_points = read(unsafe { landmarks.innerLips() })?;
  }
  if lip_points.len() < 2 {
    return None;
  }
  let (mouth_left, mouth_right) = mouth_corners(&lip_points)?;

  Some(FaceKeypoints::new(
    left_eye,
    right_eye,
    nose_tip,
    mouth_left,
    mouth_right,
  ))
}

/// The x-extremes of a lip contour, ties broken on `y`: the minimum-x
/// point is the left corner, the maximum-x point the right. `None`
/// only for a list shorter than two points.
#[cfg(target_vendor = "apple")]
pub(crate) fn mouth_corners(points: &[(f32, f32)]) -> Option<((f32, f32), (f32, f32))> {
  if points.len() < 2 {
    return None;
  }
  let key = |p: &&(f32, f32)| (p.0, p.1);
  let left = *points.iter().min_by(|a, b| {
    let (ax, ay) = key(a);
    let (bx, by) = key(b);
    ax.total_cmp(&bx).then(ay.total_cmp(&by))
  })?;
  let right = *points.iter().max_by(|a, b| {
    let (ax, ay) = key(a);
    let (bx, by) = key(b);
    ax.total_cmp(&bx).then(ay.total_cmp(&by))
  })?;
  Some((left, right))
}

/// The arithmetic mean of a non-empty point list, or `None` when the
/// list is empty. Every point is already finite and inside the unit
/// square, so the mean is too.
#[cfg(target_vendor = "apple")]
pub(crate) fn centroid(points: &[(f32, f32)]) -> Option<(f32, f32)> {
  if points.is_empty() {
    return None;
  }
  let n = points.len() as f32;
  let (sum_x, sum_y) = points
    .iter()
    .fold((0.0_f32, 0.0_f32), |(sx, sy), (x, y)| (sx + x, sy + y));
  Some((sum_x / n, sum_y / n))
}

/// The point of `points` farthest (Euclidean) from `origin`, or `None`
/// when the list is empty. Ties break on the earlier point, which is
/// deterministic because the input order is Vision's own.
#[cfg(target_vendor = "apple")]
pub(crate) fn farthest_from(points: &[(f32, f32)], origin: (f32, f32)) -> Option<(f32, f32)> {
  points
    .iter()
    .copied()
    .map(|p| {
      let dx = p.0 - origin.0;
      let dy = p.1 - origin.1;
      (dx * dx + dy * dy, p)
    })
    .reduce(|best, next| if next.0 > best.0 { next } else { best })
    .map(|(_, p)| p)
}

/// Sanitise a raw face captureQuality reading from Vision.
///
/// `None` at every stage means the same thing to a consumer: no usable
/// quality reading for this observation. Vision omitting the
/// `NSNumber?` entirely (`raw = None`) and Vision reporting a
/// non-finite value (`NaN` / `±Inf`) both collapse to `None` here, and
/// neither is defaulted to `0.0` — that default is exactly the
/// collapse this function exists to stop making (see
/// [`FaceDetection`]). `Some(v)` is a real, finite measurement,
/// including a genuine `Some(0.0)` — Vision measured this capture and
/// found it terrible, which is not the same claim as never having
/// measured it.
#[cfg(target_vendor = "apple")]
#[inline]
pub(crate) fn sanitize_capture_quality(raw: Option<f32>) -> Option<f32> {
  raw.and_then(finite_f32)
}

/// One face on the detection spine: the observation Vision reported,
/// held so it can be handed BACK to the two annotating requests, plus
/// the readings the rectangles pass alone carries.
///
/// `observation` is what the annotating requests are given; `uuid` is
/// how what they give back is placed. They are two halves of one
/// mechanism — the handoff makes the identities comparable, and the
/// uuid is the identity itself, the token Vision copies onto every
/// enriched observation it derives from this one. `uuid` is what
/// [`spine_permutation`] resolves against, and it is the ONLY thing
/// either pass is matched on: not the box, not the array position.
///
/// `rect` is this face's box, standardized once here, and it is the box
/// the caller receives. The box the annotating passes return is used
/// only where the reading needs it — the landmark points are normalized
/// within it — and is never compared, so a Vision that refined a box it
/// re-examined would still annotate.
///
/// `confidence` is a plain `f32`: a face whose confidence failed its
/// gate never became a `SpineFace` at all. There is no pool for a
/// dropped face to absorb from, so dropping it early costs no other
/// face anything.
#[cfg(target_vendor = "apple")]
struct SpineFace {
  observation: Retained<VNFaceObservation>,
  uuid: SmolStr,
  rect: CGRect,
  confidence: f32,
  roll: Option<f32>,
  yaw: Option<f32>,
  pitch: Option<f32>,
}

/// What the two annotating passes read, one seat per spine face.
///
/// Both vectors are always exactly the spine's length, on every path:
/// a refused pass yields [`absent_readings`] at that length, and an
/// accepted one yields exactly one push per element of the spine-ordered
/// observation list, which [`results_in_spine_order`] builds at the
/// spine's length. So a face's index in the spine is its seat in each,
/// and [`emit`] may index both by it.
#[cfg(target_vendor = "apple")]
struct Annotations {
  quality: Vec<Option<f32>>,
  keypoints: Vec<Option<FaceKeypoints>>,
}

#[cfg(target_vendor = "apple")]
impl Annotations {
  /// Both passes absent, for a spine of `len` faces — what a frame
  /// gets when the annotating perform did not complete, or raised.
  fn absent(len: usize) -> Self {
    Self {
      quality: absent_readings(len),
      keypoints: absent_readings(len),
    }
  }
}

/// `len` absent readings: what every face gets from a pass that did not
/// come back, or whose results did not correspond to the spine.
///
/// Absence is per pass and total — a pass the engine could not place is
/// refused for EVERY face rather than for the ones it could not place,
/// because a pass whose correspondence is broken cannot say which faces
/// it got right.
///
/// Built without requiring `T: Clone`, so a payload need not be
/// cloneable to be absent.
#[cfg(target_vendor = "apple")]
fn absent_readings<T>(len: usize) -> Vec<Option<T>> {
  core::iter::repeat_with(|| None).take(len).collect()
}

/// Resolve one annotating pass's observation identities to spine
/// positions.
///
/// Returns the permutation `p` where `p[i]` is the index, within the
/// pass's returned observations, of the observation belonging to spine
/// face `i` — or `None` when the correspondence is not a bijection.
///
/// # Why a permutation and not an identity map
///
/// Vision's `VNFaceObservationAccepting` protocol preserves the SET of
/// observation identities across the handoff, but NOT their order —
/// and the order it does return varies run to run on one unchanged
/// image, within a single process.
///
/// Two independent 30-run measurements on this host, over the same
/// 3-face frame. The returned uuid set matched the spine 30/30 in both.
/// The returned ORDER matched the spine in only 14/30 and 15/30 of the
/// capture-quality runs, and 12/30 and 15/30 of the landmarks runs;
/// every one of the six permutations of three elements was observed,
/// three-cycles included, so this is nondeterminism rather than a fixed
/// re-ordering a positional read could have been written around.
///
/// Reading either pass by array position would therefore have
/// mis-attributed about half of all multi-face frames, silently — every
/// face carrying a real, plausible reading computed for one of its
/// neighbours. That is why the correspondence is keyed and verified
/// here rather than assumed at the call site.
///
/// # Why the result is a bijection
///
/// Three checks, and each closes one way the correspondence could fail:
///
/// - **Equal lengths.** A pass that returned a different number of
///   observations than it was given cannot be placed at all: a short
///   list cannot say which faces it dropped, and a long one contains at
///   least one observation the spine never handed over.
/// - **Distinct results.** Building the index refuses a repeated key,
///   so no two spine faces can resolve to one observation through the
///   pass's own duplication.
/// - **Total, exclusive lookup.** Every spine uuid must be found, and
///   each one CLAIMS its result — the lookup removes it — so a spine
///   uuid repeated by a broken caller finds the identity already taken
///   rather than seating one reading on two faces.
///
/// Equal lengths, plus an injective total map from spine positions into
/// result positions, is a bijection: `p` is a genuine permutation of
/// `0..n`, every returned observation is used exactly once, and every
/// spine face is seated exactly once.
///
/// Refusing is always safe. The caller turns a `None` into absence for
/// every face on that pass, which costs readings; seating a reading on
/// the wrong face costs correctness, and nothing downstream could tell.
#[cfg(target_vendor = "apple")]
pub(crate) fn spine_permutation(
  spine_uuids: &[SmolStr],
  result_uuids: &[SmolStr],
) -> Option<Vec<usize>> {
  if spine_uuids.len() != result_uuids.len() {
    return None;
  }
  // Sized from a length already proven equal to the spine's, so no
  // FFI-reported number reaches the allocator through here.
  let mut unclaimed: HashMap<&str, usize> = HashMap::with_capacity(result_uuids.len());
  for (index, uuid) in result_uuids.iter().enumerate() {
    if unclaimed.insert(uuid.as_str(), index).is_some() {
      return None;
    }
  }
  let mut permutation = Vec::with_capacity(spine_uuids.len());
  for uuid in spine_uuids {
    permutation.push(unclaimed.remove(uuid.as_str())?);
  }
  Some(permutation)
}

/// The four conditions an annotating pass's results must satisfy to be
/// read against the spine, and the pass's observations re-ordered into
/// spine order once they are. `None` refuses the pass, which makes it
/// all-absent for every face; the other pass is unaffected.
///
/// 1. There is a results array at all.
/// 2. It has exactly one observation per spine face. Checked BEFORE any
///    `Vec` is built from the results, so an FFI-reported length can
///    never size an allocation — and checked again against the number
///    of observations the walk actually enumerated, because fast
///    enumeration is not obliged to agree with the count.
/// 3. Every returned observation's uuid string is readable, within the
///    same bound every FFI string in this crate is held to.
/// 4. Those uuids resolve one-to-one onto the spine's
///    ([`spine_permutation`]).
///
/// Conditions 3 and 4 are what turn "trust the order" into "do not need
/// the order": the returned order is wrong about half the time on a
/// real multi-face frame, and the uuid is what says which face a
/// reading is for regardless. The observation's `boundingBox` is NOT
/// compared. It was, in an earlier draft of this design, on the premise
/// that the returned copies carry bit-identical boxes — which measured
/// true, but is a premise about Vision's implementation rather than
/// about its contract. The uuid is the identity token; a box check on
/// top of it is redundant when Vision returns the box unchanged, and
/// wrongly fail-closed if a future Vision refines a box it re-examined.
#[cfg(target_vendor = "apple")]
fn results_in_spine_order(
  spine_uuids: &[SmolStr],
  results: Option<Retained<NSArray<VNFaceObservation>>>,
  pass: &'static str,
) -> Option<Vec<Retained<VNFaceObservation>>> {
  #[cfg(not(feature = "tracing"))]
  let _ = pass;

  let results = results?;
  if results.len() != spine_uuids.len() {
    #[cfg(feature = "tracing")]
    tracing::warn!(
      pass,
      spine = spine_uuids.len(),
      returned = results.len(),
      "an annotating face pass returned a different number of observations than it was given; \
       annotating nothing from it",
    );
    return None;
  }

  // The count check above is the cheap first refusal, not a bound on
  // the walk. `NSArray::iter()` is Objective-C fast enumeration, which
  // is not obliged to agree with `count`: a malformed array can report
  // the spine's length and then hand over more elements than that, and
  // the bijection check below cannot refuse the pass until the loop has
  // already pushed and reallocated its way through however many the
  // enumeration cared to yield. So the walk is bounded independently of
  // the count it was told — one element past the spine, which is enough
  // to SEE an over-enumeration while capping both the work and the
  // allocation at `limit + 1`. The length check after the loop is what
  // turns that sighting into a refusal, and the same check catches an
  // under-enumeration, whose length lands below `limit`.
  let limit = spine_uuids.len();
  let mut result_uuids: Vec<SmolStr> = Vec::with_capacity(limit);
  let mut returned: Vec<Retained<VNFaceObservation>> = Vec::with_capacity(limit);
  for obs in results.iter().take(limit + 1) {
    let Some(uuid) = ffi_nsstring_to_smolstr(&unsafe { obs.uuid() }.UUIDString()) else {
      #[cfg(feature = "tracing")]
      tracing::warn!(
        pass,
        "an annotating face pass returned an observation whose uuid could not be read; annotating \
         nothing from it",
      );
      return None;
    };
    result_uuids.push(uuid);
    returned.push(obs);
  }
  if result_uuids.len() != limit {
    // `walked` saturates at `limit + 1`, where the walk stops; over the
    // bound it reads "at least one more than the spine", not the array's
    // true enumeration length, which is exactly what we declined to find
    // out.
    #[cfg(feature = "tracing")]
    tracing::warn!(
      pass,
      spine = limit,
      walked = result_uuids.len(),
      "an annotating face pass enumerated a different number of observations than its own count \
       reported; annotating nothing from it",
    );
    return None;
  }

  let Some(permutation) = spine_permutation(spine_uuids, &result_uuids) else {
    #[cfg(feature = "tracing")]
    tracing::warn!(
      pass,
      "an annotating face pass returned observation identities that do not correspond one-to-one \
       to the spine's; annotating nothing from it",
    );
    return None;
  };

  // `permutation` is a permutation of `0..spine_uuids.len()`, so every
  // index is in bounds and every observation is placed exactly once.
  Some(
    permutation
      .into_iter()
      .map(|index| returned[index].clone())
      .collect(),
  )
}

/// Non-macOS stub for [`FaceDetector`].
#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct FaceDetector;

#[cfg(not(target_vendor = "apple"))]
impl FaceDetector {
  /// Constructs a non-macOS stub detector. The options are ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionFaceOptions) -> Self {
    Self
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn detect<F: FaceDetection>(
    &self,
    _jpeg_data: &[u8],
    _options: &AppleVisionFaceOptions,
  ) -> Result<Vec<F>, AnalyzeError> {
    crate::error::unsupported()
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn detect_pixels<F: FaceDetection>(
    &self,
    _pixels: &PixelPlane<'_>,
    _options: &AppleVisionFaceOptions,
  ) -> Result<Vec<F>, AnalyzeError> {
    crate::error::unsupported()
  }
}
