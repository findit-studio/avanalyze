//! The core one-pass analyzer: eight cheap detections, one batch.

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2_foundation::NSArray;
#[cfg(target_vendor = "apple")]
use objc2_vision::*;
#[cfg(target_vendor = "apple")]
use smol_str::{SmolStr, StrExt};

use crate::AnalyzeError;
#[cfg(target_vendor = "apple")]
use crate::{
  Aesthetics, Analysis, AnalyzeOptions, AppleVisionSaliencyOptions, Detection, Detections,
  DocumentSegment, HorizonInfo, SaliencyRegion, SubjectDetection,
  ffi::{
    MAX_VISION_RESULTS_PER_FRAME, effective_results_cap, ffi_nsstring_to_smolstr, finite_f32,
    guard_vision_ffi, run_requests, sanitize_confidence, vision_point_to_normalized,
    vision_rect_to_bbox,
  },
};

#[cfg(not(target_vendor = "apple"))]
use crate::{Analysis, AnalyzeOptions};

/// Hard ceiling on labels per recognised-animal observation.
#[cfg(target_vendor = "apple")]
const MAX_NESTED_LABELS_PER_OBSERVATION: usize = 32;

/// Hard ceiling on the total animal-subject rows emitted per frame.
/// Apple's animal recogniser returns a few species per frame at most;
/// 256 caps the adversarial 4096 × MAX_NESTED_LABELS_PER_OBSERVATION
/// product without restricting real workloads.
#[cfg(target_vendor = "apple")]
const MAX_TOTAL_ANIMAL_SUBJECTS_PER_FRAME: usize = 256;

/// Hard ceiling on saliency regions per frame.
#[cfg(target_vendor = "apple")]
const MAX_SALIENCY_REGIONS_PER_FRAME: usize = 64;

/// Apple Vision analyzer for the eight cheap detections — one per
/// worker thread.
///
/// Construct one [`VisionAnalyzer`] per worker thread via
/// [`VisionAnalyzer::new`]. The analyzer owns retained `VNRequest`
/// Objective-C objects that carry per-call state across
/// `performRequests` / `results()`, so they are *not* safe to share
/// across threads or clone. Construct one fresh analyzer per worker
/// rather than cloning a single shared instance — `Clone` is
/// intentionally not implemented to make that contract a compile-time
/// error.
///
/// It owns **eight** Vision requests and no others: classification,
/// human rectangles, animal recognition, both saliency passes, the
/// horizon, document segmentation, and aesthetics. Text, barcodes,
/// faces, landmarks, poses and masks are separate entry points and
/// their models are never loaded by this one.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct VisionAnalyzer {
  requests: VisionRequests,
}

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
struct VisionRequests {
  classify: Retained<VNClassifyImageRequest>,
  human_rectangles: Retained<VNDetectHumanRectanglesRequest>,
  animals: Retained<VNRecognizeAnimalsRequest>,
  attention_saliency: Retained<VNGenerateAttentionBasedSaliencyImageRequest>,
  objectness_saliency: Retained<VNGenerateObjectnessBasedSaliencyImageRequest>,
  horizon: Retained<VNDetectHorizonRequest>,
  document_segments: Retained<VNDetectDocumentSegmentationRequest>,
  aesthetics: Retained<VNCalculateImageAestheticsScoresRequest>,
}

#[cfg(target_vendor = "apple")]
impl VisionRequests {
  /// Builds the eight core requests at their pinned revisions.
  ///
  /// `_options` is unused: none of the eight has a knob Apple bakes
  /// into the request object, so every gate is read per call. The
  /// parameter stays so the constructor keeps the shape every other
  /// entry point uses, and so a future baked knob does not move the
  /// public signature.
  fn new(_options: &AnalyzeOptions) -> Self {
    unsafe {
      let classify = VNClassifyImageRequest::new();
      classify.setRevision(VNClassifyImageRequestRevision2);

      let human_rectangles = VNDetectHumanRectanglesRequest::new();
      human_rectangles.setUpperBodyOnly(false);
      human_rectangles.setRevision(VNDetectHumanRectanglesRequestRevision2);

      let animals = VNRecognizeAnimalsRequest::new();
      animals.setRevision(VNRecognizeAnimalsRequestRevision2);

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
        human_rectangles,
        animals,
        attention_saliency,
        objectness_saliency,
        horizon,
        document_segments,
        aesthetics,
      }
    }
  }

  /// The eight requests as one erased slice, in the order they are
  /// performed.
  fn as_slice(&self) -> [Retained<VNRequest>; 8] {
    unsafe {
      [
        Retained::cast_unchecked::<VNRequest>(self.classify.clone()),
        Retained::cast_unchecked::<VNRequest>(self.human_rectangles.clone()),
        Retained::cast_unchecked::<VNRequest>(self.animals.clone()),
        Retained::cast_unchecked::<VNRequest>(self.attention_saliency.clone()),
        Retained::cast_unchecked::<VNRequest>(self.objectness_saliency.clone()),
        Retained::cast_unchecked::<VNRequest>(self.horizon.clone()),
        Retained::cast_unchecked::<VNRequest>(self.document_segments.clone()),
        Retained::cast_unchecked::<VNRequest>(self.aesthetics.clone()),
      ]
    }
  }
}

#[cfg(target_vendor = "apple")]
impl VisionAnalyzer {
  /// Creates an analyzer holding the eight core Vision requests.
  ///
  /// Every knob is read per call by
  /// [`analyze_keyframe`](VisionAnalyzer::analyze_keyframe); the
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
        human_rectangles_rev = self.requests.human_rectangles.revision(),
        animals_rev = self.requests.animals.revision(),
        attention_saliency_rev = self.requests.attention_saliency.revision(),
        objectness_saliency_rev = self.requests.objectness_saliency.revision(),
        horizon_rev = self.requests.horizon.revision(),
        document_segments_rev = self.requests.document_segments.revision(),
        aesthetics_rev = self.requests.aesthetics.revision(),
        "initialized pinned Apple Vision request revisions"
      );
    }
  }

  /// Runs the eight core Vision requests against `jpeg_data` and
  /// gathers the detections into an [`Analysis`].
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
  pub fn analyze_keyframe<D: Detections>(
    &self,
    jpeg_data: &[u8],
    options: &AnalyzeOptions,
  ) -> Result<Analysis<D>, AnalyzeError> {
    run_requests(
      jpeg_data,
      &self.requests.as_slice(),
      Analysis::new(),
      || {
        // Per-detector Objective-C exception barrier. The batched
        // perform above runs every detector, but result extraction
        // re-enters Vision FFI per detector (`results()` + per-observation
        // accessors). Any one of those can raise an `NSException` that
        // `catch_unwind` cannot catch — so each `extract_*` is wrapped in
        // `objc2::exception::catch` (via `guard_vision_ffi`). A raising
        // detector contributes its empty fallback and the OTHER detectors'
        // results still land: the analysis degrades per detector, never
        // aborting the process.
        let mut analysis: Analysis<D> = Analysis::new();
        analysis
          .set_classifications(guard_vision_ffi("classify", Vec::new(), || {
            self.extract_classifications::<D>(options)
          }))
          .set_human_subjects(guard_vision_ffi("human_rectangles", Vec::new(), || {
            self.extract_human_subjects::<D>(options)
          }))
          .set_animal_subjects(guard_vision_ffi("animals", Vec::new(), || {
            self.extract_animal_subjects::<D>(options)
          }))
          .set_attention_saliency(guard_vision_ffi("attention_saliency", Vec::new(), || {
            self.extract_attention_saliency::<D>(options)
          }))
          .set_objectness_saliency(guard_vision_ffi("objectness_saliency", Vec::new(), || {
            self.extract_objectness_saliency::<D>(options)
          }))
          .set_document_segments(guard_vision_ffi("document_segments", Vec::new(), || {
            self.extract_document_segments::<D>(options)
          }))
          .set_horizon(guard_vision_ffi(
            "horizon",
            // The "no detection" sentinel — matches `extract_horizon`'s
            // own `empty`. `None` only if the vocabulary refuses even
            // that, which costs the slot rather than the frame.
            D::HorizonInfo::try_new(0.0, 0.0).ok(),
            || self.extract_horizon::<D>(options),
          ))
          .set_aesthetics(Some(guard_vision_ffi(
            "aesthetics",
            // The "no detection" sentinel — matches `extract_aesthetics`.
            D::Aesthetics::new(0.0, false),
            || self.extract_aesthetics::<D>(options),
          )));
        analysis
      },
    )
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

/// Non-macOS stub for [`VisionAnalyzer`]. Apple's Vision.framework is
/// only available on macOS, so on every other target the analyzer
/// always reports [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported)
/// rather than producing detections. The README promises the crate
/// compiles cleanly on non-macOS targets so downstream workspaces can
/// keep `avanalyze` in their dep tree unconditionally; this stub is
/// what makes that promise true.
#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct VisionAnalyzer;

#[cfg(not(target_vendor = "apple"))]
impl VisionAnalyzer {
  /// Constructs a non-macOS stub analyzer. The options are ignored —
  /// every `analyze_keyframe` call reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AnalyzeOptions) -> Self {
    Self
  }

  /// Non-macOS stub: Apple's Vision.framework is only available on
  /// macOS, so this always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  /// `_jpeg_data` is ignored.
  pub fn analyze_keyframe<D: crate::Detections>(
    &self,
    _jpeg_data: &[u8],
    _options: &AnalyzeOptions,
  ) -> Result<Analysis<D>, AnalyzeError> {
    crate::error::unsupported()
  }
}
