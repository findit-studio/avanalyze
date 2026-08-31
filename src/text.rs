//! Text recognition: its own entry point, its own request, its own
//! trait.

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2_vision::*;

#[cfg(target_vendor = "apple")]
use crate::ffi::{
  MAX_VISION_RESULTS_PER_FRAME, ffi_nsstring_to_smolstr, guard_vision_ffi, run_requests,
  sanitize_confidence, vision_rect_to_bbox,
};
use crate::{AnalyzeError, AppleVisionTextOptions, BoundingBox};

/// Hard ceiling on candidate strings per text-recognition
/// observation. Apple's
/// `VNRecognizedTextObservation::topCandidates(_:)` documents an
/// upper limit of 10 — requesting more violates the Objective-C
/// API contract and can surface as a framework exception or
/// undefined behaviour across OS versions, so we clamp to 10 here
/// even though realistic workloads ask for 1-3 candidates.
#[cfg(target_vendor = "apple")]
const MAX_TEXT_CANDIDATES_PER_OBSERVATION: usize = 10;

/// Hard ceiling on the total text detections emitted per frame.
/// 256 caps the adversarial 4096 × MAX_TEXT_CANDIDATES_PER_OBSERVATION
/// product without restricting real text-rich-document workloads.
#[cfg(target_vendor = "apple")]
const MAX_TOTAL_TEXT_DETECTIONS_PER_FRAME: usize = 256;

/// One recognised text run.
///
/// One Vision observation is one text region on the page, and it can
/// yield several *candidate* readings of that same region — Apple's
/// `topCandidates(_:)` list, best first. Every candidate re-uses the
/// observation's box, so `bbox` alone cannot tell two readings of one
/// region apart from two regions that happen to overlap.
///
/// `observation` and `rank` are what tell them apart. `observation` is
/// the index of the Vision observation within this call's result
/// array; `rank` is the candidate's position within that observation's
/// candidate list, `0` being Vision's best reading. The pair is the
/// engine's provenance for the run: candidates sharing an
/// `observation` are competing readings of ONE region, and `rank`
/// orders them. A consumer that keeps only `rank == 0` gets one row
/// per region; one that keeps them all can rank, diff, or vote across
/// readings without inventing an identity of its own.
///
/// Both are per call, not global: they index this call's results and
/// mean nothing across calls.
pub trait TextDetection: Sized {
  /// Why a text detection was refused.
  type Error;
  /// The geometry type this detection is built from.
  type BoundingBox: BoundingBox;

  /// Builds a text detection.
  ///
  /// Note the argument order: the box comes after the reading and
  /// before the provenance pair, not last as it does for
  /// [`BarcodeDetection`](crate::BarcodeDetection). `observation`
  /// indexes the Vision observation and `rank` indexes the candidate
  /// within it (`0` = Vision's best); both are zero-based and scoped
  /// to a single call.
  fn try_new(
    text: &str,
    confidence: f32,
    bbox: Self::BoundingBox,
    observation: usize,
    rank: usize,
  ) -> Result<Self, Self::Error>;
}

/// Apple Vision text recognition — one per worker thread.
///
/// Owns exactly one Vision request. Constructing a
/// [`TextRecognizer`] loads no face, pose, mask or classification
/// model, and [`recognize`](TextRecognizer::recognize) performs only
/// the text request.
///
/// The retained `VNRequest` carries per-call state across
/// `performRequests` / `results()`, so a recognizer is not safe to
/// share across threads; build one per worker.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct TextRecognizer {
  request: Retained<VNRecognizeTextRequest>,
}

#[cfg(target_vendor = "apple")]
impl TextRecognizer {
  /// Creates a recognizer holding the text request at its pinned
  /// revision.
  ///
  /// `_options` is unused: Apple bakes no knob this crate exposes into
  /// the request object, so every gate is read per call. The parameter
  /// stays so the constructor keeps the shape every other entry point
  /// uses, and so a future baked knob does not move the signature.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionTextOptions) -> Self {
    let request = unsafe {
      let request = VNRecognizeTextRequest::new();
      request.setRevision(VNRecognizeTextRequestRevision3);
      request
    };
    Self { request }
  }

  /// Logs the pinned revision of the text request.
  ///
  /// A revision drift changes recognition semantics **silently** —
  /// same API, different strings.
  #[cfg(feature = "tracing")]
  pub fn log_request_revisions(&self) {
    unsafe {
      tracing::info!(
        text_rev = self.request.revision(),
        "initialized pinned Apple Vision request revisions"
      );
    }
  }

  /// Recognises text in `jpeg_data`, best candidate first within each
  /// observation.
  ///
  /// Returns one `T` per surviving candidate. A candidate is dropped
  /// when its string exceeds the FFI string ceiling, falls below
  /// [`min_text_len`](AppleVisionTextOptions::min_text_len), carries a
  /// non-finite confidence, or sits on a box the unit square rejects.
  /// An `Err` means no recognition happened at all.
  pub fn recognize<T: TextDetection>(
    &self,
    jpeg_data: &[u8],
    options: &AppleVisionTextOptions,
  ) -> Result<Vec<T>, AnalyzeError> {
    let requests = unsafe { [Retained::cast_unchecked::<VNRequest>(self.request.clone())] };
    run_requests(jpeg_data, &requests, Vec::new(), || {
      guard_vision_ffi("text", Vec::new(), || self.extract::<T>(options))
    })
  }

  fn extract<T: TextDetection>(&self, options: &AppleVisionTextOptions) -> Vec<T> {
    let Some(results) = self.request.results() else {
      return Vec::new();
    };

    // Per-frame total cap on emitted text detections — bounds the
    // outer × inner candidate product.
    let mut text_detections = Vec::with_capacity(MAX_TOTAL_TEXT_DETECTIONS_PER_FRAME);
    // Bound the requested candidate count to the hard per-observation cap
    // — Apple's topCandidates allocates an NSArray sized to the argument.
    let candidate_cap = options
      .max_candidates_per_observation()
      .min(MAX_TEXT_CANDIDATES_PER_OBSERVATION);
    'outer: for (observation, obs) in results
      .iter()
      .take(MAX_VISION_RESULTS_PER_FRAME)
      .enumerate()
    {
      if text_detections.len() >= MAX_TOTAL_TEXT_DETECTIONS_PER_FRAME {
        break;
      }
      let candidates = obs.topCandidates(candidate_cap);
      for (rank, candidate) in candidates.iter().take(candidate_cap).enumerate() {
        if text_detections.len() >= MAX_TOTAL_TEXT_DETECTIONS_PER_FRAME {
          break 'outer;
        }
        // Bound the candidate string at MAX_FFI_STRING_BYTES before
        // routing through `to_smolstr` so a corrupted/adversarial
        // NSString length cannot drive the allocator into the abort
        // path.
        let raw_string = candidate.string();
        let Some(text) = ffi_nsstring_to_smolstr(&raw_string) else {
          continue;
        };
        if text.len() < options.min_text_len() {
          continue;
        }
        let Some(confidence) = sanitize_confidence(candidate.confidence(), 0.0) else {
          continue;
        };
        if let Some(bbox) = vision_rect_to_bbox(unsafe { obs.boundingBox() }.standardize())
          && let Ok(detection) = T::try_new(&text, confidence, bbox, observation, rank)
        {
          text_detections.push(detection);
        }
      }
    }
    text_detections
  }
}

/// Non-macOS stub for [`TextRecognizer`].
#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct TextRecognizer;

#[cfg(not(target_vendor = "apple"))]
impl TextRecognizer {
  /// Constructs a non-macOS stub recognizer. The options are ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionTextOptions) -> Self {
    Self
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn recognize<T: TextDetection>(
    &self,
    _jpeg_data: &[u8],
    _options: &AppleVisionTextOptions,
  ) -> Result<Vec<T>, AnalyzeError> {
    crate::error::unsupported()
  }
}
