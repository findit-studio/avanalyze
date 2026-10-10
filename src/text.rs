//! Text recognition: its own entry point, its own request, its own
//! trait.

#[cfg(target_vendor = "apple")]
use objc2::{ClassType, msg_send, rc::Retained};
#[cfg(target_vendor = "apple")]
use objc2_foundation::{NSArray, NSError, NSIndexSet, NSNotFound, NSString};
#[cfg(target_vendor = "apple")]
use objc2_vision::*;
#[cfg(target_vendor = "apple")]
use smol_str::SmolStr;

use crate::{AnalyzeError, AppleVisionTextOptions, BoundingBox, PixelPlane};
#[cfg(target_vendor = "apple")]
use crate::{
  AnalyzeErrorKind, TextRecognitionLevel,
  ffi::{
    ImageSource, MAX_VISION_RESULTS_PER_FRAME, ffi_nsstring_to_smolstr, guard_native,
    guard_vision_ffi, run_requests, sanitize_confidence, vision_rect_to_bbox,
  },
};

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

/// Hard ceiling on the language tags read off a
/// `supportedRecognitionLanguages` answer. Vision lists 33 at revision 3
/// on macOS 27; the bound is there because the array's length is
/// reported across the FFI, like every other array this crate walks.
#[cfg(target_vendor = "apple")]
const MAX_LISTED_LANGUAGES: usize = 256;

/// Hard ceiling on the revisions read off the text request class's own
/// `supportedRevisions` index set, which names three on macOS 27.
#[cfg(target_vendor = "apple")]
const MAX_LISTED_REVISIONS: usize = 64;

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
/// The request is configured once, by the options handed to
/// [`new`](TextRecognizer::new): its revision, recognition level,
/// language roster, language detection, language correction, custom
/// words and minimum text height follow the recognizer, not the call.
/// Each call reads only the three gates on what comes back — see
/// [`AppleVisionTextOptions`] for which option is which.
///
/// The retained `VNRequest` carries per-call state across
/// `performRequests` / `results()`, so a recognizer is not safe to
/// share across threads; build one per worker.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct TextRecognizer {
  request: Retained<VNRecognizeTextRequest>,
  /// Read back from the request above inside the same guarded closure
  /// that pinned it, so the public reader sends no message of its own.
  revision: usize,
}

#[cfg(target_vendor = "apple")]
impl TextRecognizer {
  /// Creates a recognizer holding the text request `options` describe.
  ///
  /// Seven options are the request's own and are set on it here, once:
  /// [`revision`](AppleVisionTextOptions::revision),
  /// [`recognition_level`](AppleVisionTextOptions::recognition_level),
  /// [`languages`](AppleVisionTextOptions::languages),
  /// [`language_correction`](AppleVisionTextOptions::language_correction),
  /// [`custom_words`](AppleVisionTextOptions::custom_words),
  /// [`min_text_height`](AppleVisionTextOptions::min_text_height) and
  /// [`detect_language`](AppleVisionTextOptions::detect_language). The
  /// `options` a later [`recognize`](Self::recognize) is handed cannot
  /// move them; only the per-call gates are read there.
  ///
  /// # Errors
  ///
  /// [`AnalyzeErrorKind::InvalidOptions`] when `options` ask for what
  /// the request cannot be, each refusal naming the value refused:
  ///
  /// - a [`revision`](AppleVisionTextOptions::revision) the text
  ///   request class does not list in its `supportedRevisions` on this
  ///   host — the message names the ones it does;
  /// - [`detect_language`](AppleVisionTextOptions::detect_language)
  ///   where detection cannot act: on a system without it (before
  ///   macOS 13, iOS 16, tvOS 16), or at a revision below 3, where Apple
  ///   documents it as a no-op. Vision would build either request and
  ///   read on its English-only default roster. The message names what
  ///   would be taken: `detect_language = false`, or revision 3 where
  ///   this host implements it;
  /// - a tag in [`languages`](AppleVisionTextOptions::languages) the
  ///   request does not list in `supportedRecognitionLanguages` for that
  ///   revision and recognition level — the message names the ones it
  ///   does. The match is exact: Vision lists `ja-JP`, so `ja` is
  ///   refused;
  /// - a [`min_confidence`](AppleVisionTextOptions::min_confidence) or
  ///   [`min_text_height`](AppleVisionTextOptions::min_text_height)
  ///   outside `0..=1`, or not a number.
  ///
  /// [`AnalyzeErrorKind::RequestFailed`] when Vision answers an error
  /// instead of the language list a non-empty roster is checked against.
  ///
  /// [`AnalyzeErrorKind::Environment`] when Apple's stack raises.
  /// Building a Vision request loads a model, and a model load is where
  /// Apple's stack raises instead of returning: on a host whose Neural
  /// Engine is denied it throws, and a throw that crosses into Rust
  /// unguarded takes the process down. The constructor is where a whole
  /// entry point can still be declined, before any frame has been handed
  /// to it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(options: &AppleVisionTextOptions) -> Result<Self, AnalyzeError> {
    const SITE: &str = "TextRecognizer::new";
    check_fraction(SITE, "min_confidence", options.min_confidence())?;
    check_fraction(SITE, "min_text_height", options.min_text_height())?;
    // Every send the request's configuration makes — the class's
    // revision list, the language list, each setter, the read-back —
    // happens inside the one barrier, so a raise anywhere in it refuses
    // the constructor instead of crossing into Rust. The pool inside it
    // drains what those answers autoreleased, here rather than whenever
    // the calling thread's own pool does, if it has one: a caller that
    // builds a recognizer per picture would otherwise accumulate them.
    let (request, revision) =
      guard_native(SITE, || objc2::rc::autoreleasepool(|_| configure(options)))??;
    Ok(Self { request, revision })
  }

  /// Logs the revision of the text request.
  ///
  /// A revision drift changes recognition semantics **silently** —
  /// same API, different strings. [`revision`](Self::revision) is the
  /// reader this renders — the two never fall out of step because there
  /// is only one spelling of the revision itself.
  #[cfg(feature = "tracing")]
  pub fn log_request_revisions(&self) {
    tracing::info!(
      text_rev = self.revision,
      "initialized pinned Apple Vision request revisions"
    );
  }

  /// The revision of the text request, read back from the request
  /// object [`new`](Self::new) called `setRevision` on — never from the
  /// options it was handed, so this can never disagree with what
  /// [`log_request_revisions`](Self::log_request_revisions) would log
  /// or with what Vision runs. The read happened once, inside the
  /// constructor's own exception barrier; this returns the cached answer
  /// and sends no message.
  ///
  /// A single recognizer owns a single request, so there is no roster
  /// to name: unlike [`VisionAnalyzer::revisions`](crate::VisionAnalyzer::revisions)
  /// or [`FaceDetector::revisions`](crate::FaceDetector::revisions), one
  /// `usize` is the whole answer.
  ///
  /// ```ignore
  /// let recognizer = TextRecognizer::new(&AppleVisionTextOptions::new())?;
  /// assert_eq!(recognizer.revision(), 3);
  /// # Ok::<(), avanalyze::AnalyzeError>(())
  /// ```
  pub const fn revision(&self) -> usize {
    self.revision
  }

  /// The request [`new`](Self::new) configured, for the laws that read
  /// its properties back.
  #[cfg(test)]
  pub(crate) fn request(&self) -> &VNRecognizeTextRequest {
    &self.request
  }

  /// Recognises text in `jpeg_data`, best candidate first within each
  /// observation.
  ///
  /// Returns one `T` per surviving candidate. A candidate is dropped
  /// when its string exceeds the FFI string ceiling, falls below
  /// [`min_text_len`](AppleVisionTextOptions::min_text_len), carries a
  /// confidence outside `0..=1` or below
  /// [`min_confidence`](AppleVisionTextOptions::min_confidence), or
  /// sits on a box the unit square rejects. An `Err` means no
  /// recognition happened at all.
  ///
  /// `options` is read per call for those gates and for
  /// [`max_candidates_per_observation`](AppleVisionTextOptions::max_candidates_per_observation);
  /// everything else in it was set on the request by
  /// [`new`](Self::new) and follows the recognizer.
  ///
  /// # Errors
  ///
  /// [`AnalyzeErrorKind::InvalidOptions`] when `options` carry a
  /// [`min_confidence`](AppleVisionTextOptions::min_confidence) outside
  /// `0..=1`, or not a number, before the picture is looked at: a gate
  /// outside the range Vision scores in is refused rather than run.
  /// Otherwise the refusals every entry point shares: an input past the
  /// engine's ceilings, or an error from Vision, as
  /// [`AnalyzeErrorKind::RequestFailed`]; a raise, as
  /// [`AnalyzeErrorKind::Environment`].
  pub fn recognize<T: TextDetection>(
    &self,
    jpeg_data: &[u8],
    options: &AppleVisionTextOptions,
  ) -> Result<Vec<T>, AnalyzeError> {
    self.recognize_on::<T>(ImageSource::Jpeg(jpeg_data), options)
  }

  /// Recognises text in already-decoded `pixels`.
  ///
  /// [`recognize`](Self::recognize) reached without the encode: same
  /// request, same options, same refusals, same output.
  pub fn recognize_pixels<T: TextDetection>(
    &self,
    pixels: &PixelPlane<'_>,
    options: &AppleVisionTextOptions,
  ) -> Result<Vec<T>, AnalyzeError> {
    self.recognize_on::<T>(ImageSource::Plane(pixels), options)
  }

  /// The one recognition body both doors reach.
  fn recognize_on<T: TextDetection>(
    &self,
    source: ImageSource<'_>,
    options: &AppleVisionTextOptions,
  ) -> Result<Vec<T>, AnalyzeError> {
    check_fraction(
      "TextRecognizer::recognize",
      "min_confidence",
      options.min_confidence(),
    )?;
    let requests = unsafe { [Retained::cast_unchecked::<VNRequest>(self.request.clone())] };
    run_requests(source, &requests, Vec::new(), || {
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
        let Some(confidence) =
          sanitize_confidence(candidate.confidence(), options.min_confidence())
        else {
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

/// Builds the text request `options` describe, or names what in them it
/// cannot be.
///
/// The order is load-bearing. The revision is checked first, so an
/// unimplemented one is named as what it is, and language detection is
/// checked against it. The revision is set before the recognition level
/// and both before the roster is checked, because the language list is
/// the request's answer for its configuration at the time it is asked —
/// and both levels and every revision list different languages.
///
/// Every message it sends can raise, so it runs inside
/// [`guard_native`]: [`TextRecognizer::new`] is its only caller.
#[cfg(target_vendor = "apple")]
fn configure(
  options: &AppleVisionTextOptions,
) -> Result<(Retained<VNRecognizeTextRequest>, usize), AnalyzeError> {
  let wanted = options.revision();
  // `supportedRevisions` is a class property, and the binding declares
  // it once, on `VNRequest`: called through it, the question goes to the
  // base class, whose answer is not the text request's. The message is
  // sent to the text request's own class instead.
  let implemented: Option<Retained<NSIndexSet>> =
    unsafe { msg_send![VNRecognizeTextRequest::class(), supportedRevisions] };
  let implements = |revision: usize| {
    implemented
      .as_deref()
      .is_some_and(|revisions| revisions.containsIndex(revision))
  };
  if !implements(wanted) {
    let listed = implemented
      .as_deref()
      .map(listed_revisions)
      .unwrap_or_default();
    return Err(refusal(format!(
      "TextRecognizer::new: revision {wanted} is not one this host's text request implements; \
       it implements {}",
      listed.join(", ")
    )));
  }

  // Detection asked for where it cannot act is refused: Vision would
  // build the request either way and read on its English-only default
  // roster — the very reading this option exists to end, returned as a
  // success.
  if options.detect_language() {
    let instead = if implements(VNRecognizeTextRequestRevision3) {
      "set detect_language = false, or ask for revision 3, which this host implements"
    } else {
      "set detect_language = false and name the languages instead"
    };
    if !detection_available() {
      return Err(refusal(format!(
        "TextRecognizer::new: detect_language is true, and this system's text request cannot \
         detect a language — Apple added automatic language detection in macOS 13, iOS 16 and \
         tvOS 16; {instead}"
      )));
    }
    if wanted < VNRecognizeTextRequestRevision3 {
      return Err(refusal(format!(
        "TextRecognizer::new: detect_language is true, and revision {wanted} cannot detect a \
         language — Apple documents automatic language detection as a no-op before revision 3; \
         {instead}"
      )));
    }
  }

  let request = VNRecognizeTextRequest::new();
  unsafe { request.setRevision(wanted) };
  let level = options.recognition_level();
  request.setRecognitionLevel(vision_level(level));

  if !options.languages().is_empty() {
    let listed = listed_languages(&request, vision_level(level), wanted).map_err(|error| {
      // Through the bounded FFI-string helper, as `perform` reports an
      // NSError, so a pathological description cannot drive the
      // allocator into the abort path.
      let description = ffi_nsstring_to_smolstr(&error.localizedDescription())
        .unwrap_or_else(|| SmolStr::new_static("description elided"));
      AnalyzeError::new(
        AnalyzeErrorKind::RequestFailed,
        format!(
          "TextRecognizer::new: Vision could not list the languages its text request reads: \
             {description}"
        ),
      )
    })?;
    let readable: Vec<SmolStr> = listed
      .iter()
      .take(MAX_LISTED_LANGUAGES)
      .filter_map(|tag| ffi_nsstring_to_smolstr(&tag))
      .collect();
    if let Some(unread) = options
      .languages()
      .iter()
      .find(|language| !readable.iter().any(|tag| tag.as_str() == language.as_str()))
    {
      return Err(refusal(format!(
        "TextRecognizer::new: at revision {wanted} and the {} recognition level, the text \
         request does not read {unread:?}; it reads {}",
        level_name(level),
        readable.join(", ")
      )));
    }
    request.setRecognitionLanguages(&ns_strings(options.languages()));
  }

  request.setUsesLanguageCorrection(options.language_correction());
  // Never an empty list: a fresh request's `customWords` is nil, and the
  // default keeps it that way.
  if !options.custom_words().is_empty() {
    request.setCustomWords(&ns_strings(options.custom_words()));
  }
  request.setMinimumTextHeight(options.min_text_height());
  // Sent only where the property exists. Where it does not, a request
  // asking for detection was refused above, and `false` is what a fresh
  // request already holds.
  if detection_available() {
    request.setAutomaticallyDetectsLanguage(options.detect_language());
  }

  // Read back here, inside the barrier that already spans every send
  // this constructor makes, so the public reader has to send none —
  // and so the number comes from the request object rather than from
  // the options that asked for it.
  let revision = unsafe { request.revision() };
  Ok((request, revision))
}

/// Whether this system's text request has `automaticallyDetectsLanguage`.
///
/// The versions are the header's own `API_AVAILABLE` for the property —
/// macOS 13, iOS 16, tvOS 16 — and visionOS, which has had it from its
/// first release. Below them the selector does not exist: sending it
/// would raise.
#[cfg(target_vendor = "apple")]
fn detection_available() -> bool {
  objc2::available!(macos = 13.0, ios = 16.0, tvos = 16.0, visionos = 1.0)
}

/// The tags `request` lists for its `level` and `revision` —
/// `supportedRecognitionLanguages` — asked the way this system can be
/// asked.
///
/// The instance method is macOS 12, iOS 15 and tvOS 15. Below that, the
/// same question goes to the class method it replaced, which Apple
/// shipped from macOS 10.15 and deprecated at 12, so the roster check
/// runs on every system this crate builds for. No deployment target is
/// set for this crate, so its floor is the toolchain's: macOS 11 on
/// Apple silicon, 10.12 on Intel — both below the instance method.
#[cfg(target_vendor = "apple")]
fn listed_languages(
  request: &VNRecognizeTextRequest,
  level: VNRequestTextRecognitionLevel,
  revision: usize,
) -> Result<Retained<NSArray<NSString>>, Retained<NSError>> {
  if objc2::available!(macos = 12.0, ios = 15.0, tvos = 15.0, visionos = 1.0) {
    unsafe { request.supportedRecognitionLanguagesAndReturnError() }
  } else {
    languages_listed_by_class(level, revision)
  }
}

/// The class method's answer to which tags a text request at `level`
/// and `revision` lists — the source [`listed_languages`] asks below
/// macOS 12.
///
/// Declared by the binding on the text request class itself, so the
/// question reaches that class and not a superclass.
#[cfg(target_vendor = "apple")]
#[allow(deprecated)]
pub(crate) fn languages_listed_by_class(
  level: VNRequestTextRecognitionLevel,
  revision: usize,
) -> Result<Retained<NSArray<NSString>>, Retained<NSError>> {
  unsafe {
    VNRecognizeTextRequest::supportedRecognitionLanguagesForTextRecognitionLevel_revision_error(
      level, revision,
    )
  }
}

/// Vision's spelling of a recognition level.
#[cfg(target_vendor = "apple")]
const fn vision_level(level: TextRecognitionLevel) -> VNRequestTextRecognitionLevel {
  match level {
    TextRecognitionLevel::Accurate => VNRequestTextRecognitionLevel::Accurate,
    TextRecognitionLevel::Fast => VNRequestTextRecognitionLevel::Fast,
  }
}

/// A recognition level as a refusal names it — the spelling a config
/// uses.
#[cfg(target_vendor = "apple")]
const fn level_name(level: TextRecognitionLevel) -> &'static str {
  match level {
    TextRecognitionLevel::Accurate => "accurate",
    TextRecognitionLevel::Fast => "fast",
  }
}

/// The revisions an index set names, in order, bounded.
#[cfg(target_vendor = "apple")]
fn listed_revisions(revisions: &NSIndexSet) -> Vec<String> {
  // `NSNotFound` is `NSIntegerMax`; the index set answers it, as an
  // unsigned index, once there is no next member.
  let not_found = NSNotFound as usize;
  let mut listed = Vec::new();
  let mut index = revisions.firstIndex();
  while index != not_found && listed.len() < MAX_LISTED_REVISIONS {
    listed.push(index.to_string());
    index = revisions.indexGreaterThanIndex(index);
  }
  listed
}

/// `strings` as the `NSArray<NSString>` a request property takes.
#[cfg(target_vendor = "apple")]
fn ns_strings(strings: &[String]) -> Retained<NSArray<NSString>> {
  let strings: Vec<Retained<NSString>> = strings
    .iter()
    .map(|string| NSString::from_str(string))
    .collect();
  NSArray::from_retained_slice(&strings)
}

/// Refuses `value` unless it is a fraction: inside `0..=1`, and a
/// number at all.
#[cfg(target_vendor = "apple")]
fn check_fraction(site: &str, name: &str, value: f32) -> Result<(), AnalyzeError> {
  if (0.0..=1.0).contains(&value) {
    Ok(())
  } else {
    Err(refusal(format!("{site}: {name} {value} is outside 0..=1")))
  }
}

/// An [`AnalyzeErrorKind::InvalidOptions`] refusal carrying `message`.
#[cfg(target_vendor = "apple")]
fn refusal(message: String) -> AnalyzeError {
  AnalyzeError::new(AnalyzeErrorKind::InvalidOptions, message)
}

/// Non-macOS stub for [`TextRecognizer`].
#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct TextRecognizer;

#[cfg(not(target_vendor = "apple"))]
impl TextRecognizer {
  /// Constructs a non-macOS stub recognizer. The options are ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  ///
  /// # Errors
  ///
  /// Never off Apple: there is no Vision framework to raise, so the
  /// constructor cannot fail. The `Result` is the Apple signature kept
  /// whole, so a caller writes `?` once and compiles on every host.
  pub fn new(_options: &AppleVisionTextOptions) -> Result<Self, AnalyzeError> {
    Ok(Self)
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

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn recognize_pixels<T: TextDetection>(
    &self,
    _pixels: &PixelPlane<'_>,
    _options: &AppleVisionTextOptions,
  ) -> Result<Vec<T>, AnalyzeError> {
    crate::error::unsupported()
  }
}
