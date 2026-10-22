//! The crate's system floor: the oldest systems every entry point is
//! built on, and the one check each constructor makes before anything
//! else.
//!
//! The floor is written once, in [`floor!`]. The public
//! [`SYSTEM_FLOOR`] text, the `available!` check and the version the
//! test seam compares against each expand from that list, so the three
//! cannot disagree.
//!
//! # What the floor covers
//!
//! A refusal below the floor needs a process to refuse in. The loader
//! starts a process only where it can load every library the binary links
//! strongly and bind every symbol it imports strongly, before `main` and
//! so before any floor check, and a binary may be launched on any system
//! its deployment target allows, far below the floor. So everything this
//! crate links or imports strongly, as built with objc2's default features,
//! exists on every system from macOS 10.13, iOS 11 and tvOS 11 on, the
//! first row's two frameworks being the newest: from there up to the floor
//! a process starts, and every constructor refuses. The census, every
//! framework the crate links and every Vision class, global, constant and
//! selector it references, with the release that brought it
//! (`API_AVAILABLE` in the SDK's headers; visionOS follows iOS), against
//! the floor of macOS 15, iOS 18, tvOS 18 and visionOS 2:
//!
//! | what | how it is reached | introduced (macOS, iOS, tvOS) |
//! |---|---|---|
//! | Vision, and CoreML, which `objc2-vision`'s dependencies link beside it | linked strongly, by those crates' own `#[link]`, which no library can weaken | 10.13, 11, 11 (visionOS 1): a process that links this crate starts only where both exist. rustc's own defaults deploy lower on three targets (x86_64 macOS 10.12, iOS 10, tvOS 10); a binary deployed there that must start on the systems the two predate links both weakly, `-C link-arg=-Wl,-weak_framework,Vision` and the same for `CoreML`, which ld64 and lld both honour beside the strong link |
//! | every other framework and library linked (Foundation, CoreFoundation, CoreGraphics, CoreVideo, CoreImage, CoreMedia, AVFoundation, ImageIO, Metal, libobjc, libc++, libSystem, libiconv), and every symbol imported from them | linked and imported strongly | each older than the oldest systems rustc supports: macOS 10.12, iOS 10, tvOS 10, visionOS 1 |
//! | every class below | by name at run time (objc2's `objc_getClass`, its default), after the floor check | 10.13 to 15.0, each at or below the floor |
//! | every class below, where a consumer enables `objc2/unstable-static-class` | objc2 then emits each class as a static `OBJC_CLASS_$_…` reference, a strong link-time import: by its own documentation, "OS version < Class introduced version: Dynamic linker error on program startup" (objc2 0.6.5, `src/macros/mod.rs` 33–47, 509–513) | the class's own release: such a binary does not launch below it, so [`BelowFloor`](crate::AnalyzeErrorKind::BelowFloor) cannot answer there. The consumer's trade-off for opting into an unstable feature that changes linking, which this crate does not enable and no library can undo |
//! | every selector below | by name at run time (`sel_registerName`), sent to an instance of its class, after the floor check | 10.13 to 15.0, each at or below the floor |
//! | the four `…JointsGroupNameAll` keys | by name, after the floor check (`ffi::vision_string`); never imported | body and hand 11.0, 14, 14; animal and 3-D body 14.0, 17, 17 |
//! | the revisions (`VN…RequestRevisionN`) and enumerations (`VNRequestTextRecognitionLevel`, `VNChirality`, `VNHumanBodyPose3DObservationHeightEstimation`) | integers fixed at compile time | no symbol |
//!
//! The classes:
//!
//! | class | introduced |
//! |---|---|
//! | `VNRequest`, `VNImageBasedRequest`, `VNSequenceRequestHandler`, `VNObservation`, `VNDetectedObjectObservation`, `VNFaceObservation`, `VNFaceLandmarks2D`, `VNFaceLandmarkRegion2D`, `VNClassificationObservation`, `VNBarcodeObservation`, `VNHorizonObservation`, `VNRectangleObservation`, `VNPixelBufferObservation`, `VNDetectBarcodesRequest`, `VNDetectFaceLandmarksRequest`, `VNDetectFaceRectanglesRequest`, `VNDetectHorizonRequest`; the protocol `VNFaceObservationAccepting` | 10.13, 11, 11 |
//! | `VNRecognizedObjectObservation` | 10.14, 12, 12 |
//! | `VNRecognizedTextObservation`, `VNRecognizedText`, `VNSaliencyImageObservation`, `VNClassifyImageRequest`, `VNDetectFaceCaptureQualityRequest`, `VNDetectHumanRectanglesRequest`, `VNGenerateAttentionBasedSaliencyImageRequest`, `VNGenerateObjectnessBasedSaliencyImageRequest`, `VNRecognizeAnimalsRequest`, `VNRecognizeTextRequest` | 10.15, 13, 13 |
//! | `VNRecognizedPoint`, `VNDetectedPoint`, `VNPoint`, `VNRecognizedPointsObservation`, `VNHumanBodyPoseObservation`, `VNHumanHandPoseObservation`, `VNStatefulRequest`, `VNDetectHumanBodyPoseRequest`, `VNDetectHumanHandPoseRequest` | 11.0, 14, 14 |
//! | `VNHumanObservation`, `VNDetectDocumentSegmentationRequest`, `VNGeneratePersonSegmentationRequest` | 12.0, 15, 15 |
//! | `VNPoint3D`, `VNRecognizedPoint3D`, `VNHumanBodyRecognizedPoint3D`, `VNAnimalBodyPoseObservation`, `VNHumanBodyPose3DObservation`, `VNInstanceMaskObservation`, `VNDetectAnimalBodyPoseRequest`, `VNDetectHumanBodyPose3DRequest`, `VNGeneratePersonInstanceMaskRequest` | 14.0, 17, 17 |
//! | `VNImageAestheticsScoresObservation`, `VNCalculateImageAestheticsScoresRequest` | 15.0, 18, 18 (visionOS 2): the floor |
//!
//! The selectors, each with its class's release unless it names its own:
//!
//! | sent to | selectors | introduced |
//! |---|---|---|
//! | every request class | `+new`; `results` | its class's |
//! | `VNRequest` | `revision`, `setRevision:`, `+supportedRevisions` | 10.14, 12, 12 |
//! | `VNSequenceRequestHandler` | `performRequests:onCGImage:error:`, `performRequests:onImageData:error:` | 10.13, 11, 11 |
//! | `VNObservation` | `uuid`, `confidence` | 10.13, 11, 11 |
//! | `VNDetectedObjectObservation`, `VNFaceObservation` | `boundingBox`, `landmarks` | 10.13, 11, 11 |
//! | `VNFaceObservation` | `roll`, `yaw` | 10.14, 12, 12 |
//! | `VNFaceObservation` | `faceCaptureQuality` | 10.15, 13, 13 |
//! | `VNFaceObservation` | `pitch` | 12.0, 15, 15 |
//! | `VNFaceLandmarks2D` | `allPoints`, `faceContour`, `leftEye`, `rightEye`, `leftEyebrow`, `rightEyebrow`, `nose`, `noseCrest`, `medianLine`, `outerLips`, `innerLips`, `leftPupil`, `rightPupil`, `confidence` | 10.13, 11, 11 |
//! | `VNFaceLandmarkRegion2D` | `pointCount`, `normalizedPoints` | 10.13, 11, 11 |
//! | `VNClassificationObservation` | `identifier` | 10.13, 11, 11 |
//! | `VNBarcodeObservation` | `payloadStringValue`, `symbology` | 10.13, 11, 11 |
//! | `VNHorizonObservation` | `angle` | 10.13, 11, 11 |
//! | `VNRectangleObservation` | `topLeft`, `topRight`, `bottomLeft`, `bottomRight` | 10.13, 11, 11 |
//! | `VNPixelBufferObservation` | `pixelBuffer` | 10.13, 11, 11 |
//! | `VNRecognizedObjectObservation` | `labels` | 10.14, 12, 12 |
//! | `VNRecognizedTextObservation`, `VNRecognizedText` | `topCandidates:`, `string`, `confidence` | 10.15, 13, 13 |
//! | `VNSaliencyImageObservation` | `salientObjects` | 10.15, 13, 13 |
//! | `VNRecognizeTextRequest` | `setRecognitionLevel:`, `setRecognitionLanguages:`, `setUsesLanguageCorrection:`, `setCustomWords:`, `setMinimumTextHeight:` | 10.15, 13, 13 |
//! | `VNRecognizeTextRequest` | `supportedRecognitionLanguagesAndReturnError:` | 12.0, 15, 15 |
//! | `VNRecognizeTextRequest` | `setAutomaticallyDetectsLanguage:` | 13.0, 16, 16 |
//! | `VNPoint`, `VNRecognizedPoint` | `x`, `y`, `identifier`, `confidence` | 11.0, 14, 14 |
//! | `VNRecognizedPointsObservation` | `recognizedPointsForJointsGroupName:error:` | 11.0, 14, 14 |
//! | `VNDetectHumanHandPoseRequest` | `setMaximumHandCount:` | 11.0, 14, 14 |
//! | `VNHumanHandPoseObservation` | `chirality` | 12.0, 15, 15 |
//! | `VNDetectHumanRectanglesRequest` | `setUpperBodyOnly:` | 12.0, 15, 15 |
//! | `VNHumanBodyPose3DObservation`, `VNAnimalBodyPoseObservation` | `recognizedPointsForJointsGroupName:error:`, `bodyHeight`, `heightEstimation` | 14.0, 17, 17 |
//! | `VNRecognizedPoint3D` | `identifier` | 14.0, 17, 17 |
//! | `VNInstanceMaskObservation` | `allInstances`, `generateMaskForInstances:error:` | 14.0, 17, 17 |
//! | `VNImageAestheticsScoresObservation` | `overallScore`, `isUtility` | 15.0, 18, 18 |
//!
//! No Vision symbol is imported, strong or weak:
//! `no_vision_symbol_is_imported` reads the test binary's own symbol
//! table, which holds every entry point's extraction, with `nm -m`.

#[cfg(target_vendor = "apple")]
use core::sync::atomic::{AtomicU8, Ordering};

#[cfg(target_vendor = "apple")]
use crate::{AnalyzeError, AnalyzeErrorKind};

/// Hands `$then!` the crate's floor: each Apple system's oldest version,
/// as `system major minor`.
///
/// This is the only place the floor is written. Raising it means editing
/// this list and nothing else.
macro_rules! floor {
  ($then:ident) => {
    $then! { macos 15 0, ios 18 0, tvos 18 0, visionos 2 0 }
  };
}

/// The floor as the text a refusal names.
macro_rules! floor_text {
  (
    macos $macos:tt $macos_minor:tt,
    ios $ios:tt $ios_minor:tt,
    tvos $tvos:tt $tvos_minor:tt,
    visionos $visionos:tt $visionos_minor:tt
  ) => {
    concat!(
      "macOS ",
      $macos,
      ".",
      $macos_minor,
      ", iOS ",
      $ios,
      ".",
      $ios_minor,
      ", tvOS ",
      $tvos,
      ".",
      $tvos_minor,
      " or visionOS ",
      $visionos,
      ".",
      $visionos_minor
    )
  };
}

/// The floor as the literals `objc2::available!` takes.
#[cfg(target_vendor = "apple")]
macro_rules! floor_available {
  (
    macos $macos:tt $macos_minor:tt,
    ios $ios:tt $ios_minor:tt,
    tvos $tvos:tt $tvos_minor:tt,
    visionos $visionos:tt $visionos_minor:tt
  ) => {
    objc2::available!(
      macos = $macos.$macos_minor,
      ios = $ios.$ios_minor,
      tvos = $tvos.$tvos_minor,
      visionos = $visionos.$visionos_minor
    )
  };
}

/// The floor as the `(major, minor)` this target's system version is
/// compared against. A system outside the four is never at it.
#[cfg(all(test, target_vendor = "apple"))]
macro_rules! floor_version {
  (
    macos $macos:tt $macos_minor:tt,
    ios $ios:tt $ios_minor:tt,
    tvos $tvos:tt $tvos_minor:tt,
    visionos $visionos:tt $visionos_minor:tt
  ) => {
    if cfg!(target_os = "macos") {
      ($macos, $macos_minor)
    } else if cfg!(target_os = "ios") {
      ($ios, $ios_minor)
    } else if cfg!(target_os = "tvos") {
      ($tvos, $tvos_minor)
    } else if cfg!(target_os = "visionos") {
      ($visionos, $visionos_minor)
    } else {
      (u16::MAX, u16::MAX)
    }
  };
}

/// The oldest systems this crate is built on: macOS 15, iOS 18, tvOS 18
/// and visionOS 2.
///
/// Below them, every entry point's constructor refuses with
/// [`AnalyzeErrorKind::BelowFloor`](crate::AnalyzeErrorKind::BelowFloor).
/// It refuses before it sends a message or resolves a Vision class, and
/// its message names this floor.
///
/// The floor is where the newest Vision class an entry point builds
/// arrives. [`VisionAnalyzer`](crate::VisionAnalyzer) always builds
/// `VNCalculateImageAestheticsScoresRequest`, which
/// `VNCalculateImageAestheticsScoresRequest.h` declares
/// `API_AVAILABLE(macos(15.0), ios(18.0), tvos(18.0), visionos(2.0))`. Every
/// other class the nine constructors build is older. One floor for the
/// crate, rather than one per entry point, means a consumer has one
/// question to ask and the crate has one answer to keep true.
///
/// Below a class's own floor, resolving it panics. One check, made first,
/// therefore makes every message a constructor sends safe to send.
pub const SYSTEM_FLOOR: &str = floor!(floor_text);

/// The floor's comparable version on this target, for the test seam.
#[cfg(all(test, target_vendor = "apple"))]
pub(crate) const FLOOR_VERSION: (u16, u16) = floor!(floor_version);

/// Whether this system is at or above [`SYSTEM_FLOOR`].
///
/// objc2 is asked once (it reads the version as Apple's own availability
/// check does, Mac Catalyst and the simulators included), and the answer
/// is kept for the life of the process. Under the test seam the
/// simulated version is compared against the floor instead, and nothing
/// is kept.
#[cfg(target_vendor = "apple")]
fn system_at_floor() -> bool {
  #[cfg(test)]
  if let Some(version) = seam::simulated() {
    return version >= FLOOR_VERSION;
  }
  /// Not asked yet.
  const UNKNOWN: u8 = 0;
  /// At or above the floor.
  const AT: u8 = 1;
  /// Below it.
  const BELOW: u8 = 2;
  // Relaxed is enough: two threads that race to ask both get the same
  // answer and store the same value.
  static ANSWER: AtomicU8 = AtomicU8::new(UNKNOWN);
  match ANSWER.load(Ordering::Relaxed) {
    AT => true,
    BELOW => false,
    _ => {
      let at = floor!(floor_available);
      ANSWER.store(if at { AT } else { BELOW }, Ordering::Relaxed);
      at
    }
  }
}

/// Refuses `site` below [`SYSTEM_FLOOR`] with
/// [`AnalyzeErrorKind::BelowFloor`], naming the entry point and the
/// floor.
///
/// It is the first statement of every entry point's constructor, before
/// options are checked and before any class is resolved.
#[cfg(target_vendor = "apple")]
pub(crate) fn require_floor(site: &'static str) -> Result<(), AnalyzeError> {
  if system_at_floor() {
    Ok(())
  } else {
    Err(AnalyzeError::new(
      AnalyzeErrorKind::BelowFloor,
      format!("{site}: this system is older than {SYSTEM_FLOOR}, the oldest avanalyze is built on"),
    ))
  }
}

/// The test seam: a simulated system version for the calling thread.
///
/// It is thread-local, so tests that run side by side never see each
/// other's simulation. It is scoped, so a simulation ends when its guard
/// drops, a panicking test's included.
#[cfg(all(test, target_vendor = "apple"))]
pub(crate) mod seam {
  use std::cell::Cell;

  thread_local! {
    static SIMULATED: Cell<Option<(u16, u16)>> = const { Cell::new(None) };
  }

  /// The calling thread's simulated `(major, minor)`, if any.
  pub(crate) fn simulated() -> Option<(u16, u16)> {
    SIMULATED.with(Cell::get)
  }

  /// A simulation in force. Dropping it restores what was simulated
  /// before.
  #[must_use = "the simulation ends when the guard drops"]
  pub(crate) struct Simulated {
    previous: Option<(u16, u16)>,
  }

  /// Simulates `major.minor` as the calling thread's system version until
  /// the guard drops.
  pub(crate) fn simulate(major: u16, minor: u16) -> Simulated {
    Simulated {
      previous: SIMULATED.with(|cell| cell.replace(Some((major, minor)))),
    }
  }

  impl Drop for Simulated {
    fn drop(&mut self) {
      SIMULATED.with(|cell| cell.set(self.previous));
    }
  }
}
