//! Barcode detection: its own entry point, its own request, its own
//! trait.

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2_vision::*;

#[cfg(target_vendor = "apple")]
use crate::ffi::{
  ImageSource, MAX_VISION_RESULTS_PER_FRAME, ffi_nsstring_to_smolstr, guard_native,
  guard_vision_ffi, run_requests, sanitize_confidence, vision_rect_to_bbox,
};
use crate::{AnalyzeError, AppleVisionBarcodeOptions, BoundingBox, PixelPlane};

/// One decoded barcode.
///
/// `symbology` is Apple's raw `VNBarcodeSymbology` string, not a typed
/// vocabulary. The box comes **last**, unlike every other boxed
/// detection in this crate.
pub trait BarcodeDetection: Sized {
  /// Why a barcode was refused.
  type Error;
  /// The geometry type this detection is built from.
  type BoundingBox: BoundingBox;

  /// Builds a barcode detection.
  fn try_new(
    payload: &str,
    symbology: &str,
    confidence: f32,
    bbox: Self::BoundingBox,
  ) -> Result<Self, Self::Error>;
}

/// Apple Vision barcode detection — one per worker thread.
///
/// Owns exactly one Vision request; constructing a
/// [`BarcodeDetector`] loads no other model and
/// [`detect`](BarcodeDetector::detect) performs only the barcode
/// request.
///
/// The retained `VNRequest` carries per-call state across
/// `performRequests` / `results()`, so a detector is not safe to share
/// across threads; build one per worker.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct BarcodeDetector {
  request: Retained<VNDetectBarcodesRequest>,
}

#[cfg(target_vendor = "apple")]
impl BarcodeDetector {
  /// Creates a detector holding the barcode request at its pinned
  /// revision.
  ///
  /// `_options` is unused: Apple bakes no knob this crate exposes into
  /// the request object, so every gate is read per call.
  ///
  /// # Errors
  ///
  /// Building a Vision request loads a model, and a model load is where
  /// Apple's stack raises instead of returning: on a host whose Neural
  /// Engine is denied it throws, and a throw that crosses into Rust
  /// unguarded takes the process down. This refuses with
  /// [`AnalyzeErrorKind::Environment`](crate::AnalyzeErrorKind::Environment)
  /// instead — the constructor is where a whole entry point can still
  /// be declined, before any frame has been handed to it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionBarcodeOptions) -> Result<Self, AnalyzeError> {
    let request = guard_native("BarcodeDetector::new", || unsafe {
      let request = VNDetectBarcodesRequest::new();
      request.setRevision(VNDetectBarcodesRequestRevision4);
      request
    })?;
    Ok(Self { request })
  }

  /// Logs the pinned revision of the barcode request.
  ///
  /// A revision drift changes which symbologies decode **silently** —
  /// same API, different payloads.
  #[cfg(feature = "tracing")]
  pub fn log_request_revisions(&self) {
    unsafe {
      tracing::info!(
        barcodes_rev = self.request.revision(),
        "initialized pinned Apple Vision request revisions"
      );
    }
  }

  /// Decodes every barcode in `jpeg_data`.
  ///
  /// A barcode is dropped when Vision reports no string payload, when
  /// the payload or symbology exceeds the FFI string ceiling, when the
  /// payload is shorter than
  /// [`min_payload_len`](AppleVisionBarcodeOptions::min_payload_len),
  /// or when its box does not intersect the unit square. An `Err`
  /// means no detection happened at all.
  pub fn detect<B: BarcodeDetection>(
    &self,
    jpeg_data: &[u8],
    options: &AppleVisionBarcodeOptions,
  ) -> Result<Vec<B>, AnalyzeError> {
    self.detect_on::<B>(ImageSource::Jpeg(jpeg_data), options)
  }

  /// Decodes every barcode in already-decoded `pixels`.
  ///
  /// [`detect`](Self::detect) reached without the encode: same request,
  /// same options, same refusals, same output.
  pub fn detect_pixels<B: BarcodeDetection>(
    &self,
    pixels: &PixelPlane<'_>,
    options: &AppleVisionBarcodeOptions,
  ) -> Result<Vec<B>, AnalyzeError> {
    self.detect_on::<B>(ImageSource::Plane(pixels), options)
  }

  /// The one detection body both doors reach.
  fn detect_on<B: BarcodeDetection>(
    &self,
    source: ImageSource<'_>,
    options: &AppleVisionBarcodeOptions,
  ) -> Result<Vec<B>, AnalyzeError> {
    let requests = unsafe { [Retained::cast_unchecked::<VNRequest>(self.request.clone())] };
    run_requests(source, &requests, Vec::new(), || {
      guard_vision_ffi("barcodes", Vec::new(), || self.extract::<B>(options))
    })
  }

  fn extract<B: BarcodeDetection>(&self, opts: &AppleVisionBarcodeOptions) -> Vec<B> {
    let Some(results) = (unsafe { self.request.results() }) else {
      return Vec::new();
    };

    let mut barcodes = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      let Some(confidence) =
        sanitize_confidence(unsafe { obs.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      if let Some(payload) = unsafe { obs.payloadStringValue() } {
        // Bound the payload + symbology at MAX_FFI_STRING_BYTES.
        let Some(s) = ffi_nsstring_to_smolstr(&payload) else {
          continue;
        };
        if s.len() >= opts.min_payload_len()
          && let Some(bbox) = vision_rect_to_bbox(unsafe { obs.boundingBox() }.standardize())
        {
          let raw_sym = unsafe { obs.symbology() };
          let Some(symbology) = ffi_nsstring_to_smolstr(&raw_sym) else {
            continue;
          };
          if let Ok(barcode) = B::try_new(&s, &symbology, confidence, bbox) {
            barcodes.push(barcode);
          }
        }
      }
    }
    barcodes
  }
}

/// Non-macOS stub for [`BarcodeDetector`].
#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct BarcodeDetector;

#[cfg(not(target_vendor = "apple"))]
impl BarcodeDetector {
  /// Constructs a non-macOS stub detector. The options are ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  ///
  /// # Errors
  ///
  /// Never off Apple: there is no Vision framework to raise, so the
  /// constructor cannot fail. The `Result` is the Apple signature kept
  /// whole, so a caller writes `?` once and compiles on every host.
  pub fn new(_options: &AppleVisionBarcodeOptions) -> Result<Self, AnalyzeError> {
    Ok(Self)
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn detect<B: BarcodeDetection>(
    &self,
    _jpeg_data: &[u8],
    _options: &AppleVisionBarcodeOptions,
  ) -> Result<Vec<B>, AnalyzeError> {
    crate::error::unsupported()
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn detect_pixels<B: BarcodeDetection>(
    &self,
    _pixels: &PixelPlane<'_>,
    _options: &AppleVisionBarcodeOptions,
  ) -> Result<Vec<B>, AnalyzeError> {
    crate::error::unsupported()
  }
}
