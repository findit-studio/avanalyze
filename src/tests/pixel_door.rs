//! The pixel door against the real frameworks: what Core Graphics makes
//! of each declared format, and whether Vision sees the same picture
//! through both doors.
//!
//! Two kinds of evidence, and they answer different questions.
//!
//! The round-trip tests are exact. They build the `CGImage` the door
//! builds and draw it back out into a known RGBA bitmap, so a wrong
//! `CGBitmapInfo` — a swapped red and blue, an alpha byte read as
//! colour, a stride that shifts every row — is a channel mismatch here
//! rather than a plausible-looking detection later. Nothing about them
//! is statistical.
//!
//! The parity tests are not exact, and cannot honestly be. The JPEG
//! door hands Vision a compressed stream that ImageIO decodes inside
//! the framework; the pixel door hands it pixels. Even when this file
//! decodes the very same fixture with the very same ImageIO, the two
//! paths differ in colour management and in what Vision's own
//! preprocessing does to each, so detections agree to a few thousandths
//! of a normalized coordinate rather than bit for bit. The tolerances
//! below are stated as what was measured on an Apple host, not as what
//! would be nice.

use core::{convert::Infallible, ffi::c_void};

use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_core_graphics::{
  CGBitmapContextCreate, CGColorRenderingIntent, CGColorSpace, CGContext, CGDataProvider, CGImage,
  CGImageAlphaInfo,
};

use crate::{
  AnalyzeOptions, AnimalPoser, AppleVisionAnimalPoseOptions, AppleVisionBarcodeOptions,
  AppleVisionBodyPoserOptions, AppleVisionFaceLandmarkOptions, AppleVisionFaceOptions,
  AppleVisionHandPoseOptions, AppleVisionPersonMaskerOptions, AppleVisionTextOptions,
  BarcodeDetector, BodyPoser, FaceDetector, FaceKeypoints, FaceLandmarker, HandPoser, PersonMasker,
  PixelFormat, PixelPlane, TextRecognizer, VisionAnalyzer, ffi::cg_image_from_plane,
  tests::reference::MediaSchema,
};

/// Three frontal, well-separated faces. Provenance is recorded in
/// `tests/fixtures/README.md`.
const CREW: &[u8] = include_bytes!("../../tests/fixtures/apollo11_crew.jpg");

/// The zero-face keyframe: no face, and the frame the 3-D body-pose
/// detector used to raise on.
const AIRPORT: &[u8] = include_bytes!("../../tests/fixtures/airport_keyframe.jpg");

// ----- a vocabulary just wide enough to read a face back out ----------------

/// A box that stores what it is given, so a test can compare two doors'
/// geometry. The engine's own guards are asserted elsewhere; this one
/// refuses nothing, on purpose.
#[derive(Debug, Clone, Copy)]
struct Bbox {
  x: f32,
  y: f32,
  width: f32,
  height: f32,
}

impl crate::BoundingBox for Bbox {
  type Error = Infallible;

  fn try_new(x: f32, y: f32, width: f32, height: f32) -> Result<Self, Self::Error> {
    Ok(Self {
      x,
      y,
      width,
      height,
    })
  }

  fn x(&self) -> f32 {
    self.x
  }

  fn y(&self) -> f32 {
    self.y
  }

  fn width(&self) -> f32 {
    self.width
  }

  fn height(&self) -> f32 {
    self.height
  }
}

#[derive(Debug, Clone, Copy)]
struct Face {
  bbox: Bbox,
  confidence: f32,
}

impl crate::FaceDetection for Face {
  type Error = Infallible;
  type BoundingBox = Bbox;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    _capture_quality: Option<f32>,
    _roll: Option<f32>,
    _yaw: Option<f32>,
    _pitch: Option<f32>,
    _keypoints: Option<FaceKeypoints>,
  ) -> Result<Self, Self::Error> {
    Ok(Self { bbox, confidence })
  }
}

// ----- Core Graphics helpers the tests share --------------------------------

/// Decode a JPEG the way Vision's own JPEG door does — through
/// ImageIO — into packed RGBA8, so the two doors are compared on one
/// decoder rather than two.
fn decode_rgba(jpeg: &'static [u8]) -> (u32, u32, Vec<u8>) {
  // SAFETY: `jpeg` is `'static` (an `include_bytes!` constant), so the
  // provider's borrow can never dangle; the release callback is `None`
  // because nothing is owned.
  let provider = unsafe {
    CGDataProvider::with_data(
      core::ptr::null_mut(),
      jpeg.as_ptr().cast::<c_void>(),
      jpeg.len(),
      None,
    )
  }
  .expect("a data provider over a static fixture");
  // SAFETY: `decode` is null, the documented alternative to a valid
  // pointer.
  let image = unsafe {
    CGImage::with_jpeg_data_provider(
      Some(&provider),
      core::ptr::null(),
      true,
      CGColorRenderingIntent::RenderingIntentDefault,
    )
  }
  .expect("the fixture is a decodable JPEG");
  let width = CGImage::width(Some(&image));
  let height = CGImage::height(Some(&image));
  let pixels = render_rgba(&image, width, height);
  (
    u32::try_from(width).expect("fixture width fits u32"),
    u32::try_from(height).expect("fixture height fits u32"),
    pixels,
  )
}

/// Draw `image` into a `width` × `height` RGBA8 bitmap whose fourth byte
/// is ignored, and hand back the bytes. This is the measuring stick the
/// format mappings are checked against.
fn render_rgba(image: &CGImage, width: usize, height: usize) -> Vec<u8> {
  let stride = width * 4;
  let mut pixels = vec![0u8; stride * height];
  let colour_space = CGColorSpace::new_device_rgb().expect("device RGB");
  // SAFETY: `pixels` is a live allocation of exactly `stride * height`
  // bytes and outlives the context, which is dropped at the end of this
  // function.
  let context = unsafe {
    CGBitmapContextCreate(
      pixels.as_mut_ptr().cast::<c_void>(),
      width,
      height,
      8,
      stride,
      Some(&colour_space),
      CGImageAlphaInfo::NoneSkipLast.0,
    )
  }
  .expect("an RGBA8 bitmap context");
  CGContext::draw_image(
    Some(&context),
    CGRect {
      origin: CGPoint { x: 0.0, y: 0.0 },
      size: CGSize {
        width: width as f64,
        height: height as f64,
      },
    },
    Some(image),
  );
  drop(context);
  pixels
}

/// Repack RGBA8 into `format`, and report the expected RGB triple per
/// pixel — which is the input's own for the colour formats, and the
/// luma it was reduced to for `Gray8`.
fn repack(rgba: &[u8], format: PixelFormat) -> (Vec<u8>, Vec<[u8; 3]>) {
  let mut packed = Vec::with_capacity(rgba.len());
  let mut expected = Vec::with_capacity(rgba.len() / 4);
  for pixel in rgba.as_chunks::<4>().0 {
    let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
    match format {
      PixelFormat::Rgb8 => {
        packed.extend_from_slice(&[r, g, b]);
        expected.push([r, g, b]);
      }
      PixelFormat::Rgba8 => {
        // A junk alpha the door must not read.
        packed.extend_from_slice(&[r, g, b, 0xAB]);
        expected.push([r, g, b]);
      }
      PixelFormat::Bgra8 => {
        packed.extend_from_slice(&[b, g, r, 0xAB]);
        expected.push([r, g, b]);
      }
      PixelFormat::Gray8 => {
        // Rec. 601 luma in integer arithmetic; the value is what the
        // plane carries, so it is also what must come back on all three
        // channels.
        let luma = ((u16::from(r) * 77 + u16::from(g) * 150 + u16::from(b) * 29) >> 8) as u8;
        packed.push(luma);
        expected.push([luma, luma, luma]);
      }
    }
  }
  (packed, expected)
}

/// The largest per-channel difference between a rendered RGBA8 bitmap
/// and the triples it should have reproduced.
fn worst_channel_delta(rendered: &[u8], expected: &[[u8; 3]]) -> u8 {
  rendered
    .as_chunks::<4>()
    .0
    .iter()
    .zip(expected)
    .map(|(got, want)| {
      (0..3)
        .map(|c| got[c].abs_diff(want[c]))
        .max()
        .unwrap_or_default()
    })
    .max()
    .unwrap_or_default()
}

const FORMATS: [PixelFormat; 4] = [
  PixelFormat::Rgb8,
  PixelFormat::Rgba8,
  PixelFormat::Bgra8,
  PixelFormat::Gray8,
];

// ----- the exact half -------------------------------------------------------

/// Every format's `CGBitmapInfo` mapping, checked by rendering the built
/// image back out and comparing channels.
///
/// A swapped red and blue, an alpha byte read as colour, or a
/// misdeclared bit depth all still produce a picture Vision will happily
/// find faces in — so "3 faces came back" proves nothing about the
/// mapping. Byte equality does.
#[test]
fn every_format_reaches_core_graphics_channel_exact() {
  let (width, height, rgba) = decode_rgba(CREW);
  for format in FORMATS {
    let (packed, expected) = repack(&rgba, format);
    let plane = PixelPlane::packed(&packed, width, height, format).expect("a tight plane");
    let image = cg_image_from_plane(&plane).expect("core graphics builds the image");
    let rendered = render_rgba(&image, width as usize, height as usize);
    assert_eq!(
      worst_channel_delta(&rendered, &expected),
      0,
      "{format:?} must round-trip through Core Graphics with no channel moved"
    );
  }
}

/// Inter-row padding is compacted away, not passed through: a plane with
/// 37 junk bytes after every row renders identically to the tight one,
/// rather than skewing progressively as a mis-handled stride would.
#[test]
fn a_padded_stride_is_compacted_without_shifting_the_image() {
  let (width, height, rgba) = decode_rgba(CREW);
  let (tight, expected) = repack(&rgba, PixelFormat::Rgb8);
  let row_bytes = width as usize * 3;
  let stride = row_bytes + 37;

  let mut padded = vec![0x5Au8; stride * height as usize];
  for (row, source) in tight.chunks_exact(row_bytes).enumerate() {
    padded[row * stride..row * stride + row_bytes].copy_from_slice(source);
  }

  let plane =
    PixelPlane::new(&padded, width, height, stride, PixelFormat::Rgb8).expect("a padded plane");
  let image = cg_image_from_plane(&plane).expect("core graphics builds the image");
  let rendered = render_rgba(&image, width as usize, height as usize);
  assert_eq!(
    worst_channel_delta(&rendered, &expected),
    0,
    "the padding must be dropped, and every row must land where it started"
  );
}

/// The documented contract that the fourth byte is never read, made
/// falsifiable: two planes differing ONLY in their alpha bytes must
/// render to the same colours.
#[test]
fn the_alpha_byte_is_never_read() {
  let (width, height, rgba) = decode_rgba(CREW);
  for format in [PixelFormat::Rgba8, PixelFormat::Bgra8] {
    let (mut opaque, _) = repack(&rgba, format);
    let mut transparent = opaque.clone();
    // Both 32-bit formats carry their alpha in the fourth byte —
    // `R G B A` and `B G R A` — so one index serves both.
    for pixel in opaque.as_chunks_mut::<4>().0 {
      pixel[3] = 0xFF;
    }
    for pixel in transparent.as_chunks_mut::<4>().0 {
      pixel[3] = 0x00;
    }

    // Colour only. The destination bitmap ignores its own fourth byte
    // too, so Core Graphics is free to blit all four straight through —
    // and does: the rendered alpha byte carries whatever the source's
    // held. That is the same "never read" this asserts, seen from the
    // other side, and comparing it would be comparing the byte the
    // contract says means nothing.
    let render = |bytes: &[u8]| -> Vec<[u8; 3]> {
      let plane = PixelPlane::packed(bytes, width, height, format).expect("plane");
      let image = cg_image_from_plane(&plane).expect("image");
      render_rgba(&image, width as usize, height as usize)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect()
    };
    assert_eq!(
      render(&opaque),
      render(&transparent),
      "{format:?} must present the plane as opaque whatever the alpha byte holds"
    );
  }
}

// ----- the parity half ------------------------------------------------------

/// Both doors find the same three faces in the same places.
///
/// Measured on an Apple host: three faces through each door, paired by
/// ascending x, agreeing to within 0.002 in every normalized coordinate
/// and 0.005 in confidence. The tolerance is the two decode paths, not
/// slack — the boxes below match to roughly a thousandth.
#[test]
fn the_face_detector_finds_the_same_faces_through_both_doors() {
  let options = AppleVisionFaceOptions::new();
  let detector =
    FaceDetector::new(&options).expect("FaceDetector::new builds its Vision requests on this host");

  let mut through_jpeg = detector
    .detect::<Face>(CREW, &options)
    .expect("the jpeg door must return Ok");

  let (width, height, rgba) = decode_rgba(CREW);
  let (packed, _) = repack(&rgba, PixelFormat::Rgb8);
  let plane = PixelPlane::packed(&packed, width, height, PixelFormat::Rgb8).expect("plane");
  let mut through_pixels = detector
    .detect_pixels::<Face>(&plane, &options)
    .expect("the pixel door must return Ok");

  assert_eq!(
    through_jpeg.len(),
    3,
    "the fixture's three faces are the premise of this comparison"
  );
  assert_eq!(
    through_pixels.len(),
    through_jpeg.len(),
    "both doors must find the same number of faces"
  );

  // Vision does not promise an order, and the two doors empirically
  // return one frame's faces in different orders, so pair them by
  // position rather than by index.
  let by_x = |a: &Face, b: &Face| a.bbox.x.total_cmp(&b.bbox.x);
  through_jpeg.sort_by(by_x);
  through_pixels.sort_by(by_x);

  for (jpeg, pixels) in through_jpeg.iter().zip(&through_pixels) {
    for (name, a, b) in [
      ("x", jpeg.bbox.x, pixels.bbox.x),
      ("y", jpeg.bbox.y, pixels.bbox.y),
      ("width", jpeg.bbox.width, pixels.bbox.width),
      ("height", jpeg.bbox.height, pixels.bbox.height),
    ] {
      assert!(
        (a - b).abs() < 0.002,
        "the two doors' {name} must agree within the decode paths' own difference: {a} vs {b}"
      );
    }
    assert!(
      (jpeg.confidence - pixels.confidence).abs() < 0.005,
      "confidence must agree too: {} vs {}",
      jpeg.confidence,
      pixels.confidence
    );
  }
}

/// The eight-request batch survives the pixel door intact: an `Ok`, and
/// both frame-wide slots written — with a reading or with the engine's
/// own sentinel — exactly as the JPEG door promises.
#[test]
fn the_analyzer_batch_survives_the_pixel_door() {
  let options = AnalyzeOptions::new();
  let analyzer = VisionAnalyzer::new(&options)
    .expect("VisionAnalyzer::new builds its Vision requests on this host");

  let (width, height, rgba) = decode_rgba(AIRPORT);
  let (packed, _) = repack(&rgba, PixelFormat::Rgb8);
  let plane = PixelPlane::packed(&packed, width, height, PixelFormat::Rgb8).expect("plane");

  let analysis = analyzer
    .analyze_keyframe_pixels::<MediaSchema>(&plane, &options)
    .expect("the pixel door must return Ok, never abort the process");
  assert!(
    analysis.horizon().is_some(),
    "the horizon slot carries at least the no-detection sentinel"
  );
  assert!(
    analysis.aesthetics().is_some(),
    "the aesthetics slot carries at least the no-detection sentinel"
  );
}

/// A grayscale plane is a legal image, not a degenerate one: the same
/// frame reduced to luma still finds the fixture's three faces.
#[test]
fn a_gray8_plane_is_a_real_image() {
  let options = AppleVisionFaceOptions::new();
  let detector =
    FaceDetector::new(&options).expect("FaceDetector::new builds its Vision requests on this host");
  let (width, height, rgba) = decode_rgba(CREW);
  let (packed, _) = repack(&rgba, PixelFormat::Gray8);
  let plane = PixelPlane::packed(&packed, width, height, PixelFormat::Gray8).expect("plane");
  let faces = detector
    .detect_pixels::<Face>(&plane, &options)
    .expect("the pixel door must return Ok");
  assert_eq!(
    faces.len(),
    3,
    "the crew fixture's faces survive the reduction to luma"
  );
}

/// A QR code carrying a known payload, generated for this suite rather
/// than taken from anywhere. Provenance is recorded here and in
/// `tests/fixtures/README.md`:
///
/// ```text
/// qr_code.jpg
///   content:  AVANALYZE-PIXEL-DOOR
///   produced: segno, QR version 2, error correction H, scale 12, border 4,
///             then sips -s format jpeg -s formatOptions 90
///   396x396, 18006 bytes
///   sha256:   bd93cffe8b6bcb1434838906e3514cd85806e8a6ebf8482d5590a57bb46cfacf
///   licence:  none needed — generated for this repository
///   why:      the barcode capability has no other positive material in the
///             corpus, and a barcode door that silently returned nothing
///             would otherwise pass every test in this file.
/// ```
const QR: &[u8] = include_bytes!("../../tests/fixtures/qr_code.jpg");

/// Decode `jpeg` and hand back the tight `Rgb8` plane bytes for it.
fn rgb_plane_bytes(jpeg: &'static [u8]) -> (u32, u32, Vec<u8>) {
  let (width, height, rgba) = decode_rgba(jpeg);
  let (packed, _) = repack(&rgba, PixelFormat::Rgb8);
  (width, height, packed)
}

/// The finding-1 regression, stated as a property the image itself
/// carries: a padded plane's image is built over a TIGHT buffer.
///
/// `bytesPerRow` equal to `row_bytes` rather than the caller's stride is
/// the observable half of "one copy, padding dropped during it". The
/// path that produced this used to compact into a `Vec` and then let
/// `CFData` copy that — a third live image at the ceiling, and an
/// infallible allocation whose failure aborted the process instead of
/// refusing the frame.
#[test]
fn a_padded_plane_is_imaged_over_a_tight_buffer() {
  let (width, height, tight) = rgb_plane_bytes(CREW);
  let row_bytes = width as usize * 3;
  let stride = row_bytes + 37;
  let mut padded = vec![0x5Au8; stride * height as usize];
  for (row, source) in tight.chunks_exact(row_bytes).enumerate() {
    padded[row * stride..row * stride + row_bytes].copy_from_slice(source);
  }

  let plane = PixelPlane::new(&padded, width, height, stride, PixelFormat::Rgb8).expect("plane");
  let image = cg_image_from_plane(&plane).expect("image");
  assert_eq!(
    CGImage::bytes_per_row(Some(&image)),
    row_bytes,
    "the image must be built over the compacted rows, never the caller's stride"
  );
  assert_eq!(CGImage::width(Some(&image)), width as usize);
  assert_eq!(CGImage::height(Some(&image)), height as usize);
}

/// Door parity across every capability, on a real photograph — and, for
/// the seven the fixture actually carries material for, PRESENCE.
///
/// Presence is the point. `run_requests` reports a caught Objective-C
/// exception as `Ok` with the caller's empty fallback, so "returned
/// `Ok`" is satisfied by a door that silently finds nothing, forever. A
/// count that must be non-zero is not.
///
/// Measured on an Apple host over eight consecutive runs, byte-stable
/// every time — both doors, on `apollo11_crew.jpg`:
///
/// ```text
///   classifications 4   human subjects 3   attention 1   objectness 1
///   faces 3   landmark sets 3   body poses 3   hand poses 2
///   instance masks 3   segmentation masks 1   3-D body poses 1
///   text 0   barcodes 0   animal poses 0
/// ```
///
/// The three zeros are honest: the photograph carries no text Vision
/// reads at this scale, no barcode and no animal. The barcode gap is
/// closed separately by the QR fixture below.
///
/// 3-D body poses were a fourth zero when this door landed, for a
/// defect on `main` rather than of this door. That defect is fixed and
/// the capability is covered here like any other; what its joints
/// actually contain is asserted against the framework's own numbers in
/// `src/tests/body_pose.rs`, because a count alone would not have
/// caught the second half of it.
#[test]
fn every_pixel_door_matches_its_jpeg_twin_on_the_crew_fixture() {
  use mediaschema::domain::aggregates::video as ms;

  let (width, height, packed) = rgb_plane_bytes(CREW);
  let plane = PixelPlane::packed(&packed, width, height, PixelFormat::Rgb8).expect("plane");

  let options = AnalyzeOptions::new();
  let analyzer = VisionAnalyzer::new(&options)
    .expect("VisionAnalyzer::new builds its Vision requests on this host");
  let jpeg = analyzer
    .analyze_keyframe::<MediaSchema>(CREW, &options)
    .expect("jpeg door");
  let pixels = analyzer
    .analyze_keyframe_pixels::<MediaSchema>(&plane, &options)
    .expect("pixel door");
  for (slot, a, b) in [
    (
      "classifications",
      jpeg.classifications().len(),
      pixels.classifications().len(),
    ),
    (
      "human subjects",
      jpeg.human_subjects().len(),
      pixels.human_subjects().len(),
    ),
    (
      "attention saliency",
      jpeg.attention_saliency().len(),
      pixels.attention_saliency().len(),
    ),
    (
      "objectness saliency",
      jpeg.objectness_saliency().len(),
      pixels.objectness_saliency().len(),
    ),
  ] {
    assert_eq!(a, b, "the two doors must agree on {slot}");
    assert!(
      a > 0,
      "the crew fixture carries {slot}, so a door that found none is broken"
    );
  }

  /// Both doors' counts, and whether the fixture carries the capability
  /// at all.
  fn agree(capability: &str, jpeg: usize, pixels: usize, carried: bool) {
    assert_eq!(jpeg, pixels, "the two doors must agree on {capability}");
    if carried {
      assert!(
        pixels > 0,
        "the crew fixture carries {capability}, so a pixel door that found none is broken"
      );
    }
  }

  let options = AppleVisionTextOptions::new();
  let recognizer = TextRecognizer::new(&options)
    .expect("TextRecognizer::new builds its Vision requests on this host");
  agree(
    "text",
    recognizer
      .recognize::<ms::TextDetection>(CREW, &options)
      .expect("jpeg")
      .len(),
    recognizer
      .recognize_pixels::<ms::TextDetection>(&plane, &options)
      .expect("pixels")
      .len(),
    false,
  );

  let options = AppleVisionBarcodeOptions::new();
  let detector = BarcodeDetector::new(&options)
    .expect("BarcodeDetector::new builds its Vision requests on this host");
  agree(
    "barcodes",
    detector
      .detect::<ms::BarcodeDetection>(CREW, &options)
      .expect("jpeg")
      .len(),
    detector
      .detect_pixels::<ms::BarcodeDetection>(&plane, &options)
      .expect("pixels")
      .len(),
    false,
  );

  let options = AppleVisionFaceOptions::new();
  let detector =
    FaceDetector::new(&options).expect("FaceDetector::new builds its Vision requests on this host");
  agree(
    "faces",
    detector
      .detect::<ms::FaceDetection>(CREW, &options)
      .expect("jpeg")
      .len(),
    detector
      .detect_pixels::<ms::FaceDetection>(&plane, &options)
      .expect("pixels")
      .len(),
    true,
  );

  let options = AppleVisionFaceLandmarkOptions::new();
  let landmarker = FaceLandmarker::new(&options)
    .expect("FaceLandmarker::new builds its Vision requests on this host");
  agree(
    "landmark sets",
    landmarker
      .detect::<ms::FaceLandmarksDetection>(CREW, &options)
      .expect("jpeg")
      .len(),
    landmarker
      .detect_pixels::<ms::FaceLandmarksDetection>(&plane, &options)
      .expect("pixels")
      .len(),
    true,
  );

  let options = AppleVisionBodyPoserOptions::new();
  let poser =
    BodyPoser::new(&options).expect("BodyPoser::new builds its Vision requests on this host");
  agree(
    "2-D body poses",
    poser
      .detect_2d::<ms::BodyPoseDetection>(CREW, &options)
      .expect("jpeg")
      .len(),
    poser
      .detect_2d_pixels::<ms::BodyPoseDetection>(&plane, &options)
      .expect("pixels")
      .len(),
    true,
  );
  agree(
    "3-D body poses",
    poser
      .detect_3d::<ms::BodyPose3DDetection>(CREW, &options)
      .expect("jpeg")
      .len(),
    poser
      .detect_3d_pixels::<ms::BodyPose3DDetection>(&plane, &options)
      .expect("pixels")
      .len(),
    true,
  );

  let options = AppleVisionHandPoseOptions::new();
  let hands =
    HandPoser::new(&options).expect("HandPoser::new builds its Vision requests on this host");
  agree(
    "hand poses",
    hands
      .detect::<ms::HandPoseDetection>(CREW, &options)
      .expect("jpeg")
      .len(),
    hands
      .detect_pixels::<ms::HandPoseDetection>(&plane, &options)
      .expect("pixels")
      .len(),
    true,
  );

  let options = AppleVisionAnimalPoseOptions::new();
  let animals =
    AnimalPoser::new(&options).expect("AnimalPoser::new builds its Vision requests on this host");
  agree(
    "animal poses",
    animals
      .detect::<ms::BodyPoseDetection>(CREW, &options)
      .expect("jpeg")
      .len(),
    animals
      .detect_pixels::<ms::BodyPoseDetection>(&plane, &options)
      .expect("pixels")
      .len(),
    false,
  );

  let options = AppleVisionPersonMaskerOptions::new();
  let masker =
    PersonMasker::new(&options).expect("PersonMasker::new builds its Vision requests on this host");
  agree(
    "instance masks",
    masker
      .instance_masks::<ms::PersonInstanceMaskDetection>(CREW, &options)
      .expect("jpeg")
      .len(),
    masker
      .instance_masks_pixels::<ms::PersonInstanceMaskDetection>(&plane, &options)
      .expect("pixels")
      .len(),
    true,
  );
  agree(
    "segmentation masks",
    masker
      .segmentation_masks::<ms::PersonSegmentationMask>(CREW, &options)
      .expect("jpeg")
      .len(),
    masker
      .segmentation_masks_pixels::<ms::PersonSegmentationMask>(&plane, &options)
      .expect("pixels")
      .len(),
    true,
  );
}

/// The barcode door reads a payload, not merely `Ok`.
///
/// This is the capability with no positive material anywhere else in the
/// corpus, and the one where a silently-empty door would otherwise pass
/// every other assertion in this file. The QR carries a string, and both
/// doors must come back with that exact string.
#[test]
fn the_pixel_door_reads_a_barcode_payload() {
  use mediaschema::domain::aggregates::video as ms;

  let (width, height, packed) = rgb_plane_bytes(QR);
  let plane = PixelPlane::packed(&packed, width, height, PixelFormat::Rgb8).expect("plane");

  let options = AppleVisionBarcodeOptions::new();
  let detector = BarcodeDetector::new(&options)
    .expect("BarcodeDetector::new builds its Vision requests on this host");
  let through_jpeg = detector
    .detect::<ms::BarcodeDetection>(QR, &options)
    .expect("the jpeg door must return Ok");
  let through_pixels = detector
    .detect_pixels::<ms::BarcodeDetection>(&plane, &options)
    .expect("the pixel door must return Ok");

  assert_eq!(through_jpeg.len(), 1, "the fixture carries one QR code");
  assert_eq!(
    through_pixels.len(),
    1,
    "the pixel door must read the same one QR code"
  );
  assert_eq!(
    through_pixels[0].payload(),
    "AVANALYZE-PIXEL-DOOR",
    "the pixel door must decode the payload, not merely detect a shape"
  );
  assert_eq!(
    through_pixels[0].payload(),
    through_jpeg[0].payload(),
    "both doors must decode the same payload"
  );
}

/// The text door reads text, and the two doors are allowed to disagree
/// about how much.
///
/// On `airport_keyframe.jpg` the JPEG door reads one run and the pixel
/// door reads two, stably across eight runs — a borderline reading that
/// clears the confidence gate on one decode path and not the other. That
/// is the "not bit for bit" this design states, seen in a COUNT rather
/// than in a coordinate, so the assertion here is presence on both
/// sides rather than equality.
#[test]
fn the_pixel_door_reads_text_even_where_the_two_doors_count_differently() {
  use mediaschema::domain::aggregates::video as ms;

  let (width, height, packed) = rgb_plane_bytes(AIRPORT);
  let plane = PixelPlane::packed(&packed, width, height, PixelFormat::Rgb8).expect("plane");

  let options = AppleVisionTextOptions::new();
  let recognizer = TextRecognizer::new(&options)
    .expect("TextRecognizer::new builds its Vision requests on this host");
  let through_jpeg = recognizer
    .recognize::<ms::TextDetection>(AIRPORT, &options)
    .expect("the jpeg door must return Ok");
  let through_pixels = recognizer
    .recognize_pixels::<ms::TextDetection>(&plane, &options)
    .expect("the pixel door must return Ok");

  assert!(
    !through_jpeg.is_empty(),
    "the keyframe carries text the jpeg door reads"
  );
  assert!(
    !through_pixels.is_empty(),
    "so a pixel door that reads none of it is broken"
  );
}
