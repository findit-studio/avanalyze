//! The pixel door's admission gate: what [`PixelPlane`] accepts, what
//! it refuses, and why every refusal is arithmetic rather than
//! platform — these run on Linux too.

use crate::{AnalyzeErrorKind, PixelFormat, PixelPlane, plane::MAX_DECODED_IMAGE_BYTES};

/// The four formats, with the byte width each one claims.
const FORMATS: [(PixelFormat, usize); 4] = [
  (PixelFormat::Rgb8, 3),
  (PixelFormat::Rgba8, 4),
  (PixelFormat::Bgra8, 4),
  (PixelFormat::Gray8, 1),
];

#[test]
fn every_format_reports_its_own_byte_width() {
  for (format, expected) in FORMATS {
    assert_eq!(
      format.bytes_per_pixel(),
      expected,
      "{format:?} must report the byte width its layout actually occupies"
    );
  }
}

/// A tight buffer of exactly the right size is accepted, and the plane
/// hands back what it was given.
#[test]
fn a_packed_plane_reports_the_geometry_it_was_built_with() {
  for (format, bytes_per_pixel) in FORMATS {
    let data = vec![0u8; 7 * 5 * bytes_per_pixel];
    let plane = PixelPlane::packed(&data, 7, 5, format).expect("an exact buffer must be accepted");
    assert_eq!(plane.width(), 7);
    assert_eq!(plane.height(), 5);
    assert_eq!(plane.format(), format);
    assert_eq!(
      plane.stride(),
      7 * bytes_per_pixel,
      "`packed` computes the stride rather than defaulting it"
    );
    assert_eq!(plane.row_bytes(), 7 * bytes_per_pixel);
    assert_eq!(plane.data().len(), data.len());
  }
}

/// A stride wider than a row is the padded case, and it is legal. What
/// the plane must NOT demand is padding after the final row: a decoder
/// that stops at the last pixel has handed over a complete image.
#[test]
fn the_final_row_may_end_without_its_padding() {
  const STRIDE: usize = 7 * 3 + 11;
  let exact = STRIDE * (5 - 1) + 7 * 3;

  let data = vec![0u8; exact];
  let plane = PixelPlane::new(&data, 7, 5, STRIDE, PixelFormat::Rgb8)
    .expect("a buffer that ends with the last pixel is a whole image");
  assert_eq!(plane.stride(), STRIDE);

  let one_short = vec![0u8; exact - 1];
  let err = PixelPlane::new(&one_short, 7, 5, STRIDE, PixelFormat::Rgb8)
    .expect_err("one byte short of the last pixel is not");
  assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
  assert!(err.message().contains("shorter"), "{}", err.message());
}

#[test]
fn a_zero_dimension_is_refused() {
  let data = [0u8; 64];
  for (width, height) in [(0, 4), (4, 0), (0, 0)] {
    let err = PixelPlane::packed(&data, width, height, PixelFormat::Rgb8)
      .expect_err("an image with no pixels is not an image");
    assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
    assert!(err.message().contains("zero width or height"));
  }
}

/// A stride narrower than one row would make consecutive rows overlap,
/// so the same bytes would be read as two different rows.
#[test]
fn a_stride_narrower_than_one_row_is_refused() {
  let data = [0u8; 4096];
  let err = PixelPlane::new(&data, 8, 4, 8 * 3 - 1, PixelFormat::Rgb8)
    .expect_err("rows that overlap are not rows");
  assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
  assert!(err.message().contains("narrower than one row"));
}

/// The undersized-buffer refusal is the one that keeps every later read
/// in bounds: nothing downstream re-checks, because a constructed plane
/// is the statement that the bytes are there.
#[test]
fn a_buffer_shorter_than_the_geometry_is_refused() {
  for (format, bytes_per_pixel) in FORMATS {
    let short = vec![0u8; 8 * 4 * bytes_per_pixel - 1];
    let err = PixelPlane::packed(&short, 8, 4, format)
      .expect_err("a plane may not claim more bytes than it borrows");
    assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
    assert!(err.message().contains("shorter"), "{}", err.message());
  }
}

/// The decoded-size ceiling is checked against the geometry, BEFORE the
/// buffer length — so a hostile caller cannot get the engine to reason
/// about a 13 GiB image at all. The tiny slice here is the proof: the
/// refusal that fires names the ceiling, not the length.
#[test]
fn an_extent_over_the_ceiling_is_refused_before_the_buffer_is_measured() {
  let err = PixelPlane::packed(&[], u16::MAX as u32, u16::MAX as u32, PixelFormat::Rgb8)
    .expect_err("an over-ceiling extent must be refused");
  assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
  assert!(
    err.message().contains("MAX_DECODED_IMAGE_BYTES"),
    "the ceiling must be what refuses it, not the buffer length: {}",
    err.message()
  );
}

/// The ceiling is a boundary, not an approximation: the largest plane
/// that fits is accepted and one pixel more is not. Neither is
/// allocated — both refusals precede the buffer-length check, and the
/// accepted case is measured with a slice long enough only because the
/// test builds it at a shape whose extent is exactly the cap.
#[test]
fn the_ceiling_admits_its_own_boundary_and_nothing_past_it() {
  // 512 MiB of Gray8 is 512 MiB of pixels: one row of exactly the cap.
  let width = u32::try_from(MAX_DECODED_IMAGE_BYTES).expect("the cap fits u32 as a Gray8 row");
  let over = PixelPlane::packed(&[], width, 2, PixelFormat::Gray8)
    .expect_err("two rows of the cap is twice the cap");
  assert!(over.message().contains("MAX_DECODED_IMAGE_BYTES"));

  // One row at the cap passes the ceiling and is then refused for the
  // honest reason — the test does not own 512 MiB to hand it.
  let at = PixelPlane::packed(&[], width, 1, PixelFormat::Gray8)
    .expect_err("the empty slice cannot carry it");
  assert!(
    at.message().contains("shorter"),
    "a plane exactly at the ceiling must pass the ceiling and fail on the buffer: {}",
    at.message()
  );
}

/// Every product in the geometry is checked rather than wrapped, so a
/// caller passing `usize::MAX` gets a refusal instead of a panic or a
/// small number that passes every later bound.
#[test]
fn overflowing_geometry_is_refused_rather_than_wrapped() {
  let err = PixelPlane::new(&[], 4, 3, usize::MAX, PixelFormat::Rgb8)
    .expect_err("a stride of usize::MAX overflows the extent");
  assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
  assert!(err.message().contains("overflow"), "{}", err.message());

  let err = PixelPlane::packed(&[], u32::MAX, 1, PixelFormat::Rgba8)
    .expect_err("the widest possible row is far past the ceiling");
  assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
}

/// `new` and `packed` are the same constructor; `packed` only computes
/// the stride the caller would otherwise have written out.
#[test]
fn packed_is_new_with_the_stride_computed() {
  let data = vec![0u8; 9 * 6 * 4];
  let packed = PixelPlane::packed(&data, 9, 6, PixelFormat::Bgra8).expect("packed");
  let spelled = PixelPlane::new(&data, 9, 6, 9 * 4, PixelFormat::Bgra8).expect("new");
  assert_eq!(packed, spelled);
}

#[cfg(feature = "serde")]
#[test]
fn pixel_format_round_trips_through_serde() {
  for (format, _) in FORMATS {
    let json = serde_json::to_string(&format).expect("serialize");
    let back: PixelFormat = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(format, back);
  }
}
