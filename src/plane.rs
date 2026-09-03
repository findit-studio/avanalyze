//! The raw-pixel door's input: an image a caller has already decoded.
//!
//! Nothing here touches Apple's frameworks. A [`PixelPlane`] is a
//! borrowed byte slice plus the four numbers needed to read it as an
//! image, and every rule it enforces is arithmetic — so the type, its
//! refusals, and their tests exist and run on every target, not just on
//! Apple.

use crate::{AnalyzeError, AnalyzeErrorKind};

/// Upper bound on the decoded-pixel byte count either door may put in
/// front of Vision.
///
/// Both doors reach it from opposite directions. The JPEG door never
/// sees the decoded image: it reads the compressed stream's SOF marker
/// and refuses a frame whose *declared* dimensions would make
/// Vision/ImageIO allocate past this — see
/// [`check_decoded_dimensions`](crate::ffi::check_decoded_dimensions),
/// which must bound an allocation it does not perform. The pixel door
/// is handed the decoded bytes outright, so it measures the plane's own
/// byte extent ([`PixelPlane::new`]) against the same ceiling.
///
/// 512 MiB is far past any real keyframe — ~178 megapixels of packed
/// 24-bit RGB — but bounded rather than unbounded, so a hostile or
/// corrupted input cannot drive the worker into memory pressure through
/// either door.
pub(crate) const MAX_DECODED_IMAGE_BYTES: u64 = 512 * 1024 * 1024;

/// How one pixel of a [`PixelPlane`] is laid out in memory.
///
/// Every variant is 8 bits per component and interleaved — the shape a
/// video decoder or an image codec hands back. Planar (`YCbCr`) layouts
/// are deliberately absent: they are several buffers, not one, and this
/// door takes one.
///
/// # Alpha is ignored, never composited
///
/// [`Rgba8`](Self::Rgba8) and [`Bgra8`](Self::Bgra8) name where the
/// alpha byte *sits* so the colour bytes can be found around it. The
/// byte itself is never read: the plane is presented to Vision as
/// opaque, whatever it holds. A caller with straight or premultiplied
/// alpha therefore gets the same detections either way, and one whose
/// fourth byte is padding does not have to zero it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum PixelFormat {
  /// Packed 24-bit colour, byte order `R`, `G`, `B`.
  Rgb8,
  /// 32-bit colour, byte order `R`, `G`, `B`, `A`; the alpha byte is
  /// ignored.
  Rgba8,
  /// 32-bit colour, byte order `B`, `G`, `R`, `A`; the alpha byte is
  /// ignored.
  Bgra8,
  /// 8-bit luminance, one byte per pixel.
  Gray8,
}

impl PixelFormat {
  /// Bytes one pixel of this format occupies.
  #[inline]
  pub const fn bytes_per_pixel(self) -> usize {
    match self {
      Self::Rgb8 => 3,
      Self::Rgba8 | Self::Bgra8 => 4,
      Self::Gray8 => 1,
    }
  }
}

/// A borrowed, already-decoded image plane — the pixel door's input.
///
/// Construction is the whole contract. [`new`](Self::new) and
/// [`packed`](Self::packed) refuse anything the door could not read
/// safely: a zero dimension, a stride narrower than one row, an extent
/// past `MAX_DECODED_IMAGE_BYTES` — the engine's 512 MiB decoded-size
/// ceiling, which the JPEG door enforces too — or a slice too short for the
/// geometry it claims. Everything downstream of a constructed
/// `PixelPlane` may therefore index it without re-checking, because a
/// `PixelPlane` that exists describes bytes that are there.
///
/// The slice is borrowed for `'a` and never mutated. The engine copies
/// the rows it needs during the call — see
/// [`VisionAnalyzer::analyze_keyframe_pixels`](crate::VisionAnalyzer::analyze_keyframe_pixels)
/// for why that copy exists — so the caller's buffer is free the moment
/// the call returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PixelPlane<'a> {
  data: &'a [u8],
  width: u32,
  height: u32,
  stride: usize,
  format: PixelFormat,
}

impl<'a> PixelPlane<'a> {
  /// Describes `data` as a `width` × `height` image of `format` whose
  /// rows begin every `stride` bytes.
  ///
  /// `stride` is the full byte distance from one row's first byte to
  /// the next row's, so it counts any padding a decoder left at the end
  /// of a row. Where there is none, [`packed`](Self::packed) computes
  /// it.
  ///
  /// # Errors
  ///
  /// [`AnalyzeErrorKind::RequestFailed`] — the same refusal the JPEG
  /// door returns for an input it will not put in front of Vision —
  /// when:
  ///
  /// - `width` or `height` is zero;
  /// - `stride` is smaller than one row of pixels
  ///   (`width * format.bytes_per_pixel()`), which would make rows
  ///   overlap;
  /// - the plane's byte extent exceeds `MAX_DECODED_IMAGE_BYTES`, the
  ///   engine's 512 MiB decoded-size ceiling;
  /// - `data` is shorter than that extent.
  ///
  /// The extent is `stride * (height - 1) + row_bytes`, not
  /// `stride * height`: the final row's trailing padding is never read,
  /// so a buffer that stops right after the last pixel is accepted.
  pub fn new(
    data: &'a [u8],
    width: u32,
    height: u32,
    stride: usize,
    format: PixelFormat,
  ) -> Result<Self, AnalyzeError> {
    if width == 0 || height == 0 {
      return Err(refused("pixel plane has a zero width or height"));
    }
    let row_bytes = (width as usize)
      .checked_mul(format.bytes_per_pixel())
      .ok_or_else(|| refused("pixel plane's row byte count overflows"))?;
    if stride < row_bytes {
      return Err(refused("pixel plane's stride is narrower than one row"));
    }
    // `height - 1` cannot underflow: `height != 0` was refused above.
    let extent = (stride as u64)
      .checked_mul(u64::from(height - 1))
      .and_then(|rows| rows.checked_add(row_bytes as u64))
      .ok_or_else(|| refused("pixel plane's byte extent overflows"))?;
    if extent > MAX_DECODED_IMAGE_BYTES {
      return Err(refused(
        "pixel plane's byte extent exceeds MAX_DECODED_IMAGE_BYTES",
      ));
    }
    // `extent <= MAX_DECODED_IMAGE_BYTES` bounds it well inside `usize`
    // on every target this crate builds for, so the comparison against
    // a slice length is exact rather than truncating.
    if (data.len() as u64) < extent {
      return Err(refused(
        "pixel plane's buffer is shorter than its width, height and stride require",
      ));
    }
    Ok(Self {
      data,
      width,
      height,
      stride,
      format,
    })
  }

  /// Describes `data` as a `width` × `height` image of `format` with no
  /// padding between rows.
  ///
  /// The shorthand for the common case, and it elides nothing: the
  /// stride is `width * format.bytes_per_pixel()`, computed rather than
  /// defaulted. Every refusal of [`new`](Self::new) applies.
  ///
  /// # Errors
  ///
  /// As [`new`](Self::new).
  pub fn packed(
    data: &'a [u8],
    width: u32,
    height: u32,
    format: PixelFormat,
  ) -> Result<Self, AnalyzeError> {
    let stride = (width as usize)
      .checked_mul(format.bytes_per_pixel())
      .ok_or_else(|| refused("pixel plane's row byte count overflows"))?;
    Self::new(data, width, height, stride, format)
  }

  /// The borrowed pixels.
  #[inline]
  pub const fn data(&self) -> &'a [u8] {
    self.data
  }

  /// Pixels per row.
  #[inline]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Rows.
  #[inline]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Bytes from one row's first byte to the next row's.
  #[inline]
  pub const fn stride(&self) -> usize {
    self.stride
  }

  /// How one pixel is laid out.
  #[inline]
  pub const fn format(&self) -> PixelFormat {
    self.format
  }

  /// Bytes of actual pixels in one row, padding excluded.
  ///
  /// Always `<= stride`, and never overflows: the product was computed
  /// and bounded at construction.
  #[inline]
  pub const fn row_bytes(&self) -> usize {
    // Cannot overflow: `new` refused the plane if it did.
    (self.width as usize) * self.format.bytes_per_pixel()
  }
}

/// A plane the engine will not put in front of Vision.
///
/// [`AnalyzeErrorKind::RequestFailed`] rather than a kind of its own:
/// the JPEG door already reports its own pre-flight refusals — an
/// over-ceiling payload, an SOF declaring a decoded size past the cap —
/// under exactly this kind, and a malformed plane is the same event at
/// the same point, "the frame was refused before the Vision pass".
#[inline]
fn refused(message: &'static str) -> AnalyzeError {
  AnalyzeError::new(AnalyzeErrorKind::RequestFailed, message)
}
