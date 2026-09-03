//! Person masks — per-instance and whole-frame — behind one door.

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2_core_video::{
  CVPixelBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
  CVPixelBufferGetDataSize, CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType,
  CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
  CVPixelBufferUnlockBaseAddress, kCVPixelFormatType_OneComponent8,
  kCVPixelFormatType_OneComponent32Float, kCVReturnSuccess,
};
#[cfg(target_vendor = "apple")]
use objc2_foundation::{NSIndexSet, NSNotFound};
#[cfg(target_vendor = "apple")]
use objc2_vision::*;

#[cfg(target_vendor = "apple")]
use crate::ffi::{
  ImageSource, MAX_VISION_RESULTS_PER_FRAME, guard_vision_ffi, run_requests, sanitize_confidence,
};
use crate::{AnalyzeError, AppleVisionPersonMaskerOptions, BoundingBox, PixelPlane};

/// Upper bound on a single mask payload (post-packing, 8 bits per
/// pixel) before we refuse to allocate. 64 MiB covers any sane image
/// resolution Apple Vision returns today (8K = ~33 MiB at 8 bits per
/// pixel) and prevents a runaway / corrupted `width * height` from
/// driving the worker process into the allocator's abort path.
#[cfg(target_vendor = "apple")]
pub(crate) const MAX_MASK_BYTES: usize = 64 * 1024 * 1024;

/// Hard ceiling on instances per segmentation-mask observation.
#[cfg(target_vendor = "apple")]
pub(crate) const MAX_NESTED_INSTANCES_PER_OBSERVATION: usize = 64;

/// Hard ceiling on the total mask count emitted per call across
/// the inner observation × instance loop. Without this, an outer
/// cap of 4096 observations × an inner cap of 64 instances would
/// permit 256K mask emissions per call even though each individual
/// mask is capped at [`MAX_MASK_BYTES`]. 256 is a generous total
/// matching realistic Vision output for a single frame.
#[cfg(target_vendor = "apple")]
pub(crate) const MAX_TOTAL_MASKS_PER_CALL: usize = 256;

/// Hard ceiling on the cumulative mask payload bytes emitted per
/// call. Even at the per-mask cap, a worst-case 256 masks × 64 MiB
/// = 16 GiB would crush the worker. 256 MiB total is generous for
/// realistic Vision output while bounding the cumulative budget.
#[cfg(target_vendor = "apple")]
pub(crate) const MAX_TOTAL_MASK_BYTES_PER_CALL: usize = 256 * 1024 * 1024;

/// Hard ceiling on the cumulative mask WALK STEPS attempted per call.
/// One step is one observation visited by an extractor, or one
/// instance index visited inside the instance walk. The emission-only
/// counters (count, bytes) only increment after a successful copy and
/// push; a corrupted Vision result whose confidence / u32-fit /
/// generate / copy gates all fail would otherwise drive unbounded
/// traversal — `NSIndexSet` walks and `generateMaskForInstances_error`
/// calls, each forcing Vision to compute/allocate a mask buffer —
/// while the emission counters stay below their caps. The attempt
/// budget bounds the failure-path work itself, and is charged at each
/// step's entry so no rejection branch beneath it is free.
///
/// Sized as `4 * MAX_TOTAL_MASKS_PER_CALL` (= 1024): each emitted
/// mask gets up to 3 further steps — a failed sibling attempt, the
/// visit of the observation that carried it — before the budget
/// trips, which leaves ample headroom for transient Vision
/// failures while keeping the attempt cap tied to the emission
/// cap rather than the much larger general results-array cap.
#[cfg(target_vendor = "apple")]
pub(crate) const MAX_TOTAL_MASK_ATTEMPTS_PER_CALL: usize = 4 * MAX_TOTAL_MASKS_PER_CALL;

/// One person-instance segmentation mask.
///
/// `data` is **always** one byte per pixel, `width * height` bytes
/// long, row-major and top-down, regardless of which pixel format
/// Vision produced. `width` / `height` describe the mask buffer, which
/// need not match the source frame's dimensions.
///
/// Note the argument order against [`PersonSegmentationMask::try_new`]:
/// the two differ only by `instance_index` in the middle.
pub trait PersonInstanceMaskDetection: Sized {
  /// Why a mask was refused.
  type Error;
  /// The geometry type this mask is built from.
  type BoundingBox: BoundingBox;

  /// Builds an instance mask. `bbox` is the normalized bounding box of
  /// the mask's foreground pixels.
  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    instance_index: u32,
    width: u32,
    height: u32,
    data: &[u8],
  ) -> Result<Self, Self::Error>;
}

/// One whole-frame person segmentation mask.
///
/// Same payload contract as [`PersonInstanceMaskDetection`], without
/// an instance index.
pub trait PersonSegmentationMask: Sized {
  /// Why a mask was refused.
  type Error;
  /// The geometry type this mask is built from.
  type BoundingBox: BoundingBox;

  /// Builds a segmentation mask. `bbox` is the normalized bounding box
  /// of the mask's foreground pixels.
  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    width: u32,
    height: u32,
    data: &[u8],
  ) -> Result<Self, Self::Error>;
}

/// Apple Vision person masking, both kinds — one per worker thread.
///
/// Owns two Vision requests behind one door, and each method performs
/// only its own: a consumer of whole-frame segmentation never runs the
/// instance model, and vice versa.
///
/// **Budget note.** The per-call mask ceilings (count, cumulative
/// bytes, generation attempts) are charged per *call*, so calling both
/// methods spends two budgets where the pre-split single-pass engine
/// shared one across both mask surfaces. That is strictly more
/// permissive; a caller that needs the old cumulative ceiling should
/// impose it above this crate.
///
/// The retained `VNRequest`s carry per-call state across
/// `performRequests` / `results()`, so a masker is not safe to share
/// across threads; build one per worker.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct PersonMasker {
  instances: Retained<VNGeneratePersonInstanceMaskRequest>,
  segmentation: Retained<VNGeneratePersonSegmentationRequest>,
}

#[cfg(target_vendor = "apple")]
impl PersonMasker {
  /// Creates a masker holding both mask requests at their pinned
  /// revisions.
  ///
  /// `_options` is unused: Apple bakes no knob this crate exposes into
  /// these request objects, so every gate is read per call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionPersonMaskerOptions) -> Self {
    unsafe {
      let instances = VNGeneratePersonInstanceMaskRequest::new();
      instances.setRevision(VNGeneratePersonInstanceMaskRequestRevision1);

      let segmentation = VNGeneratePersonSegmentationRequest::new();
      segmentation.setRevision(VNGeneratePersonSegmentationRequestRevision1);

      Self {
        instances,
        segmentation,
      }
    }
  }

  /// Logs the pinned revision of both mask requests.
  ///
  /// A revision drift changes mask geometry **silently** — same API,
  /// different pixels.
  #[cfg(feature = "tracing")]
  pub fn log_request_revisions(&self) {
    unsafe {
      tracing::info!(
        person_instance_mask_rev = self.instances.revision(),
        person_segmentation_rev = self.segmentation.revision(),
        "initialized pinned Apple Vision request revisions"
      );
    }
  }

  /// Generates one mask per detected person instance in `jpeg_data`.
  ///
  /// Performs only the instance request; the whole-frame segmentation
  /// model is not loaded.
  pub fn instance_masks<M: PersonInstanceMaskDetection>(
    &self,
    jpeg_data: &[u8],
    options: &AppleVisionPersonMaskerOptions,
  ) -> Result<Vec<M>, AnalyzeError> {
    self.instance_masks_on::<M>(ImageSource::Jpeg(jpeg_data), options)
  }

  /// Generates one mask per detected person instance in already-decoded
  /// `pixels`.
  ///
  /// [`instance_masks`](Self::instance_masks) reached without the
  /// encode: same request, same options, same budgets, same output. The
  /// whole-frame segmentation model is still not loaded.
  pub fn instance_masks_pixels<M: PersonInstanceMaskDetection>(
    &self,
    pixels: &PixelPlane<'_>,
    options: &AppleVisionPersonMaskerOptions,
  ) -> Result<Vec<M>, AnalyzeError> {
    self.instance_masks_on::<M>(ImageSource::Plane(pixels), options)
  }

  /// The one instance-mask body both doors reach.
  fn instance_masks_on<M: PersonInstanceMaskDetection>(
    &self,
    source: ImageSource<'_>,
    options: &AppleVisionPersonMaskerOptions,
  ) -> Result<Vec<M>, AnalyzeError> {
    let requests = unsafe {
      [Retained::cast_unchecked::<VNRequest>(
        self.instances.clone(),
      )]
    };
    run_requests(source, &requests, Vec::new(), || {
      // The mask closure captures `&mut` budget counters, hence the
      // `AssertUnwindSafe` inside `guard_vision_ffi`: a caught
      // exception leaves a counter at its partial (over-counted,
      // never under-counted) value, the safe direction for a cap.
      let mut budget = MaskBudget::new();
      guard_vision_ffi("person_instance_mask", Vec::new(), || {
        self.extract_instances::<M>(options, &mut budget)
      })
    })
  }

  /// Generates the whole-frame person segmentation mask(s) for
  /// `jpeg_data`.
  ///
  /// Performs only the segmentation request; the instance model is not
  /// loaded.
  pub fn segmentation_masks<M: PersonSegmentationMask>(
    &self,
    jpeg_data: &[u8],
    options: &AppleVisionPersonMaskerOptions,
  ) -> Result<Vec<M>, AnalyzeError> {
    self.segmentation_masks_on::<M>(ImageSource::Jpeg(jpeg_data), options)
  }

  /// Generates the whole-frame person segmentation mask(s) for
  /// already-decoded `pixels`.
  ///
  /// [`segmentation_masks`](Self::segmentation_masks) reached without
  /// the encode: same request, same options, same budgets, same output.
  /// The instance model is still not loaded.
  pub fn segmentation_masks_pixels<M: PersonSegmentationMask>(
    &self,
    pixels: &PixelPlane<'_>,
    options: &AppleVisionPersonMaskerOptions,
  ) -> Result<Vec<M>, AnalyzeError> {
    self.segmentation_masks_on::<M>(ImageSource::Plane(pixels), options)
  }

  /// The one segmentation-mask body both doors reach.
  fn segmentation_masks_on<M: PersonSegmentationMask>(
    &self,
    source: ImageSource<'_>,
    options: &AppleVisionPersonMaskerOptions,
  ) -> Result<Vec<M>, AnalyzeError> {
    let requests = unsafe {
      [Retained::cast_unchecked::<VNRequest>(
        self.segmentation.clone(),
      )]
    };
    run_requests(source, &requests, Vec::new(), || {
      let mut budget = MaskBudget::new();
      guard_vision_ffi("person_segmentation", Vec::new(), || {
        self.extract_segmentation::<M>(options, &mut budget)
      })
    })
  }

  fn extract_instances<M: PersonInstanceMaskDetection>(
    &self,
    options: &AppleVisionPersonMaskerOptions,
    budget: &mut MaskBudget,
  ) -> Vec<M> {
    let Some(results) = (unsafe { self.instances.results() }) else {
      return Vec::new();
    };
    let opts = options.instances();

    let mut masks = Vec::new();
    'outer: for observation in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      // The walk step's first act: ceiling test AND attempt charge,
      // before the confidence gate below can `continue` and before the
      // inner instance walk can reject an index. Visiting an
      // observation costs an FFI read whether or not it survives its
      // gates.
      if !budget.charge_walk_step() {
        break;
      }
      let Some(confidence) =
        sanitize_confidence(unsafe { observation.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      let inner_cap = opts
        .max_instances_per_observation()
        .min(MAX_NESTED_INSTANCES_PER_OBSERVATION);
      // `max_instances_per_observation` is an unbounded `usize` knob,
      // so a configured zero is a valid instruction to walk no
      // instances at all. Short-circuit BEFORE `allInstances` /
      // `firstIndex`, so a cap of zero cannot read an index it will
      // only reject. The old loop emitted nothing in this case either —
      // it broke on its first test — so the output is identical.
      if inner_cap == 0 {
        continue;
      }
      let instances = unsafe { observation.allInstances() };
      // Track ATTEMPTS (every iteration), not just successful
      // emissions — otherwise a corrupted NSIndexSet whose entries
      // all fail generation/copy/u32 conversion can drive unbounded
      // traversal at full Vision-call cost per iteration.
      //
      // The walk is bounded by `inner_cap` in the loop header, and the
      // ONE advancement site is the top of each iteration after the
      // first — so every rejection below is a plain `continue` that
      // cannot forget to advance, and no index beyond the cap is ever
      // fetched.
      let mut instance_index = instances.firstIndex();
      for visited in 0..inner_cap {
        if visited > 0 {
          instance_index = instances.indexGreaterThanIndex(instance_index);
        }
        if instance_index == NSNotFound as usize {
          break;
        }
        // Per-call budget check AND the attempt charge for this index,
        // as one step, before ANY branch that can skip the index: the
        // `u32` narrowing below, the Vision generation, and the copy
        // are all rejection paths, and each visited index costs an
        // `NSIndexSet` traversal whether or not it survives them.
        // Stops the entire extraction once any ceiling is reached
        // (success-path counters OR the failure-path attempt counter).
        if !budget.charge_walk_step() {
          break 'outer;
        }

        // Validate u32-fit of the instance index BEFORE generating
        // or copying the mask — overflowing here would force a
        // costly retry per-iteration; cheaper to skip up-front.
        let Ok(wire_instance_index) = u32::try_from(instance_index) else {
          continue;
        };

        // The attempt budget is already charged above, so an `Err` from
        // the expensive Vision call below — which still costs Vision
        // time + intermediate alloc — is budgeted.
        let selected_instances = NSIndexSet::indexSetWithIndex(instance_index);
        let Ok(mask_buffer) =
          (unsafe { observation.generateMaskForInstances_error(&selected_instances) })
        else {
          continue;
        };

        // Pre-allocation budget check: pass the remaining cumulative
        // budget into the copier so it rejects the mask BEFORE
        // allocating if the packed size would overshoot.
        let Some((bbox, width, height, data)) =
          copy_instance_mask_buffer::<M::BoundingBox>(&mask_buffer, budget.remaining_bytes())
        else {
          continue;
        };

        let data_len = data.len();
        match M::try_new(bbox, confidence, wire_instance_index, width, height, &data) {
          Ok(mask) => {
            budget.charge_emission(data_len);
            masks.push(mask);
          }
          Err(_) => {
            // Refused — `width`/`height` are already verified > 0 and
            // `data` is non-empty before reaching here, so this only
            // triggers on a vocabulary invariant the engine does not
            // model.
          }
        }
      }
    }

    masks
  }

  fn extract_segmentation<M: PersonSegmentationMask>(
    &self,
    options: &AppleVisionPersonMaskerOptions,
    budget: &mut MaskBudget,
  ) -> Vec<M> {
    let Some(results) = (unsafe { self.segmentation.results() }) else {
      return Vec::new();
    };
    let opts = options.segmentation();

    let mut masks = Vec::new();
    for observation in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      // Ceiling test AND attempt charge as one step, before the
      // confidence gate below can `continue`. The charge used to sit
      // after that gate, which left every refused observation's visit
      // uncharged against a ceiling that claimed to bound the walk.
      // Charging first also keeps the failure paths beneath it — the
      // pixel-buffer pull and the copy, both of which cost FFI
      // traversal + bounded alloc — budgeted, same policy as the
      // instance-mask extractor.
      if !budget.charge_walk_step() {
        break;
      }
      let Some(confidence) =
        sanitize_confidence(unsafe { observation.confidence() }, opts.min_confidence())
      else {
        continue;
      };

      let pixel_buffer = unsafe { observation.pixelBuffer() };
      // Pre-allocation budget check: refuse the mask before alloc
      // if it would overshoot the per-call cumulative budget.
      let Some((bbox, width, height, data)) =
        copy_instance_mask_buffer::<M::BoundingBox>(&pixel_buffer, budget.remaining_bytes())
      else {
        continue;
      };

      let data_len = data.len();
      if let Ok(mask) = M::try_new(bbox, confidence, width, height, &data) {
        budget.charge_emission(data_len);
        masks.push(mask);
      }
    }

    masks
  }
}

/// The three cumulative mask ceilings for one call: emitted count,
/// emitted payload bytes, and attempted walk steps.
///
/// Grouping them keeps the three checks impossible to spell
/// inconsistently between the two extractors — the failure that the
/// attempt counter exists to catch was originally a missing check in
/// exactly one of them.
///
/// # Attempt accounting precedes every rejection branch
///
/// An emission counter rises only on success, so it bounds what a call
/// EMITS and nothing else. The failure paths — a confidence gate, a
/// `u32` narrowing, a Vision call that returns `Err`, a copy the byte
/// budget refuses — each cost FFI traversal, and an adversarial result
/// set can reach them once per visited item. Only an ATTEMPT counter
/// bounds that, and only if it is charged BEFORE the walk can branch:
/// a charge that sits after an early `continue` is not a bound on the
/// loop, it is a bound on the loop's productive steps.
///
/// [`charge_walk_step`](Self::charge_walk_step) is therefore the ONLY
/// place the attempt counter is spent, and it fuses the ceiling test
/// with the charge into one call, so a call site cannot reach a
/// rejection branch without having paid.
#[cfg(target_vendor = "apple")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MaskBudget {
  count: usize,
  bytes: usize,
  attempts: usize,
}

#[cfg(target_vendor = "apple")]
impl MaskBudget {
  #[inline]
  pub(crate) const fn new() -> Self {
    Self {
      count: 0,
      bytes: 0,
      attempts: 0,
    }
  }

  /// Whether either emission ceiling is reached. Checked before the
  /// per-observation work begins.
  #[inline]
  pub(crate) const fn emission_exhausted(&self) -> bool {
    self.count >= MAX_TOTAL_MASKS_PER_CALL || self.bytes >= MAX_TOTAL_MASK_BYTES_PER_CALL
  }

  /// Whether any of the three ceilings is reached.
  #[inline]
  pub(crate) const fn exhausted(&self) -> bool {
    self.emission_exhausted() || self.attempts >= MAX_TOTAL_MASK_ATTEMPTS_PER_CALL
  }

  /// The payload bytes still affordable, for the pre-allocation gate.
  #[inline]
  pub(crate) const fn remaining_bytes(&self) -> usize {
    MAX_TOTAL_MASK_BYTES_PER_CALL.saturating_sub(self.bytes)
  }

  /// Gate one walk step on all three ceilings and charge the attempt
  /// budget for it, indivisibly.
  ///
  /// A step is one observation visited by either extractor, or one
  /// instance index visited inside the instance walk. `false` means a
  /// ceiling is reached and the caller stops walking; the budget is
  /// left exactly as it was, so a refusal charges nothing.
  ///
  /// The count and byte ceilings are read, never charged, here — they
  /// are emission budgets, advanced by
  /// [`charge_emission`](Self::charge_emission) on a successful push.
  #[inline]
  pub(crate) const fn charge_walk_step(&mut self) -> bool {
    if self.exhausted() {
      return false;
    }
    self.attempts = self.attempts.saturating_add(1);
    true
  }

  #[inline]
  pub(crate) const fn charge_emission(&mut self, bytes: usize) {
    self.bytes = self.bytes.saturating_add(bytes);
    self.count = self.count.saturating_add(1);
  }
}

// ----- CVPixelBuffer RAII lock ----------------------------------------------

/// RAII guard that holds a `CVPixelBufferLockBaseAddress` lock for the
/// lifetime of the guard. `Drop` unlocks even on panic-unwind so the
/// buffer cannot be left in a locked state by a panicking slice index.
#[cfg(target_vendor = "apple")]
struct CVPixelBufferLockGuard<'a> {
  buffer: &'a CVPixelBuffer,
  flags: CVPixelBufferLockFlags,
}

#[cfg(target_vendor = "apple")]
impl<'a> CVPixelBufferLockGuard<'a> {
  /// Acquire a lock on `buffer` with `flags`. Returns `None` if Core
  /// Video refused the lock; on success the guard's `Drop` is
  /// responsible for releasing it.
  #[inline]
  fn lock(buffer: &'a CVPixelBuffer, flags: CVPixelBufferLockFlags) -> Option<Self> {
    // SAFETY: `buffer` is a valid `CVPixelBuffer`; `flags` is a valid
    // `CVPixelBufferLockFlags`. The function is documented as safe to
    // call from any thread.
    let rc = unsafe { CVPixelBufferLockBaseAddress(buffer, flags) };
    if rc == kCVReturnSuccess {
      Some(Self { buffer, flags })
    } else {
      None
    }
  }

  /// Borrow the locked buffer.
  #[inline]
  fn buffer(&self) -> &CVPixelBuffer {
    self.buffer
  }
}

#[cfg(target_vendor = "apple")]
impl Drop for CVPixelBufferLockGuard<'_> {
  fn drop(&mut self) {
    // SAFETY: the corresponding lock was acquired successfully in
    // `lock`; calling unlock with matching flags is required by Core
    // Video. We ignore the return code — even if unlock fails, the
    // buffer is going away with us and there's nothing the caller can
    // do about it.
    let _ = unsafe { CVPixelBufferUnlockBaseAddress(self.buffer, self.flags) };
  }
}

/// Allocate a zero-initialised packed mask buffer with bounded
/// `try_reserve_exact`. Returns `None` on either bound violation or
/// allocator failure — both surface to the caller as a dropped mask
/// detection rather than aborting the process.
#[cfg(target_vendor = "apple")]
pub(crate) fn try_alloc_packed_mask(packed_len: usize) -> Option<Vec<u8>> {
  if packed_len > MAX_MASK_BYTES {
    return None;
  }
  let mut packed: Vec<u8> = Vec::new();
  packed.try_reserve_exact(packed_len).ok()?;
  packed.resize(packed_len, 0u8);
  Some(packed)
}

/// Validate mask dimensions BEFORE constructing the raw-parts slice
/// over a `CVPixelBuffer`'s base address. Two preconditions are
/// checked here so the unsafe `std::slice::from_raw_parts` call
/// downstream is sound even against a corrupted or adversarial
/// `CVPixelBuffer`:
///
/// 1. `width * height` (the output payload size after packing to
///    `OneComponent8`) must not exceed [`MAX_MASK_BYTES`].
/// 2. `total_src_len = bytes_per_row * height` (the raw slice
///    length) must fit in `isize::MAX`, which is the
///    [`std::slice::from_raw_parts`] contract.
///
/// Returns `None` on either violation; the caller propagates the
/// `None` so the mask detection is dropped rather than triggering
/// UB.
#[cfg(target_vendor = "apple")]
#[inline]
pub(crate) fn validate_mask_dims_for_slice(
  width: usize,
  height: usize,
  total_src_len: usize,
) -> Option<()> {
  let output_payload = width.checked_mul(height)?;
  if output_payload > MAX_MASK_BYTES {
    return None;
  }
  if total_src_len > isize::MAX as usize {
    return None;
  }
  Some(())
}

/// Copy a Vision mask `CVPixelBuffer` into a packed byte payload plus
/// a normalized bounding box of the foreground and the buffer's own
/// `(width, height)`.
///
/// The returned payload is **always** 8 bits per pixel
/// (`width * height` bytes); Vision's two supported source formats
/// (`OneComponent32Float`, `OneComponent8`) are both normalised to
/// canonical u8 at the boundary so downstream consumers don't have
/// to disambiguate from the returned dimensions alone. f32 input
/// is mapped `v` → `(v.clamp(0.0, 1.0) * 255.0).round() as u8` with
/// non-finite values collapsing to `0` (background).
///
/// Returns `None` when the buffer is unlockable, has zero extent, a null
/// base address, an unsupported pixel format, fails one of the
/// stride/size sanity checks, or contains no foreground pixels (an
/// all-zero mask is represented by skipping the detection rather than
/// emitting one with a degenerate bbox). The lock is held via
/// [`CVPixelBufferLockGuard`] for the duration of the copy and is
/// released by `Drop` on every exit path — including a panic — so the
/// buffer cannot be left locked.
#[cfg(target_vendor = "apple")]
fn copy_instance_mask_buffer<B: BoundingBox>(
  pixel_buffer: &CVPixelBuffer,
  remaining_byte_budget: usize,
) -> Option<(B, u32, u32, Vec<u8>)> {
  let guard = CVPixelBufferLockGuard::lock(pixel_buffer, CVPixelBufferLockFlags::ReadOnly)?;
  copy_instance_mask_buffer_locked(guard.buffer(), remaining_byte_budget)
}

/// Internal worker that runs the locked copy and assembles the wire
/// payload. The caller is responsible for holding the
/// [`CVPixelBufferLockGuard`].
///
/// The returned payload is **always** 8 bits per pixel
/// (`width * height` bytes) regardless of the source pixel format.
/// Vision can emit either `kCVPixelFormatType_OneComponent32Float`
/// (4 bytes/pixel) or `kCVPixelFormatType_OneComponent8`
/// (1 byte/pixel); both are normalised to the canonical u8 wire
/// representation so downstream consumers don't have to disambiguate
/// from the returned dimensions alone. The f32 → u8 quantisation
/// is `(v.clamp(0.0, 1.0) * 255.0).round() as u8` with non-finite
/// inputs collapsed to `0` (background); see
/// [`process_mask_bytes_f32`] for the per-pixel logic.
#[cfg(target_vendor = "apple")]
#[allow(non_upper_case_globals)]
fn copy_instance_mask_buffer_locked<B: BoundingBox>(
  pixel_buffer: &CVPixelBuffer,
  remaining_byte_budget: usize,
) -> Option<(B, u32, u32, Vec<u8>)> {
  let width = CVPixelBufferGetWidth(pixel_buffer);
  let height = CVPixelBufferGetHeight(pixel_buffer);
  if width == 0 || height == 0 {
    return None;
  }
  // Pre-allocation budget check: refuse to allocate this mask if
  // its packed size (`width * height` bytes) would exceed the
  // caller's remaining per-call budget. This prevents the peak
  // memory from briefly exceeding the per-call cap by one full
  // mask payload.
  let output_payload = width.checked_mul(height)?;
  if output_payload > remaining_byte_budget {
    return None;
  }

  let pixel_format = CVPixelBufferGetPixelFormatType(pixel_buffer);
  let bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
  let base_address = CVPixelBufferGetBaseAddress(pixel_buffer) as *const u8;
  if base_address.is_null() || bytes_per_row == 0 {
    return None;
  }

  // Total foreground-mask byte count cannot overflow `usize`, and the
  // stride must be wide enough to hold one row of pixels of the
  // expected size — otherwise our row-slice indexing would read past
  // the end of the buffer.
  let bytes_per_pixel: usize = match pixel_format {
    kCVPixelFormatType_OneComponent32Float => core::mem::size_of::<f32>(),
    kCVPixelFormatType_OneComponent8 => 1,
    _ => return None,
  };
  let row_pixel_bytes = width.checked_mul(bytes_per_pixel)?;
  if bytes_per_row < row_pixel_bytes {
    return None;
  }
  let total_src_len = bytes_per_row.checked_mul(height)?;

  // Pre-validate the two mask preconditions that `from_raw_parts`
  // requires (`total_src_len <= isize::MAX`) and that the bounded
  // allocator requires (`width * height <= MAX_MASK_BYTES`).
  // Centralised in `validate_mask_dims_for_slice` so a corrupted or
  // adversarial `CVPixelBuffer` cannot reach the unsafe slice with
  // values that would either trigger UB or drive the worker into
  // the allocator's abort path.
  validate_mask_dims_for_slice(width, height, total_src_len)?;

  // FFI-truth check: `total_src_len = bytes_per_row * height` is
  // derived from the buffer's own metadata, but `CVPixelBufferGetDataSize`
  // reports the actual ALLOCATED size of the buffer's data plane.
  // A malformed buffer with valid `base_address` but inconsistent
  // stride/height metadata could otherwise let `from_raw_parts`
  // create an overlong slice (UB on the row reads). Reject if the
  // computed length exceeds the buffer's reported data size.
  let data_size: usize = CVPixelBufferGetDataSize(pixel_buffer);
  if total_src_len > data_size {
    return None;
  }

  // SAFETY: `base_address` points at a buffer whose allocated size
  // is `CVPixelBufferGetDataSize(pixel_buffer)` (verified just above
  // to be at least `total_src_len`); the buffer is locked by the
  // surrounding `CVPixelBufferLockGuard`. The pre-validation
  // satisfies the `from_raw_parts` contract
  // (`total_src_len <= isize::MAX` AND
  // `total_src_len <= data_size`) regardless of what Core Video
  // reports for the dimensions; the downstream bounded allocator
  // re-checks `width * height` against `MAX_MASK_BYTES`.
  let src = unsafe { std::slice::from_raw_parts(base_address, total_src_len) };

  // The mask seat reports its dimensions as `u32`. A mask whose
  // width or height exceeds `u32::MAX` cannot be represented
  // faithfully — we'd have to saturate to a value smaller than the
  // actual packed payload, which would silently desynchronise
  // consumers that size buffers from the metadata. Reject overflow
  // here so the detection is dropped rather than poisoning storage.
  let dim_width = u32::try_from(width).ok()?;
  let dim_height = u32::try_from(height).ok()?;

  let (bbox, packed) = match pixel_format {
    kCVPixelFormatType_OneComponent32Float => {
      process_mask_bytes_f32(width, height, bytes_per_row, src)?
    }
    kCVPixelFormatType_OneComponent8 => process_mask_bytes_u8(width, height, bytes_per_row, src)?,
    _ => return None,
  };

  Some((bbox, dim_width, dim_height, packed))
}

/// Walk an `OneComponent32Float` mask, quantise each pixel to 8 bits,
/// and derive a normalized foreground bbox. Returns `None` for an
/// all-zero mask so the caller skips emitting a detection.
///
/// The result is a `(bbox, packed_bytes)` pair where `packed_bytes`
/// has length `width * height` — i.e. one **u8** per pixel, NOT four
/// `f32` little-endian bytes. Vision emits f32 mask values in
/// `[0.0, 1.0]`; we map `v` → `(v.clamp(0.0, 1.0) * 255.0).round() as
/// u8`. Non-finite values (`NaN`, `±Inf`) collapse to `0`
/// (background), matching Vision's documented "non-finite = no
/// confidence in foreground" convention and keeping the wire payload
/// canonically 8-bit per pixel across both source pixel formats.
#[cfg(target_vendor = "apple")]
pub(crate) fn process_mask_bytes_f32<B: BoundingBox>(
  width: usize,
  height: usize,
  bytes_per_row: usize,
  src: &[u8],
) -> Option<(B, Vec<u8>)> {
  let src_row_pixel_bytes = width.checked_mul(core::mem::size_of::<f32>())?;
  let packed_len = width.checked_mul(height)?;
  // Bounded allocation: cap at `MAX_MASK_BYTES` and use
  // `try_reserve_exact` so an oversized or corrupted dimensions value
  // returns `None` instead of aborting the worker process via the
  // allocator's OOM path.
  let mut packed = try_alloc_packed_mask(packed_len)?;

  let mut min_x = usize::MAX;
  let mut min_y = usize::MAX;
  let mut max_x = 0usize;
  let mut max_y = 0usize;
  let mut has_foreground = false;

  for row in 0..height {
    let src_start = row.checked_mul(bytes_per_row)?;
    let src_end = src_start.checked_add(src_row_pixel_bytes)?;
    let src_row = src.get(src_start..src_end)?;
    let dst_start = row.checked_mul(width)?;
    let dst_end = dst_start.checked_add(width)?;
    let dst_row = packed.get_mut(dst_start..dst_end)?;
    for col in 0..width {
      let pixel_start = col.checked_mul(4)?;
      let pixel_end = pixel_start.checked_add(4)?;
      let bytes: [u8; 4] = src_row.get(pixel_start..pixel_end)?.try_into().ok()?;
      let value = f32::from_le_bytes(bytes);
      // f32 mask in `[0.0, 1.0]` → u8 mask in `[0, 255]`. Non-finite
      // values (`NaN`, `±Inf`) collapse to `0` (background) — Vision
      // documents non-finite as "no confidence", which is the same
      // semantic as background in the u8 representation.
      let quantised: u8 = if value.is_finite() {
        (value.clamp(0.0, 1.0) * 255.0).round() as u8
      } else {
        0
      };
      *dst_row.get_mut(col)? = quantised;
      if quantised > 0 {
        has_foreground = true;
        min_x = min_x.min(col);
        min_y = min_y.min(row);
        max_x = max_x.max(col);
        max_y = max_y.max(row);
      }
    }
  }

  if !has_foreground {
    // All-zero mask — skip the detection rather than emit one with a
    // degenerate bbox. A validating vocabulary rejects zero-extent
    // boxes, so a `default()`-style fallback would poison downstream
    // conversion.
    return None;
  }
  let bbox = normalized_bbox_from_pixel_bounds(min_x, min_y, max_x, max_y, width, height)?;
  Some((bbox, packed))
}

/// Walk an `OneComponent8` mask, copy it tightly packed, and derive a
/// normalized foreground bbox. Returns `None` for an all-zero mask.
#[cfg(target_vendor = "apple")]
pub(crate) fn process_mask_bytes_u8<B: BoundingBox>(
  width: usize,
  height: usize,
  bytes_per_row: usize,
  src: &[u8],
) -> Option<(B, Vec<u8>)> {
  let packed_len = width.checked_mul(height)?;
  // Bounded allocation: see `process_mask_bytes_f32` for the rationale.
  let mut packed = try_alloc_packed_mask(packed_len)?;

  let mut min_x = usize::MAX;
  let mut min_y = usize::MAX;
  let mut max_x = 0usize;
  let mut max_y = 0usize;
  let mut has_foreground = false;

  for row in 0..height {
    let src_start = row.checked_mul(bytes_per_row)?;
    let src_end = src_start.checked_add(width)?;
    let src_row = src.get(src_start..src_end)?;
    let dst_start = row.checked_mul(width)?;
    let dst_end = dst_start.checked_add(width)?;
    let dst_row = packed.get_mut(dst_start..dst_end)?;
    dst_row.copy_from_slice(src_row);
    for (col, value) in dst_row.iter().copied().enumerate() {
      if value > 0 {
        has_foreground = true;
        min_x = min_x.min(col);
        min_y = min_y.min(row);
        max_x = max_x.max(col);
        max_y = max_y.max(row);
      }
    }
  }

  if !has_foreground {
    return None;
  }
  let bbox = normalized_bbox_from_pixel_bounds(min_x, min_y, max_x, max_y, width, height)?;
  Some((bbox, packed))
}

/// Convert the foreground pixel bounds of a `CVPixelBuffer` mask into a
/// normalized bounding box in the top-left convention.
///
/// `CVPixelBuffer` rows are stored top-to-bottom in memory (row 0 is the
/// top of the image), so the natural mapping `min_y / height` is already
/// top-left and no y-flip is needed here.
///
/// The intermediate division is performed in `f64` because the bounded
/// mask cap (`MAX_MASK_BYTES = 64 MiB`) admits widths above `2^24`,
/// where consecutive `usize` values round to the same `f32` (mantissa
/// exhaustion). A naive `min_x as f32 / width as f32` then produces
/// `x = 1.0` with a positive width on right-edge foreground at width
/// `2^24 + 1`, which violates the schema's `[0, 1]` invariant.
///
/// `f64` has 52 mantissa bits and represents every `usize` up to
/// `2^52` exactly on 64-bit targets; only the final narrow to `f32`
/// loses precision, which is invariant-safe because the result is
/// in `[0, 1]`. Returns `None` if the final box is refused — a
/// corrupted pixel-bound input cannot produce a box that downstream
/// storage would reject.
#[cfg(target_vendor = "apple")]
pub(crate) fn normalized_bbox_from_pixel_bounds<B: BoundingBox>(
  min_x: usize,
  min_y: usize,
  max_x: usize,
  max_y: usize,
  width: usize,
  height: usize,
) -> Option<B> {
  if width == 0 || height == 0 {
    return None;
  }
  // Compute all four EDGES in f64, then narrow to f32. The width
  // and height are derived from the narrowed edges (right - left,
  // bottom - top) rather than from a separate f64-division. This
  // guarantees the bbox is internally consistent in f32 arithmetic:
  // `left + width == right` exactly, and `right <= 1.0` by
  // construction (numerator <= denominator).
  //
  // Why edges-then-subtract instead of left-then-width: at widths
  // above the f32 mantissa exhaustion point (2^24+1, 2^25+1, …),
  // separate f32 narrowings of `min_x / width` and `(max_x + 1 - min_x)
  // / width` can land on (1.0, positive) for right-edge foreground,
  // producing `x == 1.0 && w > 0.0` — a box a validating
  // vocabulary does NOT reliably reject (an `x + w` check is also
  // f32 and rounds back to 1.0). Edge-based computation eliminates the
  // class entirely: the right edge is constructed directly as a
  // normalized value, not synthesised from `left + width`.
  let w64 = width as f64;
  let h64 = height as f64;
  // `max_x + 1` would overflow `usize::MAX`; the caller bounds
  // `max_x < width <= MAX_MASK_BYTES` (well below `usize::MAX`),
  // but use `checked_add` for defence-in-depth.
  let right_pixel = max_x.checked_add(1)?;
  let bottom_pixel = max_y.checked_add(1)?;
  if right_pixel > width || bottom_pixel > height || min_x > max_x || min_y > max_y {
    return None;
  }
  let left = (min_x as f64 / w64) as f32;
  let top = (min_y as f64 / h64) as f32;
  let right = (right_pixel as f64 / w64) as f32;
  let bottom = (bottom_pixel as f64 / h64) as f32;
  let w = right - left;
  let h = bottom - top;
  // Reject pathological f32 narrowings: a foreground bbox whose
  // left or top edge rounded to 1.0 (i.e. mantissa exhaustion
  // pushed an "almost-1.0" value over the line) is geometrically
  // a point, not a region. Drop it rather than emitting an
  // out-of-spec wire bbox.
  if !(left < 1.0 && top < 1.0) {
    return None;
  }
  if !(w > 0.0 && h > 0.0) {
    return None;
  }
  B::try_new(left, top, w, h).ok()
}

/// Non-macOS stub for [`PersonMasker`].
#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct PersonMasker;

#[cfg(not(target_vendor = "apple"))]
impl PersonMasker {
  /// Constructs a non-macOS stub masker. The options are ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionPersonMaskerOptions) -> Self {
    Self
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn instance_masks<M: PersonInstanceMaskDetection>(
    &self,
    _jpeg_data: &[u8],
    _options: &AppleVisionPersonMaskerOptions,
  ) -> Result<Vec<M>, AnalyzeError> {
    crate::error::unsupported()
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn instance_masks_pixels<M: PersonInstanceMaskDetection>(
    &self,
    _pixels: &PixelPlane<'_>,
    _options: &AppleVisionPersonMaskerOptions,
  ) -> Result<Vec<M>, AnalyzeError> {
    crate::error::unsupported()
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn segmentation_masks<M: PersonSegmentationMask>(
    &self,
    _jpeg_data: &[u8],
    _options: &AppleVisionPersonMaskerOptions,
  ) -> Result<Vec<M>, AnalyzeError> {
    crate::error::unsupported()
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn segmentation_masks_pixels<M: PersonSegmentationMask>(
    &self,
    _pixels: &PixelPlane<'_>,
    _options: &AppleVisionPersonMaskerOptions,
  ) -> Result<Vec<M>, AnalyzeError> {
    crate::error::unsupported()
  }
}
