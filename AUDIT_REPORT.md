# Adversarial Audit Report — avanalyze

**Date:** 2026-05-27
**Auditor:** Hermes Agent (automated 30-round adversarial audit)
**Scope:** All `.rs` files (2 source files, 5,072 lines: `lib.rs` 3,758 + `options.rs` 1,280 + `build.rs` 31)
**Test suite:** 66 new audit tests + 53 existing tests = **119 total, all passing**

---

## Executive Summary

`avanalyze` is an Apple Vision.framework wrapper that analyses video keyframes and emits
`mediaschema`-shaped detections. The crate contains one `VisionAnalyzer` struct owning
19 retained `VN*Request` objects, each performing a specific Vision task (face detection,
body pose, OCR, barcode, segmentation masks, etc.).

**Overall assessment: HIGH quality with notable defensive programming.** The codebase
demonstrates institutional knowledge from prior adversarial audits (documented as
"codex R13–R17" findings). Every FFI boundary has NaN/Inf guards, bounded allocation
caps, and overflow-safe arithmetic. The main risks are in the test infrastructure and
one serde edge case, not in the core logic.

**Overall Risk Rating: LOW** — No critical or high-severity issues found.

---

## Test Results Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| `lib.rs` (existing) | 53 | ALL PASS |
| `audit_avanalyze_options.rs` | 61 | ALL PASS |
| `audit_avanalyze_api.rs` | 5 | ALL PASS |
| `audit_avanalyze_serde.rs` | — | Blocked (missing `serde_json` dev-dep) |
| `tests/foo.rs` (existing) | — | **BROKEN** (35 compile errors) |
| **Total** | **119** | **ALL PASS** |

---

## 30-Round Findings

### R1: Coverage Review

**Existing test coverage:** 53 tests in `lib.rs` covering:
- Coordinate conversion (`vision_bbox_to_schema`, `vision_point_to_schema`)
- NaN/Inf rejection at all boundaries
- Mask processing (f32 and u8 paths, padding, NaN quantization)
- Bbox normalization (mantissa exhaustion, degenerate input)
- Pose bbox derivation (single joint, vertical, horizontal, diagonal)
- Document quad validation (collapsed corners, bow-tie)
- Helper function unit tests (`finite_f32`, `try_alloc_packed_mask`, `sanitize_*`, `validate_*`)

**Untested paths:**
- All 19 `extract_*` methods (require live Vision.framework)
- `VisionRequests::new()` request construction
- `VisionRequests::perform()` batch execution
- `analyze_keyframe()` end-to-end flow
- `ffi_nsstring_to_smolstr()` (FFI-dependent)
- `extract_face_landmark_regions()` / `push_face_landmark_region()` (FFI-dependent)
- `process_mask_bytes_f32()` / `process_mask_bytes_u8()` only tested with in-crate tests, not integration tests

**Verdict:** Coverage is appropriate for a platform-specific FFI crate. Pure logic is well-tested;
FFI-dependent code requires macOS integration testing (not automatable in CI on non-Apple).

### R2: TODO/FIXME/Dead Code

**Finding: Commented-out service framework block (lines 674–920)**
~250 lines of commented code implementing a `ThreadService` pattern including `Service`,
`Request`, `Reply`, and `handle_message`. This is development scaffolding that was disabled
during the mediaschema migration. It adds noise to the codebase and should either be
re-enabled or removed.

**Severity:** LOW (code quality)

**Finding: `#[allow(dead_code)]` on `validate_raw_slice_bytes` and `apple_vision_keyframe_error`**
- `validate_raw_slice_bytes`: Documented as "retained for future FFI byte-slice surfaces".
  Currently only exercised by tests. Reasonable to keep.
- `apple_vision_keyframe_error`: Used by the commented-out service framework. Dead code
  unless the framework is re-enabled.

**Severity:** LOW (code quality)

### R3: Dead Code Analysis

See R2. No additional findings beyond the commented service framework.

### R4: cfg Conditional Coverage

**Finding: Non-macOS stub is well-implemented.**
- `VisionAnalyzer::new()` + `analyze_keyframe()` both stubbed
- Returns structured `ErrorInfo` with `ErrorCode::AppleVisionFailed`
- Error message mentions "macOS" for discoverability
- ServiceOptions stored but unused (forward-compatible)
- `apple_vision_error()` has separate signatures for apple vs non-apple

**Verdict:** Complete. The non-macOS stub covers the entire public API surface.

### R5: Feature Flag Combinations

**Finding: Serde tests blocked by missing `serde_json` dev-dependency.**
The `Cargo.toml` has `serde` as an optional dependency but no `serde_json` dev-dependency,
so serde round-trip tests cannot run. The `audit_avanalyze_serde.rs` test file is ready
but needs `serde_json = "1"` added to `[dev-dependencies]`.

**Severity:** LOW (test infrastructure)

**Finding: `tracing` feature is wired but currently unused.**
The `tracing` feature gates `tracing::info!` / `tracing::warn!` calls in:
- `log_request_revisions()` (dead code — called only from commented service framework)
- `extract_body_poses_3d` catch_unwind warning

The tracing integration is minimal and correctly gated. No behavioral difference.

**Severity:** SUGGESTION

---

### R6–R10: NaN/Inf Boundary Conditions

**Finding: Comprehensive NaN/Inf protection — EXCELLENT**

Every FFI return path is guarded:

| Guard | Location | Policy |
|-------|----------|--------|
| `clamp01()` | Coordinate conversion | `debug_assert!(is_finite)` + clamp. Non-finite caught upstream. |
| `vision_bbox_to_schema()` | All bbox conversions | Explicit `is_finite()` check on all 4 components. Returns `None` for any non-finite. |
| `vision_point_to_schema()` | All point conversions | `is_finite()` on both x and flipped_y. Returns `None`. |
| `finite_f32()` | Scores, angles, heights | Returns `Option<f32>` — `None` for non-finite. |
| `sanitize_confidence()` | All confidence checks | Triple gate: `is_finite() && [0,1] && >= min`. NaN fails all. |
| `sanitize_capture_quality()` | Face quality | Three-state: absent→0.0, finite→pass, non-finite→drop. |
| `sanitize_body_height_pair()` | 3D pose height | Couples height+estimation; non-finite forces (0.0, Unknown). |

**Verdict:** The NaN/Inf protection is exemplary. Every Vision return value passes through
at least one guard before entering the domain model. The `debug_assert!` in `clamp01`
catches regressions in debug builds while maintaining safe degradation in release.

**No issues found in R6–R10.**

---

### R11–R15: FFI Safety & Memory Safety

**Finding: `unsafe` usage is well-justified and documented.**

Counted `unsafe` blocks: ~45 across the crate. Categories:

1. **Vision request construction** (`VisionRequests::new`) — 19 `VN*Request::new()` calls,
   all within a single `unsafe` block. Each sets a pinned revision. SAFETY is inherent
   in the objc2 bindings.

2. **`Retained::cast_unchecked`** — 19 casts from concrete `VN*Request` to `VNRequest`
   for the `NSArray`. All source types are subtypes of `VNRequest` per the Vision framework
   hierarchy. Sound.

3. **Request results** — `obs.results()`, `obs.confidence()`, `obs.boundingBox()`, etc.
   All are FFI calls on Vision observation objects. Return types are Option-wrapped.

4. **`CVPixelBuffer` operations** — Lock/unlock, base address, data size, pixel format.
   All guarded by `CVPixelBufferLockGuard` RAII. Lock failure returns `None`.

5. **`std::slice::from_raw_parts`** — 2 call sites:
   - `copy_instance_mask_buffer_locked`: Pre-validated by `validate_mask_dims_for_slice`
     + `CVPixelBufferGetDataSize` cross-check. Sound.
   - `push_face_landmark_region`: Pre-validated by `validate_raw_slice_elems::<CGPoint>`
     + `region_cap <= point_count`. Sound.

6. **`objc2::msg_send!`** — 2 calls for `SimdFloat4x4 position` and `confidence` on
   3D pose points. Both return value types, not pointers.

7. **`catch_unwind(AssertUnwindSafe(...))`** — Only on `extract_body_poses_3d`. The 3D
   pose API uses `msg_send!` which can panic on unexpected Objective-C types.

**Finding: CVPixelBufferLockGuard RAII is correct.**
- Lock acquired in `lock()`, returns `None` on failure
- `Drop` calls `CVPixelBufferUnlockBaseAddress` unconditionally
- Unwind-safe: Drop runs on panic

**Finding: `from_raw_parts` pre-validation is thorough.**
- `validate_mask_dims_for_slice`: checks `width*height <= MAX_MASK_BYTES` AND `total_src_len <= isize::MAX`
- `copy_instance_mask_buffer_locked`: additionally checks `total_src_len <= CVPixelBufferGetDataSize`
- `validate_raw_slice_elems::<T>`: checks `elem_count <= max_elems` AND `elem_count * size_of::<T>() <= isize::MAX`

**Severity:** No issues found. FFI safety is exemplary.

---

### R16–R20: Numerical Stability

**Finding: Overflow-safe arithmetic throughout.**

| Pattern | Usage | Count |
|---------|-------|-------|
| `checked_mul` | width*height, bytes_per_row*height, col*4, etc. | ~20 |
| `checked_add` | src_start+width, max_x+1, slot offsets | ~15 |
| `saturating_add` | mask budgets, landmark attempts | ~10 |
| `u32::try_from` | mask dimensions, instance indices | 3 |
| f64 intermediate | `normalized_bbox_from_pixel_bounds` | 1 (critical) |

**Finding: f64 intermediate in `normalized_bbox_from_pixel_bounds` is correct and necessary.**
At widths above 2^24, consecutive `usize` values round to the same `f32`. The edge-based
computation (`right - left` after both narrow to f32) eliminates the class where
`x == 1.0 && width > 0.0`. Explicit guard: `left < 1.0 && top < 1.0`.

**Finding: `process_mask_bytes_f32` uses `get()` instead of direct indexing.**
Every pixel access uses `src_row.get(pixel_start..pixel_end)?` which returns `None` on
out-of-bounds instead of panicking. The `?` propagates to the caller. This is correct
defensive programming against a corrupted `CVPixelBuffer`.

**No issues found in R16–R20.**

---

### R21–R25: Defensive Caps & OOM Protection

**Finding: 17 independent ceiling constants — COMPREHENSIVE**

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_MASK_BYTES` | 64 MiB | Per-mask allocation cap |
| `MAX_VISION_RESULTS_PER_FRAME` | 4,096 | Per-extractor observation cap |
| `MAX_LANDMARK_POINTS` | 1,024 | Per-region point cap |
| `MAX_POSE_JOINTS` | 256 | Per-pose joint dictionary cap |
| `MAX_NESTED_INSTANCES_PER_OBSERVATION` | 64 | Per-observation instance cap |
| `MAX_NESTED_LABELS_PER_OBSERVATION` | 32 | Per-observation label cap |
| `MAX_TEXT_CANDIDATES_PER_OBSERVATION` | 10 | Apple's documented limit |
| `MAX_SALIENCY_REGIONS_PER_FRAME` | 64 | Per-frame saliency cap |
| `MAX_TOTAL_MASKS_PER_FRAME` | 256 | Cross-extractor mask count |
| `MAX_TOTAL_MASK_BYTES_PER_FRAME` | 256 MiB | Cross-extractor byte budget |
| `MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME` | 1,024 | Failure-path attempt budget |
| `MAX_FACE_LANDMARK_POINTS_PER_FRAME` | 16,384 | Cross-detection point budget |
| `MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME` | 65,536 | Failure-path landmark budget |
| `MAX_TOTAL_ANIMAL_SUBJECTS_PER_FRAME` | 256 | Per-frame animal cap |
| `MAX_TOTAL_TEXT_DETECTIONS_PER_FRAME` | 256 | Per-frame text cap |
| `MAX_INPUT_IMAGE_BYTES` | 64 MiB | Input payload cap |
| `MAX_HAND_POSE_MAXIMUM_HAND_COUNT` | 6 | Apple's documented limit |
| `MAX_FFI_STRING_BYTES` | 4,096 | NSString conversion cap |

**Finding: Budget cascading is correct for mask extractors.**
Both `extract_person_instance_masks` and `extract_person_segmentation_masks` share the
SAME `mask_total_bytes`, `mask_total_count`, and `mask_total_attempts` mutable references.
The cumulative cap holds across both extractors, preventing a worst-case 2× overshoot.

**Finding: Attempt budget covers failure paths.**
`MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME` bounds the total `generateMaskForInstances_error` calls
even when the success-path counters (`mask_total_count`, `mask_total_bytes`) stay below
their caps. This prevents a corrupted `NSIndexSet` from driving unbounded Vision calls.

**Finding: `effective_results_cap` correctly composes user + hard ceiling.**
`user_max.min(MAX_VISION_RESULTS_PER_FRAME)` is used for `with_capacity`, `.take()`, and
the in-loop `if len >= cap { break }` guard — all three bound to the SAME value.

**Finding: `try_alloc_packed_mask` uses `try_reserve_exact`.**
Returns `None` on either bound violation or allocator failure. Zero-init via `resize()`.
No panic path.

**No issues found in R21–R25.**

---

### R26–R27: Options & API Design

**Finding: All 18 Options structs follow a consistent pattern.**
- `const fn new()` with documented defaults
- `const fn with_*()` builder (returns new value)
- `const fn set_*()` setter (returns `&mut Self`)
- `const fn *()` accessor
- `#[derive(Debug, Clone, Copy)]`
- Optional serde support via `#[cfg_attr(feature = "serde", ...)]`

**Finding: `num_workers = 0` coercion is inconsistent.**
`set_workers(0)` and `with_workers(0)` coerce to 1. But serde deserialization of
`{"num_workers": 0}` produces 0 (bypasses the coercion). This means a config file
with `"num_workers": 0` would create a 0-worker service, which contradicts the API's
intentional coercion.

**Severity:** LOW (logic inconsistency)

**Finding: Options are passive data — no validation.**
Options structs accept any `f32` (including NaN, negative, >1.0) and any `usize`.
Sanitization happens at the extractor level. This is a deliberate design choice
(separation of config from validation) but means a misconfigured option (e.g.,
`min_confidence: -1.0`) silently passes through to the extractor where it has no
effect (since `sanitize_confidence` checks `value >= min` and `-1.0 >= -1.0` is true,
but the `[0,1]` range check catches it). No real issue in practice.

**Severity:** SUGGESTION (consider documenting this design choice)

**Finding: `ServiceOptions::set_workers` uses `if == 0 { 1 } else { n }` pattern.**
This is correct but does not prevent `usize::MAX` workers. In practice, the OS thread
limit would catch this, but a `min(MAX_WORKERS, n)` would be more defensive.

**Severity:** SUGGESTION

---

### R28: Thread Safety

**Finding: `VisionAnalyzer` is `!Clone` — intentional.**
The doc comment explicitly states: "Clone is intentionally not implemented to make
that contract a compile-time error." Each worker thread must own its own analyzer
because `Retained<VN*Request>` carries per-call state.

**Finding: `ServiceOptions` is `Send` (verified by test).**
Options are plain data with no interior mutability. Safe to cross thread boundaries.

**Finding: `VisionAnalyzer` is `Send` on macOS (contains `Retained` which is `Send`).**
Safe to move to a worker thread. Not `Sync` (correct — single-consumer).

**No issues found.**

---

### R29: catch_unwind Analysis

**Finding: `catch_unwind` only on `extract_body_poses_3d` (line 1611).**
This extractor uses `objc2::msg_send![point, confidence]` which is a raw ObjC message
send that can panic on unexpected types. The `catch_unwind` catches panics from the
3D pose API and returns an empty result.

**Finding: Other extractors do NOT use `catch_unwind`.**
All other extractors use typed objc2 bindings (e.g., `obs.confidence()` returns
`Option<f32>`) which handle errors gracefully. The 3D pose path is unique in using
raw `msg_send!`. This is consistent and correct.

**Finding: `AssertUnwindSafe` is justified.**
The closure captures `&self` (immutable borrow) and local variables. No shared mutable
state could be corrupted by a panic.

**No issues found.**

---

### R30: Public API Consistency

**Finding: macOS and non-macOS `analyze_keyframe` signatures differ.**
- macOS: `pub fn analyze_keyframe(&self, scene_id: Id, keyframe_id: Id, pts: Timestamp, dimensions: Dimensions, extractor: KeyframeExtractor, jpeg_data: &[u8]) -> Result<Keyframe, ErrorInfo>`
- Non-macOS: Same signature but `Id` is `Uuid7` (via `type Id = Uuid7` on macOS, direct on non-macOS)

Both resolve to the same types. No API mismatch.

**Finding: `#![deny(missing_docs)]` enforced crate-wide.**
All public items have documentation. Internal helpers have `///` doc comments explaining
their purpose and safety considerations. This is excellent.

**Finding: `#[non_exhaustive]` not used on `ServiceOptions`.**
Since `ServiceOptions` is a struct (not an enum), adding fields is a breaking change
unless it has `#[non_exhaustive]`. However, the struct is constructed via `new()` and
builder pattern, so new fields with defaults are backward-compatible in practice.

**Severity:** SUGGESTION

**Finding: Missing `#[must_use]` on builder methods.**
`with_*()` methods return a new value — `#[must_use]` would catch accidental discards.
The `set_*()` methods return `&mut Self` for chaining — also candidates for `#[must_use]`.

**Severity:** SUGGESTION

---

## Pre-existing Issues (Not Introduced by This Audit)

### tests/foo.rs is BROKEN

The file `tests/foo.rs` has **35 compilation errors** against the current `mediaschema` API:
- `humans()` should be `humans_ref()` (API renamed)
- `ErrorInfo` doesn't implement `Display`
- `classifications()` method not found on `Keyframe`
- Various type mismatches

This file appears to be a placeholder/example that was never updated after the mediaschema
domain migration. It blocks `cargo test` from running without `--test` filtering.

**Severity:** MEDIUM (blocks default `cargo test`)

**Recommendation:** Either fix or delete `tests/foo.rs`. If it's meant as documentation,
move it to `examples/`.

---

## Findings Summary

| # | Severity | Category | Description | Action |
|---|----------|----------|-------------|--------|
| 1 | MEDIUM | Test Infrastructure | `tests/foo.rs` has 35 compile errors — blocks `cargo test` | Fix or delete |
| 2 | LOW | Test Infrastructure | Missing `serde_json` dev-dependency blocks serde round-trip tests | Add `serde_json = "1"` to `[dev-dependencies]` |
| 3 | LOW | Logic | `serde` deserialization of `num_workers: 0` bypasses the 0→1 coercion | Apply coercion in `Deserialize` or add validation |
| 4 | LOW | Code Quality | ~250 lines of commented-out service framework (lines 674–920) | Remove or re-enable |
| 5 | LOW | Code Quality | `apple_vision_keyframe_error` is dead code (used only by commented framework) | Remove with the framework |
| 6 | SUGGESTION | API Design | Options accept any f32/usize without validation (by design) | Document the design choice |
| 7 | SUGGESTION | API Design | Missing `#[must_use]` on builder/setter methods | Add `#[must_use]` |
| 8 | SUGGESTION | API Design | `ServiceOptions` lacks `#[non_exhaustive]` | Consider adding for forward-compat |
| 9 | SUGGESTION | API Design | `set_workers(usize::MAX)` not capped | Add reasonable upper bound |
| 10 | SUGGESTION | Code Quality | `tracing` feature only used in 2 places (1 dead code) | Evaluate if feature is worth maintaining |

---

## Positive Findings (What's Done Well)

1. **NaN/Inf protection is exemplary** — every FFI return path has at least one guard.
   The `debug_assert!` in `clamp01` catches regressions in debug builds without
   changing release behavior.

2. **Bounded allocation everywhere** — `try_reserve_exact`, `try_alloc_packed_mask`,
   `with_capacity(cap)` + `.take(cap)` + `if len >= cap { break }` triple-guard pattern.

3. **17 independent ceiling constants** — each with clear documentation of the
   adversarial scenario it defends against (e.g., "codex R13 F2", "codex R15").

4. **Budget cascading for masks** — instance masks and segmentation masks share
   the SAME per-frame budget (count + bytes + attempts), preventing 2× overshoot.

5. **Attempt budgets cover failure paths** — `MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME` and
   `MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME` bound the work done even when generation
   fails, preventing unbounded Vision calls on corrupted data.

6. **f64 intermediate for mask bbox normalization** — correctly handles widths above
   2^24 where f32 mantissa exhaustion would produce invalid bboxes.

7. **CVPixelBufferLockGuard RAII** — unwind-safe lock management. Drop releases even
   on panic.

8. **Consistent Options API** — 18 structs all follow the same builder/setter/accessor
   pattern with `const fn` where possible.

9. **Non-macOS stub** — the crate compiles on all platforms with a clear error message.

10. **`#![deny(missing_docs)]`** — all public items documented.

---

## Files Created

1. `/Users/joe/dev/avanalyze/tests/audit_avanalyze_options.rs` — 61 tests (Options + ServiceOptions)
2. `/Users/joe/dev/avanalyze/tests/audit_avanalyze_api.rs` — 5 tests (Public API + non-macOS stub)
3. `/Users/joe/dev/avanalyze/tests/audit_avanalyze_serde.rs` — Serde round-trip tests (blocked by missing dev-dep)
4. `/Users/joe/dev/avanalyze/AUDIT_REPORT.md` — This report

---

## Conclusion

`avanalyze` is production-quality code with exceptional defensive programming for an
FFI-heavy crate. The NaN/Inf guards, bounded allocation, and budget cascading are
the best I've seen in a Vision.framework wrapper. The only actionable findings are
infrastructure issues (broken `foo.rs`, missing serde dev-dep) and minor API polish
(`#[must_use]`, serde coercion consistency). The core logic is sound.
