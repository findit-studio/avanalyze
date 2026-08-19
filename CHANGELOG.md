# Changelog

## UNRELEASED

## 0.2.0

The engine stops assembling and stops depending.

### Breaking

- **`analyze_keyframe` is generic over an output vocabulary and returns
  detections, not an aggregate.**
  `analyze_keyframe(scene_id, keyframe_id, pts, dimensions, extractor, jpeg)`
  → `analyze_keyframe::<D: Detections>(jpeg, &AnalyzeOptions)`. The five
  carrier parameters were never read — they were moved verbatim into the
  aggregate the engine built — so they are gone along with the aggregate. The
  result is `Analysis<D>`: eighteen flat slots the caller composes into
  whatever record it stores.
- **New: the `Detections` contract.** One bundle trait naming nineteen output
  types, each with its own per-part trait (`BoundingBox`, `FaceDetection`,
  `DocumentSegment`, …). Every constructor seat is a primitive (`f32`, `u32`,
  `usize`, `&str`, `&[u8]`, tuples) or an engine-owned enum, so no foreign type
  crosses the boundary. Associated-type bounds make a mismatched bundle a
  compile error.
- **The three media dependencies are gone.** `mediaschema`, `mediaframe`, and
  `mediatime` leave `[dependencies]` entirely; `mediaschema` and `mediaframe`
  return as **dev**-dependencies for the reference implementation in
  `src/tests/reference.rs`. `bytes` leaves too — mask payloads cross the seam
  as `&[u8]`.
- **`AnalyzeError` replaces the borrowed error record**, with
  `AnalyzeErrorKind::{RequestFailed, Unsupported}` in place of the two
  `ErrorCode` variants. It implements `Display` and `std::error::Error`, so
  callers propagate it with `?` instead of reassembling a string from `code()`
  and `message()`.
- **`ServiceOptions` is now `AnalyzeOptions`**, passed per call rather than
  stored. `VisionAnalyzer::new` takes `&AnalyzeOptions` and consumes exactly
  one knob — `maximum_hand_count`, which Apple bakes into the retained
  request — so the analyzer holds no configuration of its own.
- **Engine-owned enums.** `Chirality` and `HeightEstimation` replace the
  imported handedness and height-estimation vocabularies.
- **`VisionAnalyzer::log_request_revisions` is public** and no longer takes
  service/worker parameters. It was dead code annotated as being called from a
  block that does not exist.

### Added

- `avanalyze::conformance`: assertion families an implementor runs to prove
  their vocabulary fits. `assert_contract` is the hard family (every value the
  engine can emit is accepted, in the argument order the engine uses);
  `assert_refuses_invalid` is the optional family for vocabularies that
  validate. The split is deliberate — the engine filters before it constructs,
  so a vocabulary that stores raw values is legal.
- `DEFAULT_*` associated constants on every options type, single-sourced with
  `new()` and with the serde defaults.
- `tests/common`: a vocabulary implemented from **outside** the crate, which is
  the only way to prove the contract is open (the orphan rule lets a crate
  implement its own traits for anyone's types, so the in-crate reference
  implementation cannot show it).

### Fixed

- **Six options types were mandatory-in-full when deserialized.** The saliency,
  horizon, document-segmentation, aesthetics, person-instance-mask and
  person-segmentation options carried no field-level `serde(default)`, so a
  config that named one knob in such a section — `{"attention_saliency":
  {"max_regions": 4}}` — failed on the field it did *not* name instead of
  filling that field in. Every field now defaults from the same function its
  `DEFAULT_*` constant reads, which is what the `DEFAULT_*` entry above already
  claimed for the whole set.
- **No panic path remains.** The two `expect`s on the horizon sentinel are
  gone: a vocabulary that refuses `try_new(0.0, 0.0)` now costs the horizon
  slot, not the worker thread.
- The keyframe-construction failure mode disappeared with the aggregate — the
  error surface is three paths on Apple targets and one off them.
- Manifest and documentation now describe the crate that exists: the git
  dependency that was never a git dependency, the re-export module that never
  existed, and the changelog that still described a 2022 template are all gone.

### Unchanged

Engine behaviour is preserved verbatim: the per-frame resource budgets, the
IoU ≥ 0.5 join between the two Vision face passes, all three coordinate
conventions, the per-field non-finite policy, the joint sort, the nested
exception barriers, the pinned request revisions, and the silent-refusal
policy.
