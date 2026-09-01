# Changelog

## 0.5.0 — 2026-09-02

The one 18-seat bundle becomes nine entry points, each owning exactly its own
Vision requests.

### Breaking

- **`VisionAnalyzer` keeps eight detections and loses ten.** It now owns eight
  Vision requests — classification, human rectangles, animal recognition, both
  saliency passes, horizon, document segmentation, aesthetics — and
  `analyze_keyframe` performs only those. Text, barcodes, faces, landmarks,
  poses and masks each moved to their own entry point. The old shape charged
  every consumer for every capability: asking for one text run constructed
  nineteen requests and ran nineteen models. A consumer of one capability now
  constructs one entry point, loads one model, and names one output type.
- **`Analysis` shrinks from eighteen slots to eight** —
  `classifications`, `human_subjects`, `animal_subjects`,
  `attention_saliency`, `objectness_saliency`, `document_segments`, `horizon`,
  `aesthetics`. The `faces`, `face_landmarks`, `body_poses`, `hand_poses`,
  `body_poses_3d`, `person_instance_masks`, `person_segmentation_masks`,
  `animal_body_poses`, `text_detections` and `barcodes` slots are gone; their
  entry points return a plain `Vec` instead.
- **The `Detections` bundle names seven associated types, not twenty-one** —
  `BoundingBox`, `Detection`, `SubjectDetection`, `SaliencyRegion`,
  `HorizonInfo`, `DocumentSegment`, `Aesthetics`. Every other trait is now
  reached directly, as a single generic parameter on the entry point that
  builds it, and lives in that entry point's module.
- **New entry points**, each with `new(&Options)` plus its own detection
  method, each returning `Result<Vec<_>, AnalyzeError>`:
  `TextRecognizer::recognize`, `BarcodeDetector::detect`,
  `FaceDetector::detect`, `FaceLandmarker::detect`, `BodyPoser::detect_2d` /
  `detect_3d`, `HandPoser::detect`, `AnimalPoser::detect`,
  `PersonMasker::instance_masks` / `segmentation_masks`.
- **`TextDetection::try_new` gains `observation: usize, rank: usize`.** One
  Vision observation is one text region and can yield several candidate
  readings of it, every one re-using the observation's box — so the box alone
  could not tell two readings of one region from two overlapping regions. Both
  indices were already in the engine's candidate loop and were being discarded
  at the seam; they are now threaded out. `observation` indexes the observation
  within the call's results, `rank` the candidate within that observation
  (`0` = Vision's best). A consumer that keeps only `rank == 0` gets one row per
  region; one that keeps them all can rank, diff or vote across readings without
  inventing an identity of its own. Both are per call and mean nothing across
  calls.
- **`FaceDetector` fuses three passes into one record per face**, and
  `FaceDetection::try_new` gains `keypoints: Option<FaceKeypoints>`. The
  face-rectangles pass is the detection spine; its observations are handed to
  the capture-quality and landmarks requests, which return them enriched with a
  quality reading and with the 76→5-point reduction. `keypoints` is `None` where
  Vision returned no landmark set for that face or the engine could not complete
  the reduction — never an origin-valued placeholder. How a reading reaches a
  face changed entirely — see **Behaviour** below.
- **New `FaceKeypoints`**, an engine-owned five-point set in image-normalized,
  top-left coordinates. `left_eye` / `right_eye` are the `leftPupil` /
  `rightPupil` centroid when Vision reports one, else the eye contour's;
  `nose_tip` is the `noseCrest` point (or `nose`, as fallback) farthest from the
  eye midpoint, which is the tip by construction and does not depend on
  Vision's undocumented point ordering; `mouth_left` / `mouth_right` are the
  `outerLips` contour's minimum-x and maximum-x points, ties broken on `y`.
  All five must be derivable or the whole set is `None`. `points()` returns them
  in canonical alignment order. Cropping, alignment and embedding stay the
  caller's business — the engine produces geometry and never an identity.
- **`AnalyzeOptions` keeps eight sections and loses eleven.** `face_capture`,
  `face_rectangles`, `face_landmarks`, `text`, `body_pose`, `hand_pose`,
  `animal_pose`, `body_pose_3d`, `barcodes`, `person_instance_masks` and
  `person_segmentation_masks` moved to the options type of the entry point that
  reads them. The per-request options structs themselves are unchanged.
  **Migrating a serialized config:** one that still names the eleven parses as
  before and those keys are ignored, exactly as any other unknown key is on
  every options type in this crate — so their values must be moved to the new
  options types by hand, or the entry points that read them run on defaults.
  The compile-time signal is the API break: all eleven accessors (`text()`,
  `face_capture()`, …) are gone, so code that read one fails to build and names
  the section it lost.
- **New composed options**, following the existing section-per-subsystem idiom:
  `AppleVisionFaceOptions` (`rectangles` / `capture` / `keypoints`),
  `AppleVisionBodyPoserOptions` (`pose_2d` / `pose_3d`),
  `AppleVisionPersonMaskerOptions` (`instances` / `segmentation`), plus
  `AppleVisionFaceKeypointsOptions` for the reduction's own confidence floor.
- **`conformance` is recast per entry point.** `assert_contract` and
  `assert_refuses_invalid` now cover the core bundle only; every other
  capability has its own assertion over its own single type
  (`assert_text_accepts`, `assert_face_accepts`, `assert_body_pose_accepts`,
  `assert_person_instance_mask_refusals`, …). Coverage does not shrink — the
  old bundle-wide assertions are redistributed, and the text provenance pair,
  the keypoint reduction, and the per-entry non-macOS stubs are new.

### Behaviour

- **The four pose extractors carry cumulative per-call joint, attempt and
  name-byte budgets.** 2-D body, 3-D body, hand and animal pose each cap the
  joints one call may walk, the joints it may emit, and the joint-name bytes it
  may retain, summed across every observation in the frame — where before only
  the per-observation joint dictionary was capped, and the two caps composed
  into 4096 observations × 256 joints of names. A pose that cannot fit the
  remainder is dropped WHOLE and extraction stops there; no pose is truncated to
  fit. Hardening of pre-existing behaviour: a real frame is a couple of subjects
  at ~20 joints each, orders of magnitude below every one of the three
  ceilings.
- **The per-frame mask budgets are now per call.** `instance_masks` and
  `segmentation_masks` are separate Vision passes and each charges its own
  count / cumulative-bytes / generation-attempt ceiling, where the single-pass
  engine shared one budget across both mask surfaces. That is strictly more
  permissive; a caller needing the old cumulative ceiling imposes it above this
  crate.
- **Every entry point enforces the 64 MiB input ceiling** and the Objective-C
  exception barrier, not just the analyzer, and each gained a
  `log_request_revisions` diagnostic (feature `tracing`) for the requests it
  owns.
- **A face's readings reach it by observation identity, and the overlap join is
  gone.** The face-rectangles pass now runs FIRST and alone; its observations are
  then set as the `inputFaceObservations` of the capture-quality and landmarks
  requests, through Vision's own `VNFaceObservationAccepting` protocol. Those
  requests process exactly the faces they were handed and return them enriched,
  each carrying the `VNObservation.uuid` it was given — and a reading is seated
  by that uuid, so it arrives at the face Vision computed it for because the
  observation carrying it names that face. Attribution is a property of the
  mechanism rather than a conclusion drawn from geometry. Deleted with the join:
  the IoU ≥ 0.5 match floor, the global one-to-one assignment and its 65536-pair
  ceiling, and the engine-internal join rectangle the assignment ran on. Argmax
  overlap was never an ownership invariant — an annotating observation restricted
  to its maximum-IoU spine face need not belong to that face, and no floor could
  make it — so a face could wear a neighbour's `capture_quality` or `keypoints`
  with nothing downstream able to tell. **This can change which readings a face
  receives relative to 0.4.0**, in the direction of correctness: a face now wears
  the reading Vision computed for it, or none.
- **The handoff preserves observation identities, not their order — so the
  correspondence is keyed and verified, never assumed.** `VNFaceObservationAccepting`
  returns the same SET of observations, in an order of Vision's own choosing that
  varies from run to run on one unchanged image. Measured on this host across two
  independent 30-run samples of a 3-face frame: the returned uuid set matched the
  spine 30/30 both times, while the returned ORDER matched it in only 14/30 and
  15/30 of the capture-quality runs and 12/30 and 15/30 of the landmarks runs, with
  all six permutations of three elements observed. Reading either pass by array position
  would have mis-seated roughly half of all multi-face frames — every face carrying
  a real reading computed for one of its neighbours. The engine resolves each pass's
  uuids onto spine positions instead, refuses the pass unless that resolution is a
  bijection, and reduces in SPINE order so that which face loses its reduction under
  the per-frame landmark budgets depends on the image alone, not on the order Vision
  happened to return.
- **A face detection is two `performRequests` on one image, not one.** The spine
  must exist before the other two requests can be fed, so the three face passes
  no longer run in a single batch. The two annotating requests still run
  together, in one perform.
- **An annotating face pass annotates only when its results correspond to the
  spine.** Each pass is refused independently on four conditions: no results
  array, a length that differs from the spine's, a returned observation whose
  uuid string could not be read, or a uuid set that does not resolve one-to-one
  onto the spine's. Any of the four makes that ONE pass absent for every face;
  the other still annotates. With the default positive `min_capture_quality` a
  frame whose quality pass did not correspond therefore yields no faces at all;
  a caller who wants the faces anyway sets `min_capture_quality` to 0.0 and gets
  them with both annotations absent. The returned observation's `boundingBox` is
  **not** compared: the uuid is the identity token, so a box check on top of it
  is redundant where Vision returns the box unchanged (measured bit-identical,
  60/60 pass-reads) and wrongly fail-closed if a future Vision refined a box it
  re-examined.
- **A caught Objective-C exception can no longer replay an earlier frame.** The
  exception barrier turned a raising perform into a bare `Ok`, so a caller could
  not tell a completed perform from an aborted one and read `results` off the
  request either way. A `VNRequest` is retained across calls and Vision gives no
  clearing postcondition for a request it did not process, so those `results`
  may still describe the last SUCCESSFUL call — and in the face fusion that
  would have made the previous frame's faces this frame's spine, handed verbatim
  to the annotating passes, which preserve their uuids: the identity check would
  have seen a valid bijection because the entire identity universe was stale
  together. A perform now reports whether it ran, and request state is read on
  exactly one condition — this call's own perform completed. A raise costs THIS
  frame's detections (an empty `Vec`) or THIS frame's face annotations (absent at
  every seat), which is what the barrier always claimed it cost. Two further
  cases in the same class: `FaceDetector::detect` performs the annotating passes
  only when BOTH took this call's input observations, because a face request
  performed with a nil `inputFaceObservations` runs its own face detection and
  returns faces the spine never saw; and the two input clears are now
  independently guarded, so one raising cannot leave the other request holding
  this frame's observations into the next call.
- **The four pose joint-dictionary readers no longer go through
  `NSDictionary::to_vecs()`.** The 2-D body, 3-D body, hand and animal paths now
  enumerate the joint dictionary bounded — `keys()` taken at one past
  `MAX_POSE_JOINTS` — and pair each joint name with the value found under it by
  keyed lookup, refusing the pose when the dictionary's self-reported count
  disagrees with what it actually enumerates. `to_vecs()` presents a safe
  surface over an unsafe bulk copy: it allocates two vectors sized to the
  FFI-reported `count`, fills them with the deprecated unbounded
  `getObjects:andKeys:` — which Apple's own header calls out as unsafe because
  it can overrun — and then `set_len`s both to that same count, so a malformed
  Vision dictionary could write past the allocations or expose uninitialised
  pointers before any Rust-side check could reject it. A joint-count guard in
  front of the call could not help, because it read the very number that was not
  to be believed. Keyed lookup also retires the parallel-array assumption the
  old `zip` carried. This is a hardening of pre-existing behaviour, not a
  contract change: a sound dictionary reads exactly as before, and over the cap
  the pose is still DROPPED rather than truncated.


Attempt accounting now precedes every rejection branch it guards.

### Fixed

- **The per-frame mask attempt budget is charged at each walk step's entry**,
  not after the gates that can skip the step. The charge used to sit below the
  `u32::try_from` narrowing in the instance walk and below the confidence gate
  in both mask extractors, so an adversarial result set bought unmetered work
  under a ceiling that claimed to bound it: up to
  `MAX_NESTED_INSTANCES_PER_OBSERVATION` (64) index visits ×
  `MAX_VISION_RESULTS_PER_FRAME` (4096) observations = 262,144 `NSIndexSet`
  traversals, and 4096 observation visits per extractor, all against a 1,024
  ceiling. `MAX_TOTAL_MASK_ATTEMPTS_PER_FRAME` now bounds the walk itself.
- **A face-landmark region visit is charged one unit before it can be
  refused.** A region Vision did not report, an empty region, an over-cap
  `pointCount`, and a null point buffer each returned without charging, so 13
  named regions × 4096 observations = 53,248 region visits moved neither
  budget. That total stayed under `MAX_FACE_LANDMARK_ATTEMPTS_PER_FRAME` only
  by arithmetic accident; the ceiling now bounds the walk by construction. The
  visit unit is a floor on a region refused before it walks anything, never a
  surcharge on one that walks: the point walk is sized against the budget as it
  stood before the visit and only the balance is charged, so a region that
  walks costs exactly the points it walks and the frame's point cap falls
  exactly where it fell before.
- **A configured `max_instances_per_observation` of zero no longer reads an
  instance index it will only reject.** The instance walk short-circuits before
  `allInstances` / `firstIndex`, and its single advancement site is now the top
  of each iteration, so no index beyond the per-observation cap is ever
  fetched.
- **A pose joint entry is charged as it is walked, inside the joint dictionary
  reader.** The charge used to be one bulk `charge_attempts(pairs.len())` after
  the read had returned, so it never ran on the read's own rejection paths: a
  dictionary reporting `MAX_POSE_JOINTS` (256) entries and enumerating one
  fewer was allocated for, enumerated, and keyed-looked-up entry by entry, and
  the refusal that followed cost nothing — 256 entry walks × 4096 observations
  = 1,048,576 walks against an 8192-attempt ceiling that never moved. The
  reader now takes the budget and charges one unit per entry, before that
  entry's keyed lookup, and reports budget exhaustion distinctly from
  malformation: a malformed dictionary drops its own pose and the frame
  continues, an exhausted budget stops the extraction. The
  `reported > max_entries` refusal is decided before the enumeration begins, so
  it still walks nothing and charges nothing.

These are resource-accounting fixes: for conforming Vision output the emitted
detections, the emission budgets, and every `try_new` call are unchanged — a
real frame reaches neither ceiling, and a conforming joint dictionary costs
exactly the entries it enumerates, which is what the bulk charge took. What
changes is that a corrupted or adversarial observation set can no longer run a
rejection path for free.

- **A JPEG's SOF marker can no longer buy an oversized Vision decode.**
  `MAX_INPUT_IMAGE_BYTES` (R18) already caps the *compressed* input every
  entry point accepts, but a small JPEG could still declare gigantic
  *decoded* dimensions in its SOF marker — Vision / ImageIO allocates
  buffers proportional to `width × height` once it decodes the frame,
  before any downstream cap in this crate runs. `with_image` — the one
  door every entry point's `NSData::with_bytes` goes through — now walks
  the marker segments, allocation-free, to the first SOF and refuses
  anything declaring more than `MAX_DECODED_IMAGE_BYTES` (512 MiB) once
  decoded. The walk is defensive against truncated input, a missing SOF,
  and a forged length field: every exit is a structured refusal, never a
  panic or an out-of-bounds read. The byte budget is precision-aware —
  SOF1/SOF3 and the other extended/lossless markers permit sample
  precision above 8 bits, and ImageIO decodes those at double the
  baseline byte rate (confirmed against real ImageIO), so any precision
  above 8 charges the wider rate rather than under-counting by 2×. A SOF
  declaring a zero width or height is refused outright rather than
  costed at zero decoded bytes: JPEG permits a baseline SOF to defer its
  real height to a later DNL marker, which this preflight — stopping at
  the first SOF, never reading into entropy-coded scan data — cannot see,
  so an unreadable dimension is refused rather than treated as small. A
  DHP marker (hierarchical JPEG, ITU-T T.81 Annex J) is refused outright
  for the same reason: DHP declares the *completed* hierarchical image's
  dimensions ahead of a sequence of SOF-delimited frames whose first
  member can be small, so trusting only the first SOF once a DHP has been
  seen would miss the real, larger completed size. Tracked as #2, the one
  item PR #1 (R18) deferred.

### Internal

- `MaskBudget::charge_walk_step`, `charge_landmark_region_visit`,
  `charge_landmark_points` and `PoseBudget::charge_joint_visit` are the only
  places any attempt budget is charged. Each fuses its ceiling test with its
  charge, so a walk step cannot reach a rejection branch without having paid,
  and a refusal charges nothing.
- The pose joint-dictionary reader is `read_pose_joints` (was
  `bounded_dictionary_pairs`) and returns `PoseJoints::{Read, Malformed,
  Exhausted}` rather than an `Option`, because the two refusals mean opposite
  things to the caller: skip this observation, or stop the frame.

**Semver.** This breaks a published API — 0.4.0 is on crates.io — so it rides
the next breaking line (0.5.0). The version is not bumped here; the release
stamp is its own commit, matching the 0.3.0 and 0.4.0 precedent.

## 0.4.0 — 2026-08-22

Face-pose angles and capture quality stop pretending "not measured" is
"measured at zero."

### Breaking

- **`FaceDetection::try_new`'s `roll` / `yaw` / `pitch` become `Option<f32>`.**
  Vision reports each pose angle independently and optionally — availability
  differs by OS version and by detection path, and some faces arrive with
  none of the three computed. The engine previously collapsed that absence
  with `unwrap_or(0.0)`, so "Vision never computed this angle" and "Vision
  measured a level head" were both written as `0.0` and could not be told
  apart downstream — a consumer filtering "pitched down more than 20°"
  wrongly excluded a 30°-down face whose pitch was simply never computed.
  `Some(0.0)` now means a head Vision actually measured as level; `None`
  means Vision did not compute that angle.
- `extract_faces` (the Vision face-rectangles pass) carries the absence
  through to the contract seat instead of defaulting it away.
- `conformance::assert_faces_accept` follows the seat: it passes
  `Option<f32>` for every angle and gains two canonical cases — a face with
  no angles computed at all, and a face mixing a genuinely-level `Some(0.0)`
  roll with an absent yaw on the same detection.
- `tests/common::Face`, the outside vocabulary, stores the three angles as
  `Option<f32>` and round-trips the absence. `src/tests/reference.rs`'s
  `mediaschema`-backed vocabulary cannot: `mediaschema` 0.2.1 has no seat for
  the absence yet, so that one adapter collapses `None` to `0.0` at the
  boundary — a limitation of the pinned dependency, not a reopening of this
  crate's loss. `mediaschema` / `mediagraph` adopt the `Option` shape in
  their own knife once this publishes.
- **`FaceDetection::try_new`'s `capture_quality` becomes `Option<f32>`** —
  the same collapse, found in the same #18 census, fixed in the same window
  (#20). `sanitize_capture_quality` no longer maps a nil Vision reading to
  `Some(0.0)`, and `matched_capture_quality` no longer annotates a face the
  capture-quality pass never covered with `0.0`; both now produce `None`.
  `Some(0.0)` means Vision measured this face's capture quality and found it
  terrible; `None` means Vision never measured it, whether the raw reading
  was nil or no capture-quality observation overlapped the face's box (a
  join-miss). The two were previously indistinguishable downstream — a
  consumer filtering "quality below X" could not tell a genuinely
  zero-scored face from one the quality pass skipped entirely.
- `extract_faces`'s `min_capture_quality` filter keeps its prior observable
  behaviour: an unmeasured face still compares as `0.0` against the
  threshold, so the default (0.1) still drops it and `min_capture_quality
  == 0.0` still keeps it. What changes is what a face that DOES clear the
  filter unmeasured now carries to the contract seat: `None`, not
  `Some(0.0)`.
- `conformance::assert_faces_accept` follows the seat: it passes
  `Option<f32>` for capture quality and gains an unmatched (`None`) case
  plus a mixed case — a measured-and-terrible (`Some(0.0)`) face and an
  unmatched (`None`) face built in the same detection set, proving the two
  remain independently representable.
- `tests/common::Face` stores `capture_quality` as `Option<f32>` and
  round-trips the absence; `src/tests/reference.rs`'s `mediaschema`-backed
  adapter collapses `None` to `0.0` at the same pinned-dependency boundary
  as the angles, for the same reason.

**Semver.** This breaks a published API — 0.3.0 is on crates.io — so it
rides the next breaking line (0.4.0). The version is not bumped here; the
release stamp is its own commit, matching the 0.3.0 precedent.

## 0.3.0 — 2026-08-22

The joint seats stop pretending the four skeletons are one.

### Breaking

- **The `Detections` bundle names four joint types where it named two.**
  `BodyPoseJoint` and `BodyPose3DJoint` give way to `BodyJoint`, `HandJoint`,
  `AnimalJoint` and `Body3Joint`, and every pose seat binds its own: body poses
  collect `BodyJoint`, hand poses `HandJoint`, animal poses `AnimalJoint`, 3-D
  poses `Body3Joint`. The old single 2-D seat asserted an identity that does not
  exist — Apple names a different joint roster for each skeleton — and so forced
  a vocabulary that models them as three types to collapse them into one. The
  compile-time same-source guarantee is untouched *within* a skeleton family and
  gone across families, which is where it never belonged.
- **New bundle seat: `AnimalPoseDetection`.** Animal 2-D poses had been reusing
  `BodyPoseDetection`'s type; they get their own seat over the same
  `BodyPoseDetection` trait — one trait because the payload is one, two seats
  because the joints are two. `Analysis::animal_body_poses` and its four
  accessors are typed `D::AnimalPoseDetection`.
- **`Detections` therefore names twenty-two output types, up from nineteen.**
  The per-part trait set is unchanged at nineteen: no trait was added, renamed,
  or reshaped — only the bundle's seats and the equalities between them moved.

Migrating a vocabulary that already models one joint type per skeleton is a set
of renames. A vocabulary with a single joint type names it in all three 2-D
seats, names one pose type in both 2-D pose seats, and keeps compiling: the
contract still *permits* the identity, it merely stops *imposing* it. The
reference implementation in `src/tests/reference.rs` is exactly that case.

**Semver.** This breaks a published API — 0.2.0 is on crates.io — so it rides a
**0.3.0**. The version is not bumped here; the release stamp is its own commit.

### Changed

- `conformance::assert_poses_accept` exercises the four joint seats one by one,
  and `assert_coordinate_refusals` now checks all three 2-D seats — a validating
  vocabulary that guards only one joint type hears about the other two. Both
  panic with the seat that refused, not just the trait.
- `tests/common`, the vocabulary written from outside the crate, names four
  genuinely distinct joint types, so the decoupling is compiled rather than
  merely asserted.

### Unchanged

Engine behaviour is preserved verbatim: the same Vision passes, the same
filtering, the same joint sort, the same eighteen `Analysis` slots. Only which
type each joint is built into changed.

## 0.2.0 — 2026-08-20

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
