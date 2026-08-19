use core::fmt;

use crate::contract::Detections;

/// Everything one frame produced, in eighteen flat slots.
///
/// The engine builds this and stops. It does not know what a keyframe
/// is, does not mint identifiers, and does not nest the human or
/// animal slots into an aggregate — assembling those into whatever
/// record the caller stores is the caller's job, and the caller is the
/// only party that knows the frame's identity.
///
/// Slots start empty ([`Analysis::new`]) and are filled by
/// `set_*`. Nothing here is `Option` except [`horizon`](Analysis::horizon)
/// and [`aesthetics`](Analysis::aesthetics), which are single-valued:
/// the engine emits its "nothing detected" sentinel for those rather
/// than `None`, so `None` means the implementation refused even the
/// sentinel.
///
/// To move values out without cloning, take them through the `_mut`
/// accessors:
///
/// ```ignore
/// let faces = core::mem::take(analysis.faces_mut());
/// ```
pub struct Analysis<D: Detections> {
  classifications: Vec<D::Detection>,
  human_subjects: Vec<D::SubjectDetection>,
  faces: Vec<D::FaceDetection>,
  face_landmarks: Vec<D::FaceLandmarksDetection>,
  body_poses: Vec<D::BodyPoseDetection>,
  hand_poses: Vec<D::HandPoseDetection>,
  body_poses_3d: Vec<D::BodyPose3DDetection>,
  person_instance_masks: Vec<D::PersonInstanceMaskDetection>,
  person_segmentation_masks: Vec<D::PersonSegmentationMask>,
  animal_subjects: Vec<D::SubjectDetection>,
  animal_body_poses: Vec<D::BodyPoseDetection>,
  text_detections: Vec<D::TextDetection>,
  barcodes: Vec<D::BarcodeDetection>,
  attention_saliency: Vec<D::SaliencyRegion>,
  objectness_saliency: Vec<D::SaliencyRegion>,
  document_segments: Vec<D::DocumentSegment>,
  horizon: Option<D::HorizonInfo>,
  aesthetics: Option<D::Aesthetics>,
}

/// Generates the four-method surface every collection slot carries:
/// slice accessor, exclusive accessor, in-place setter, chaining
/// setter.
macro_rules! collection_slots {
  ($(
    $(#[$doc:meta])*
    $name:ident, $name_mut:ident, $set:ident, $with:ident: $ty:ty;
  )*) => {
    $(
      $(#[$doc])*
      #[inline]
      pub fn $name(&self) -> &[$ty] {
        &self.$name
      }

      #[doc = concat!(
        "Exclusive access to the `", stringify!($name), "` slot. \
         `core::mem::take` it to move the values out."
      )]
      #[inline]
      pub fn $name_mut(&mut self) -> &mut Vec<$ty> {
        &mut self.$name
      }

      #[doc = concat!("Replaces the `", stringify!($name), "` slot in place.")]
      #[inline]
      pub fn $set(&mut self, $name: Vec<$ty>) -> &mut Self {
        self.$name = $name;
        self
      }

      #[doc = concat!(
        "Returns this analysis with the `", stringify!($name), "` slot replaced."
      )]
      #[inline]
      pub fn $with(mut self, $name: Vec<$ty>) -> Self {
        self.$set($name);
        self
      }
    )*
  };
}

impl<D: Detections> Analysis<D> {
  /// An empty analysis: every collection slot empty, both
  /// single-valued slots absent.
  #[inline]
  pub const fn new() -> Self {
    Self {
      classifications: Vec::new(),
      human_subjects: Vec::new(),
      faces: Vec::new(),
      face_landmarks: Vec::new(),
      body_poses: Vec::new(),
      hand_poses: Vec::new(),
      body_poses_3d: Vec::new(),
      person_instance_masks: Vec::new(),
      person_segmentation_masks: Vec::new(),
      animal_subjects: Vec::new(),
      animal_body_poses: Vec::new(),
      text_detections: Vec::new(),
      barcodes: Vec::new(),
      attention_saliency: Vec::new(),
      objectness_saliency: Vec::new(),
      document_segments: Vec::new(),
      horizon: None,
      aesthetics: None,
    }
  }

  collection_slots! {
    /// Whole-frame image classifications.
    classifications, classifications_mut, set_classifications, with_classifications:
      D::Detection;

    /// People detected as subjects. Every label is the literal
    /// `"person"` — Apple's human-rectangles pass reports no species
    /// string.
    human_subjects, human_subjects_mut, set_human_subjects, with_human_subjects:
      D::SubjectDetection;

    /// Detected faces, one record per face: the face-rectangles pass
    /// is the spine, annotated with capture quality joined from the
    /// separate capture-quality pass.
    faces, faces_mut, set_faces, with_faces: D::FaceDetection;

    /// Faces carrying their landmark regions. A separate Vision pass
    /// from [`faces`](Analysis::faces) — the two are not joined, so a
    /// face may appear in one and not the other.
    face_landmarks, face_landmarks_mut, set_face_landmarks, with_face_landmarks:
      D::FaceLandmarksDetection;

    /// Human 2-D body poses.
    body_poses, body_poses_mut, set_body_poses, with_body_poses: D::BodyPoseDetection;

    /// Hand poses.
    hand_poses, hand_poses_mut, set_hand_poses, with_hand_poses: D::HandPoseDetection;

    /// Human 3-D body poses, in model-space metres.
    body_poses_3d, body_poses_3d_mut, set_body_poses_3d, with_body_poses_3d:
      D::BodyPose3DDetection;

    /// Per-instance person masks.
    person_instance_masks, person_instance_masks_mut, set_person_instance_masks,
      with_person_instance_masks: D::PersonInstanceMaskDetection;

    /// Whole-frame person masks. Shares a per-frame resource budget
    /// with [`person_instance_masks`](Analysis::person_instance_masks),
    /// so a mask-heavy frame can exhaust the budget in the instance
    /// pass and leave this slot short.
    person_segmentation_masks, person_segmentation_masks_mut,
      set_person_segmentation_masks, with_person_segmentation_masks:
      D::PersonSegmentationMask;

    /// Animals detected as subjects, labelled with Apple's species
    /// identifier verbatim.
    animal_subjects, animal_subjects_mut, set_animal_subjects, with_animal_subjects:
      D::SubjectDetection;

    /// Animal 2-D body poses. Same type as
    /// [`body_poses`](Analysis::body_poses) — the slot is what
    /// distinguishes animal from human.
    animal_body_poses, animal_body_poses_mut, set_animal_body_poses,
      with_animal_body_poses: D::BodyPoseDetection;

    /// Recognised text runs. One Vision observation can contribute
    /// several candidates, all sharing that observation's box.
    text_detections, text_detections_mut, set_text_detections, with_text_detections:
      D::TextDetection;

    /// Decoded barcodes.
    barcodes, barcodes_mut, set_barcodes, with_barcodes: D::BarcodeDetection;

    /// Attention-based salient regions.
    attention_saliency, attention_saliency_mut, set_attention_saliency,
      with_attention_saliency: D::SaliencyRegion;

    /// Objectness-based salient regions. Same type as
    /// [`attention_saliency`](Analysis::attention_saliency) — the slot
    /// is what distinguishes the two passes.
    objectness_saliency, objectness_saliency_mut, set_objectness_saliency,
      with_objectness_saliency: D::SaliencyRegion;

    /// Detected document quadrilaterals.
    document_segments, document_segments_mut, set_document_segments,
      with_document_segments: D::DocumentSegment;
  }

  /// The frame's horizon line.
  ///
  /// The engine writes its zero-angle, zero-confidence sentinel when
  /// no horizon was detected, so `None` means the output vocabulary
  /// refused even that sentinel.
  #[inline]
  pub const fn horizon(&self) -> Option<&D::HorizonInfo> {
    self.horizon.as_ref()
  }

  /// Exclusive access to the horizon slot. `Option::take` it to move
  /// the value out.
  #[inline]
  pub const fn horizon_mut(&mut self) -> &mut Option<D::HorizonInfo> {
    &mut self.horizon
  }

  /// Replaces the horizon slot in place.
  #[inline]
  pub fn set_horizon(&mut self, horizon: Option<D::HorizonInfo>) -> &mut Self {
    self.horizon = horizon;
    self
  }

  /// Returns this analysis with the horizon slot replaced.
  #[inline]
  pub fn with_horizon(mut self, horizon: Option<D::HorizonInfo>) -> Self {
    self.set_horizon(horizon);
    self
  }

  /// The frame's aesthetics score.
  ///
  /// As with [`horizon`](Analysis::horizon), the engine writes a
  /// sentinel rather than leaving this absent.
  #[inline]
  pub const fn aesthetics(&self) -> Option<&D::Aesthetics> {
    self.aesthetics.as_ref()
  }

  /// Exclusive access to the aesthetics slot. `Option::take` it to
  /// move the value out.
  #[inline]
  pub const fn aesthetics_mut(&mut self) -> &mut Option<D::Aesthetics> {
    &mut self.aesthetics
  }

  /// Replaces the aesthetics slot in place.
  #[inline]
  pub fn set_aesthetics(&mut self, aesthetics: Option<D::Aesthetics>) -> &mut Self {
    self.aesthetics = aesthetics;
    self
  }

  /// Returns this analysis with the aesthetics slot replaced.
  #[inline]
  pub fn with_aesthetics(mut self, aesthetics: Option<D::Aesthetics>) -> Self {
    self.set_aesthetics(aesthetics);
    self
  }
}

impl<D: Detections> Default for Analysis<D> {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl<D: Detections> fmt::Debug for Analysis<D>
where
  D::Detection: fmt::Debug,
  D::SubjectDetection: fmt::Debug,
  D::FaceDetection: fmt::Debug,
  D::FaceLandmarksDetection: fmt::Debug,
  D::BodyPoseDetection: fmt::Debug,
  D::HandPoseDetection: fmt::Debug,
  D::BodyPose3DDetection: fmt::Debug,
  D::PersonInstanceMaskDetection: fmt::Debug,
  D::PersonSegmentationMask: fmt::Debug,
  D::TextDetection: fmt::Debug,
  D::BarcodeDetection: fmt::Debug,
  D::SaliencyRegion: fmt::Debug,
  D::DocumentSegment: fmt::Debug,
  D::HorizonInfo: fmt::Debug,
  D::Aesthetics: fmt::Debug,
{
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Analysis")
      .field("classifications", &self.classifications)
      .field("human_subjects", &self.human_subjects)
      .field("faces", &self.faces)
      .field("face_landmarks", &self.face_landmarks)
      .field("body_poses", &self.body_poses)
      .field("hand_poses", &self.hand_poses)
      .field("body_poses_3d", &self.body_poses_3d)
      .field("person_instance_masks", &self.person_instance_masks)
      .field("person_segmentation_masks", &self.person_segmentation_masks)
      .field("animal_subjects", &self.animal_subjects)
      .field("animal_body_poses", &self.animal_body_poses)
      .field("text_detections", &self.text_detections)
      .field("barcodes", &self.barcodes)
      .field("attention_saliency", &self.attention_saliency)
      .field("objectness_saliency", &self.objectness_saliency)
      .field("document_segments", &self.document_segments)
      .field("horizon", &self.horizon)
      .field("aesthetics", &self.aesthetics)
      .finish()
  }
}
