#![doc = include_str!("../README.md")]
// #![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

//! Long-running Apple Vision.framework service thread.
//!
//! Each worker thread owns an `AppleVisionAnalyzer` and processes keyframes
//! independently. Vision.framework is stateless per-request, so multiple
//! workers can run in parallel.
//!
//! Input: `Request` via crossbeam bounded channel
//! Output: `Reply` via callback back to the processor-local coordinator

use std::panic::{AssertUnwindSafe, catch_unwind};

use bytes::Bytes;
use mediaschema::domain::ErrorCode;
use mediaschema::{
  Aesthetics, AnimalAnalysis, BarcodeDetection, BodyPose3DDetection, BodyPose3DHeightEstimation,
  BodyPose3DJoint, BodyPoseDetection, BodyPoseJoint, BoundingBox, ClassificationDetection,
  Dimensions, DocumentSegment, ErrorInfo, FaceDetection, FaceLandmarkPoint, FaceLandmarkRegion,
  FaceLandmarksDetection, FeaturePrint, HandChirality, HandPoseDetection, HorizonInfo,
  HumanAnalysis, Id, Keyframe, PersonInstanceMaskDetection, PersonSegmentationMask, SaliencyRegion,
  SubjectDetection, TextDetection,
};

use wire_ext::*;

// use tracing::{info, warn};

use objc2::{
  encode::{Encode, Encoding},
  rc::Retained,
};
use objc2_core_video::{
  CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight,
  CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress,
  CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVPixelFormatType_OneComponent8,
  kCVPixelFormatType_OneComponent32Float, kCVReturnSuccess,
};
use objc2_foundation::{NSArray, NSData, NSIndexSet, NSNotFound};
use objc2_vision::*;
use smol_str::{SmolStr, StrExt, ToSmolStr};

pub use options::*;

mod options;
mod wire_ext;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
struct SimdFloat4([f32; 4]);

unsafe impl Encode for SimdFloat4 {
  const ENCODING: Encoding = Encoding::Unknown;
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
struct SimdFloat4x4 {
  columns: [SimdFloat4; 4],
}

unsafe impl Encode for SimdFloat4x4 {
  // Clang reports @encode(simd_float4x4) as "{?=[4]}" because the vector element
  // encoding is intentionally opaque.
  const ENCODING: Encoding = Encoding::Struct("?", &[Encoding::Array(4, &Encoding::Unknown)]);
}

// #[derive(Debug, Clone, Copy)]
// pub struct Service(());

// impl ThreadService for Service {
//   type Input = Request;
//   type Options = ServiceOptions;
//   type SpawnError = SpawnError;
//   type Handle = ThreadHandles<Self::Input>;

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   fn name() -> &'static str {
//     "apple-vision"
//   }

//   fn health_spec(options: &Self::Options) -> ThreadServiceHealthSpec {
//     ThreadServiceHealthSpec::new(options.num_workers.max(1), ServiceHealthConfig::default())
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   fn spawn(
//     options: Self::Options,
//     ctx: ThreadServiceContext,
//   ) -> Result<Self::Handle, Self::SpawnError>
//   where
//     Self: Sized,
//     Self::Handle: findit_service::MessageHandle<Self::Input>,
//   {
//     let (tx, rx) = unbounded::<Self::Input>();
//     let (shutdown, health_reporter, health_handle, health_config) = ctx.into_parts();
//     let mut handles = Vec::with_capacity(options.num_workers);

//     for idx in 0..options.num_workers {
//       let rx = rx.clone();
//       let shutdown = shutdown.clone();
//       let opts = options.clone();
//       let health = health_reporter.clone();
//       let handle = std::thread::Builder::new()
//         .name(format!("{}-{idx}", Self::name()))
//         .spawn(move || {
//           run_apple_vision_worker(
//             Self::name(),
//             idx,
//             rx,
//             shutdown,
//             opts,
//             health,
//             health_config.heartbeat_interval(),
//           )
//         })
//         .map_err(|error| SpawnError::io("failed to spawn worker thread", error))?;
//       handles.push(handle);
//     }

//     Ok(ThreadHandles::with_named_service_health(
//       Self::name(),
//       tx,
//       handles,
//       Some(health_handle),
//     ))
//   }
// }

// impl ProviderIdentifier for Service {
//   const KEY: ProviderKey = ProviderKey::internal_after(
//     Lifecycle::Video(VideoLifecycle::KeyframeExtract),
//     Lifecycle::Video(VideoLifecycle::VisionAnalysis),
//     "apple-vision",
//   );
//   const IMPLEMENTATION_HASH: u64 = 0;
// }

// impl ProviderThreadService for Service {
//   const KIND: ProviderKind = ProviderKind::Standard;

//   type LifecycleInput = Request;
//   type LifecycleOutput = Reply;
// }

// /// Messages sent from processor tasks to the Apple Vision service.
// pub struct Request {
//   video_id: Id,
//   scene_id: Id,
//   keyframes: Arc<[Identified<Bytes>]>,
//   reply: Callback,
// }

// impl Request {
//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn new(
//     video_id: Id,
//     scene_id: Id,
//     keyframes: Arc<[Identified<Bytes>]>,
//     reply: Callback,
//   ) -> Self {
//     Self {
//       video_id,
//       scene_id,
//       keyframes,
//       reply,
//     }
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub const fn video_id(&self) -> Id {
//     self.video_id
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_video_id(&mut self, video_id: Id) -> &mut Self {
//     self.video_id = video_id;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_video_id(mut self, video_id: Id) -> Self {
//     self.set_video_id(video_id);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub const fn scene_id(&self) -> Id {
//     self.scene_id
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_scene_id(&mut self, scene_id: Id) -> &mut Self {
//     self.scene_id = scene_id;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_scene_id(mut self, scene_id: Id) -> Self {
//     self.set_scene_id(scene_id);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn keyframes(&self) -> &[Identified<Bytes>] {
//     &self.keyframes
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_keyframes(&mut self, keyframes: Arc<[Identified<Bytes>]>) -> &mut Self {
//     self.keyframes = keyframes;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_keyframes(mut self, keyframes: Arc<[Identified<Bytes>]>) -> Self {
//     self.set_keyframes(keyframes);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn reply(&self) -> &Callback {
//     &self.reply
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_reply(&mut self, reply: Callback) -> &mut Self {
//     self.reply = reply;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_reply(mut self, reply: Callback) -> Self {
//     self.set_reply(reply);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn into_parts(self) -> (Id, Id, Arc<[Identified<Bytes>]>, Callback) {
//     (self.video_id, self.scene_id, self.keyframes, self.reply)
//   }
// }

// pub struct Reply {
//   scene_id: Id,
//   results: Vec<Keyframe>,
//   errors: Vec<ErrorInfo>,
// }

// impl Reply {
//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn new(scene_id: Id, results: Vec<Keyframe>, errors: Vec<ErrorInfo>) -> Self {
//     Self {
//       scene_id,
//       results,
//       errors,
//     }
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub const fn scene_id(&self) -> Id {
//     self.scene_id
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_scene_id(&mut self, scene_id: Id) -> &mut Self {
//     self.scene_id = scene_id;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_scene_id(mut self, scene_id: Id) -> Self {
//     self.set_scene_id(scene_id);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn results(&self) -> &[Keyframe] {
//     &self.results
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_results(&mut self, results: Vec<Keyframe>) -> &mut Self {
//     self.results = results;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_results(mut self, results: Vec<Keyframe>) -> Self {
//     self.set_results(results);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn errors(&self) -> &[ErrorInfo] {
//     &self.errors
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn set_errors(&mut self, errors: Vec<ErrorInfo>) -> &mut Self {
//     self.errors = errors;
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn with_errors(mut self, errors: Vec<ErrorInfo>) -> Self {
//     self.set_errors(errors);
//     self
//   }

//   #[cfg_attr(not(tarpaulin), inline(always))]
//   pub fn into_parts(self) -> (Id, Vec<Keyframe>, Vec<ErrorInfo>) {
//     (self.scene_id, self.results, self.errors)
//   }
// }

// fn handle_message(worker_id: usize, analyzer: &VisionAnalyzer, request: Request) {
//   let (video_id, scene_id, keyframes, reply) = request.into_parts();
//   let svc = Service::name();

//   #[cfg(feature = "tracing")]
//   tracing::info!(service = svc, worker = worker_id, video_id = %video_id, scene_id = %scene_id, "analyzing scene");

//   let mut results = Vec::with_capacity(keyframes.len());
//   let mut errors = Vec::new();

//   for keyframe in keyframes.iter() {
//     match analyzer.analyze_keyframe(scene_id, keyframe.id(), keyframe.data()) {
//       Ok(r) => results.push(r),
//       Err(e) => {
//         #[cfg(feature = "tracing")]
//         tracing::warn!(
//           service = svc,
//           worker = worker_id,
//           video_id = %video_id,
//           keyframe_id = %keyframe.id(),
//           err = %e,
//           "Apple Vision analysis failed"
//         );
//         errors.push(apple_vision_keyframe_error(keyframe.id(), e));
//       }
//     }
//   }

//   reply(Reply::new(scene_id, results, errors));
// }

/// Apple vision analyzer
#[derive(Debug, Clone)]
pub struct VisionAnalyzer {
  opts: ServiceOptions,
  requests: VisionRequests,
}

#[derive(Debug, Clone)]
struct VisionRequests {
  classify: Retained<VNClassifyImageRequest>,
  face_rectangles: Retained<VNDetectFaceRectanglesRequest>,
  face_landmarks: Retained<VNDetectFaceLandmarksRequest>,
  face_quality: Retained<VNDetectFaceCaptureQualityRequest>,
  human_rectangles: Retained<VNDetectHumanRectanglesRequest>,
  body_pose: Retained<VNDetectHumanBodyPoseRequest>,
  body_pose_3d: Retained<VNDetectHumanBodyPose3DRequest>,
  hand_pose: Retained<VNDetectHumanHandPoseRequest>,
  animals: Retained<VNRecognizeAnimalsRequest>,
  animal_body_pose: Retained<VNDetectAnimalBodyPoseRequest>,
  person_instance_mask: Retained<VNGeneratePersonInstanceMaskRequest>,
  person_segmentation: Retained<VNGeneratePersonSegmentationRequest>,
  text: Retained<VNRecognizeTextRequest>,
  barcodes: Retained<VNDetectBarcodesRequest>,
  attention_saliency: Retained<VNGenerateAttentionBasedSaliencyImageRequest>,
  objectness_saliency: Retained<VNGenerateObjectnessBasedSaliencyImageRequest>,
  horizon: Retained<VNDetectHorizonRequest>,
  document_segments: Retained<VNDetectDocumentSegmentationRequest>,
  aesthetics: Retained<VNCalculateImageAestheticsScoresRequest>,
  feature_print: Retained<VNGenerateImageFeaturePrintRequest>,
}

fn apple_vision_error(code: ErrorCode, message: impl Into<String>) -> ErrorInfo {
  ErrorInfo::new(code, message.into())
}

// Used by the (currently commented) service-framework `handle_message` plumbing
// — kept here so we don't have to rewrite the error path when the service
// block is re-enabled.
#[allow(dead_code)]
fn apple_vision_keyframe_error(keyframe_id: Id, error: ErrorInfo) -> ErrorInfo {
  apple_vision_error(
    error.code(),
    format!("keyframe {:?}: {}", keyframe_id, error.message()),
  )
}

impl VisionRequests {
  fn new(opts: ServiceOptions) -> Self {
    unsafe {
      let classify = VNClassifyImageRequest::new();
      classify.setRevision(VNClassifyImageRequestRevision2);

      let face_rectangles = VNDetectFaceRectanglesRequest::new();
      face_rectangles.setRevision(VNDetectFaceRectanglesRequestRevision3);

      let face_landmarks = VNDetectFaceLandmarksRequest::new();
      face_landmarks.setRevision(VNDetectFaceLandmarksRequestRevision3);

      let face_quality = VNDetectFaceCaptureQualityRequest::new();
      face_quality.setRevision(VNDetectFaceCaptureQualityRequestRevision3);

      let human_rectangles = VNDetectHumanRectanglesRequest::new();
      human_rectangles.setUpperBodyOnly(false);
      human_rectangles.setRevision(VNDetectHumanRectanglesRequestRevision2);

      let body_pose = VNDetectHumanBodyPoseRequest::new();
      body_pose.setRevision(VNDetectHumanBodyPoseRequestRevision1);

      let body_pose_3d = VNDetectHumanBodyPose3DRequest::new();
      body_pose_3d.setRevision(VNDetectHumanBodyPose3DRequestRevision1);

      let hand_pose = VNDetectHumanHandPoseRequest::new();
      hand_pose.setMaximumHandCount(opts.hand_pose().maximum_hand_count());
      hand_pose.setRevision(VNDetectHumanHandPoseRequestRevision1);

      let animals = VNRecognizeAnimalsRequest::new();
      animals.setRevision(VNRecognizeAnimalsRequestRevision2);

      let animal_body_pose = VNDetectAnimalBodyPoseRequest::new();
      animal_body_pose.setRevision(VNDetectAnimalBodyPoseRequestRevision1);

      let person_instance_mask = VNGeneratePersonInstanceMaskRequest::new();
      person_instance_mask.setRevision(VNGeneratePersonInstanceMaskRequestRevision1);

      let person_segmentation = VNGeneratePersonSegmentationRequest::new();
      person_segmentation.setRevision(VNGeneratePersonSegmentationRequestRevision1);

      let text = VNRecognizeTextRequest::new();
      text.setRevision(VNRecognizeTextRequestRevision3);

      let barcodes = VNDetectBarcodesRequest::new();
      barcodes.setRevision(VNDetectBarcodesRequestRevision4);

      let attention_saliency = VNGenerateAttentionBasedSaliencyImageRequest::new();
      attention_saliency.setRevision(VNGenerateAttentionBasedSaliencyImageRequestRevision2);

      let objectness_saliency = VNGenerateObjectnessBasedSaliencyImageRequest::new();
      objectness_saliency.setRevision(VNGenerateObjectnessBasedSaliencyImageRequestRevision2);

      let horizon = VNDetectHorizonRequest::new();
      horizon.setRevision(VNDetectHorizonRequestRevision1);

      let document_segments = VNDetectDocumentSegmentationRequest::new();
      document_segments.setRevision(VNDetectDocumentSegmentationRequestRevision1);

      let aesthetics = VNCalculateImageAestheticsScoresRequest::new();
      aesthetics.setRevision(VNCalculateImageAestheticsScoresRequestRevision1);

      let feature_print = VNGenerateImageFeaturePrintRequest::new();
      feature_print.setRevision(VNGenerateImageFeaturePrintRequestRevision2);

      Self {
        classify,
        face_rectangles,
        face_landmarks,
        face_quality,
        human_rectangles: { human_rectangles },
        body_pose,
        body_pose_3d,
        hand_pose: { hand_pose },
        animals,
        animal_body_pose,
        person_instance_mask,
        person_segmentation,
        text,
        barcodes,
        attention_saliency,
        objectness_saliency,
        horizon,
        document_segments,
        aesthetics,
        feature_print,
      }
    }
  }

  fn perform(&self, handler: &VNSequenceRequestHandler, data: &NSData) -> Result<(), ErrorInfo> {
    unsafe {
      let requests = NSArray::from_retained_slice(&[
        Retained::cast_unchecked::<VNRequest>(self.classify.clone()),
        Retained::cast_unchecked::<VNRequest>(self.face_rectangles.clone()),
        Retained::cast_unchecked::<VNRequest>(self.face_landmarks.clone()),
        Retained::cast_unchecked::<VNRequest>(self.face_quality.clone()),
        Retained::cast_unchecked::<VNRequest>(self.human_rectangles.clone()),
        Retained::cast_unchecked::<VNRequest>(self.body_pose.clone()),
        Retained::cast_unchecked::<VNRequest>(self.body_pose_3d.clone()),
        Retained::cast_unchecked::<VNRequest>(self.hand_pose.clone()),
        Retained::cast_unchecked::<VNRequest>(self.animals.clone()),
        Retained::cast_unchecked::<VNRequest>(self.animal_body_pose.clone()),
        Retained::cast_unchecked::<VNRequest>(self.person_instance_mask.clone()),
        Retained::cast_unchecked::<VNRequest>(self.person_segmentation.clone()),
        Retained::cast_unchecked::<VNRequest>(self.text.clone()),
        Retained::cast_unchecked::<VNRequest>(self.barcodes.clone()),
        Retained::cast_unchecked::<VNRequest>(self.attention_saliency.clone()),
        Retained::cast_unchecked::<VNRequest>(self.objectness_saliency.clone()),
        Retained::cast_unchecked::<VNRequest>(self.horizon.clone()),
        Retained::cast_unchecked::<VNRequest>(self.document_segments.clone()),
        Retained::cast_unchecked::<VNRequest>(self.aesthetics.clone()),
        Retained::cast_unchecked::<VNRequest>(self.feature_print.clone()),
      ]);

      handler
        .performRequests_onImageData_error(&requests, data)
        .map_err(|e| {
          apple_vision_error(
            ErrorCode::AppleVisionRequestFailed,
            e.localizedDescription().to_string(),
          )
        })
    }
  }
}

impl VisionAnalyzer {
  /// Creates a new Apple Vision analyzer with the specified options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(opts: ServiceOptions) -> Self {
    Self {
      requests: VisionRequests::new(opts.clone()),
      opts,
    }
  }

  #[cfg(feature = "tracing")]
  #[allow(dead_code)] // called from the (currently commented) service-framework block
  fn log_request_revisions(&self, svc: &'static str, worker_id: usize) {
    unsafe {
      tracing::info!(
        service = svc,
        worker = worker_id,
        classify_rev = self.requests.classify.revision(),
        face_rectangles_rev = self.requests.face_rectangles.revision(),
        face_landmarks_rev = self.requests.face_landmarks.revision(),
        face_quality_rev = self.requests.face_quality.revision(),
        human_rectangles_rev = self.requests.human_rectangles.revision(),
        body_pose_rev = self.requests.body_pose.revision(),
        body_pose_3d_rev = self.requests.body_pose_3d.revision(),
        hand_pose_rev = self.requests.hand_pose.revision(),
        animals_rev = self.requests.animals.revision(),
        animal_body_pose_rev = self.requests.animal_body_pose.revision(),
        person_instance_mask_rev = self.requests.person_instance_mask.revision(),
        person_segmentation_rev = self.requests.person_segmentation.revision(),
        text_rev = self.requests.text.revision(),
        barcodes_rev = self.requests.barcodes.revision(),
        attention_saliency_rev = self.requests.attention_saliency.revision(),
        objectness_saliency_rev = self.requests.objectness_saliency.revision(),
        horizon_rev = self.requests.horizon.revision(),
        document_segments_rev = self.requests.document_segments.revision(),
        aesthetics_rev = self.requests.aesthetics.revision(),
        feature_print_rev = self.requests.feature_print.revision(),
        "initialized pinned Apple Vision request revisions"
      );
    }
  }

  /// Run every Apple Vision request configured at construction time against
  /// the supplied JPEG bytes and gather the resulting detections into a
  /// fully-populated [`Keyframe`].
  ///
  /// `scene_id` and `keyframe_id` are attached verbatim to the returned
  /// `Keyframe`; the engine does not derive or generate identifiers itself.
  pub fn analyze_keyframe(
    &self,
    scene_id: Id,
    keyframe_id: Id,
    jpeg_data: &[u8],
  ) -> Result<Keyframe, ErrorInfo> {
    objc2::rc::autoreleasepool(|_| {
      let ns_data = NSData::with_bytes(jpeg_data);
      let handler = unsafe { VNSequenceRequestHandler::new() };
      self.requests.perform(&handler, &ns_data)?;

      Ok(
        Keyframe::default()
          .with_id(keyframe_id)
          .with_scene_id(scene_id)
          .with_classifications(self.extract_classifications())
          .with_humans(
            HumanAnalysis::new()
              .with_subjects(self.extract_human_subjects())
              .with_faces(self.extract_faces())
              .with_face_rectangles(self.extract_face_rectangles())
              .with_face_landmarks(self.extract_face_landmarks())
              .with_body_poses(self.extract_body_poses())
              .with_hand_poses(self.extract_hand_poses())
              .with_body_poses_3d(self.extract_body_poses_3d())
              .with_instance_masks(self.extract_person_instance_masks())
              .with_segmentation_masks(self.extract_person_segmentation_masks()),
          )
          .with_animals(
            AnimalAnalysis::new()
              .with_subjects(self.extract_animal_subjects())
              .with_body_poses(self.extract_animal_body_poses()),
          )
          .with_text_detections(self.extract_text_detections())
          .with_barcodes(self.extract_barcodes())
          .with_attention_saliency(self.extract_attention_saliency())
          .with_objectness_saliency(self.extract_objectness_saliency())
          .with_horizon(self.extract_horizon())
          .with_document_segments(self.extract_document_segments())
          .with_feature_print(self.extract_feature_print())
          .with_aesthetics(self.extract_aesthetics()),
      )
    })
  }

  fn extract_classifications(&self) -> Vec<ClassificationDetection> {
    let opts = self.opts.classifications();
    let Some(results) = (unsafe { self.requests.classify.results() }) else {
      return Vec::new();
    };

    let mut tags = Vec::with_capacity(results.len().min(opts.max_results()));
    for obs in results.iter() {
      let confidence = unsafe { obs.confidence() };
      if confidence < opts.min_confidence() {
        continue;
      }

      let label = normalize_classification_label(unsafe { obs.identifier() }.to_smolstr());
      if !label.is_empty() {
        tags.push(ClassificationDetection::new(label, confidence));
        if tags.len() >= opts.max_results() {
          break;
        }
      }
    }

    tags
  }

  fn extract_faces(&self) -> Vec<FaceDetection> {
    let Some(results) = (unsafe { self.requests.face_quality.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.face_capture();

    let mut faces = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let bbox = unsafe { obs.boundingBox() }.standardize();
      let confidence = unsafe { obs.confidence() };
      let capture_quality = unsafe { obs.faceCaptureQuality() }
        .map(|q| q.floatValue())
        .unwrap_or(0.0);
      if confidence < opts.min_confidence() || capture_quality < opts.min_capture_quality() {
        continue;
      }

      faces.push(
        FaceDetection::default()
          .with_bbox(BoundingBox::new(
            bbox.origin.x as f32,
            bbox.origin.y as f32,
            bbox.size.width as f32,
            bbox.size.height as f32,
          ))
          .with_confidence(confidence)
          .with_capture_quality(capture_quality)
          .with_roll(unsafe { obs.roll() }.map(|v| v.floatValue()).unwrap_or(0.0))
          .with_yaw(unsafe { obs.yaw() }.map(|v| v.floatValue()).unwrap_or(0.0))
          .with_pitch(
            unsafe { obs.pitch() }
              .map(|v| v.floatValue())
              .unwrap_or(0.0),
          ),
      );
    }

    faces
  }

  fn extract_face_rectangles(&self) -> Vec<FaceDetection> {
    let Some(results) = (unsafe { self.requests.face_rectangles.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.face_rectangles();

    let mut faces = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let confidence = unsafe { obs.confidence() };
      if confidence < opts.min_confidence() {
        continue;
      }

      let bbox = unsafe { obs.boundingBox() }.standardize();
      faces.push(
        FaceDetection::default()
          .with_bbox(BoundingBox::new(
            bbox.origin.x as f32,
            bbox.origin.y as f32,
            bbox.size.width as f32,
            bbox.size.height as f32,
          ))
          .with_confidence(confidence)
          .with_roll(unsafe { obs.roll() }.map(|v| v.floatValue()).unwrap_or(0.0))
          .with_yaw(unsafe { obs.yaw() }.map(|v| v.floatValue()).unwrap_or(0.0))
          .with_pitch(
            unsafe { obs.pitch() }
              .map(|v| v.floatValue())
              .unwrap_or(0.0),
          ),
      );
    }

    faces
  }

  fn extract_face_landmarks(&self) -> Vec<FaceLandmarksDetection> {
    let Some(results) = (unsafe { self.requests.face_landmarks.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.face_landmarks();

    let mut detections = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let Some(landmarks) = (unsafe { obs.landmarks() }) else {
        continue;
      };
      let confidence = unsafe { landmarks.confidence() };
      if confidence < opts.min_confidence() {
        continue;
      }

      let regions = extract_face_landmark_regions(&landmarks);
      if regions.len() < opts.min_region_count() {
        continue;
      }

      let bbox = unsafe { obs.boundingBox() }.standardize();
      detections.push(FaceLandmarksDetection::new(
        BoundingBox::new(
          bbox.origin.x as f32,
          bbox.origin.y as f32,
          bbox.size.width as f32,
          bbox.size.height as f32,
        ),
        confidence,
        regions,
      ));
    }

    detections
  }

  fn extract_human_subjects(&self) -> Vec<SubjectDetection> {
    let Some(results) = (unsafe { self.requests.human_rectangles.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.human_subjects();

    let mut humans = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let confidence = unsafe { obs.confidence() };
      if confidence < opts.min_confidence() {
        continue;
      }

      let bbox = unsafe { obs.boundingBox() }.standardize();
      humans.push(SubjectDetection::new(
        SmolStr::from("person"),
        confidence,
        BoundingBox::new(
          bbox.origin.x as f32,
          bbox.origin.y as f32,
          bbox.size.width as f32,
          bbox.size.height as f32,
        ),
      ));
    }

    humans
  }

  fn extract_body_poses(&self) -> Vec<BodyPoseDetection> {
    let Some(results) = (unsafe { self.requests.body_pose.results() }) else {
      return Vec::new();
    };

    let mut body_poses = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let Ok(points_by_joint) = (unsafe {
        obs.recognizedPointsForJointsGroupName_error(VNHumanBodyPoseObservationJointsGroupNameAll)
      }) else {
        continue;
      };

      let (joint_names, points) = points_by_joint.to_vecs();
      let mut joints = Vec::with_capacity(points.len());
      let mut min_x = f32::INFINITY;
      let mut min_y = f32::INFINITY;
      let mut max_x = f32::NEG_INFINITY;
      let mut max_y = f32::NEG_INFINITY;

      for (joint_name, point) in joint_names.into_iter().zip(points) {
        let name = joint_name.to_smolstr();
        if name.is_empty() {
          continue;
        }

        let x = unsafe { point.x() } as f32;
        let y = unsafe { point.y() } as f32;
        let confidence = unsafe { point.confidence() };
        if confidence < self.opts.body_pose().min_joint_confidence() {
          continue;
        }

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);

        joints.push(BodyPoseJoint::new(name, x, y, confidence));
      }

      if joints.is_empty() {
        continue;
      }

      joints.sort_by(|lhs, rhs| lhs.name().cmp(rhs.name()));
      let bbox = if min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()
      {
        BoundingBox::new(
          min_x,
          min_y,
          (max_x - min_x).max(0.0),
          (max_y - min_y).max(0.0),
        )
      } else {
        BoundingBox::default()
      };

      body_poses.push(BodyPoseDetection::new(
        bbox,
        unsafe { obs.confidence() },
        joints,
      ));
    }

    body_poses
  }

  fn extract_body_poses_3d(&self) -> Vec<BodyPose3DDetection> {
    catch_unwind(AssertUnwindSafe(|| {
      let Some(results) = (unsafe { self.requests.body_pose_3d.results() }) else {
        return Vec::new();
      };
      let Some(group_name) = (unsafe { VNHumanBodyPose3DObservationJointsGroupNameAll }) else {
        return Vec::new();
      };

      let mut body_poses = Vec::with_capacity(results.len());
      for obs in results.iter() {
        let Ok(points_by_joint) =
          (unsafe { obs.recognizedPointsForJointsGroupName_error(group_name) })
        else {
          continue;
        };

        let (joint_names, points) = points_by_joint.to_vecs();
        let mut joints = Vec::with_capacity(points.len());

        for (joint_name, point) in joint_names.into_iter().zip(points) {
          let name = joint_name.to_smolstr();
          if name.is_empty() {
            continue;
          }

          let Some((x, y, z)) = extract_body_pose_3d_coordinates(&point) else {
            continue;
          };
          let confidence: f32 = unsafe { objc2::msg_send![&*point, confidence] };
          if confidence < self.opts.body_pose_3d().min_joint_confidence() {
            continue;
          }

          joints.push(BodyPose3DJoint::new(name, x, y, z, confidence));
        }

        if joints.is_empty() {
          continue;
        }

        joints.sort_by(|lhs, rhs| lhs.name().cmp(rhs.name()));
        body_poses.push(BodyPose3DDetection::new(
          unsafe { obs.confidence() },
          unsafe { obs.bodyHeight() },
          map_body_pose_3d_height_estimation(unsafe { obs.heightEstimation() }),
          joints,
        ));
      }

      body_poses
    }))
    .unwrap_or_else(|_| {
      #[cfg(feature = "tracing")]
      tracing::warn!("caught panic while extracting human body pose 3D; returning empty result");
      Vec::new()
    })
  }

  fn extract_hand_poses(&self) -> Vec<HandPoseDetection> {
    let Some(results) = (unsafe { self.requests.hand_pose.results() }) else {
      return Vec::new();
    };

    let mut hand_poses = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let Ok(points_by_joint) = (unsafe {
        obs.recognizedPointsForJointsGroupName_error(VNHumanHandPoseObservationJointsGroupNameAll)
      }) else {
        continue;
      };

      let (joint_names, points) = points_by_joint.to_vecs();
      let mut joints = Vec::with_capacity(points.len());
      let mut min_x = f32::INFINITY;
      let mut min_y = f32::INFINITY;
      let mut max_x = f32::NEG_INFINITY;
      let mut max_y = f32::NEG_INFINITY;

      for (joint_name, point) in joint_names.into_iter().zip(points) {
        let name = joint_name.to_smolstr();
        if name.is_empty() {
          continue;
        }

        let x = unsafe { point.x() } as f32;
        let y = unsafe { point.y() } as f32;
        let confidence = unsafe { point.confidence() };
        if confidence < self.opts.hand_pose().min_joint_confidence() {
          continue;
        }

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);

        joints.push(BodyPoseJoint::new(name, x, y, confidence));
      }

      if joints.is_empty() {
        continue;
      }

      joints.sort_by(|lhs, rhs| lhs.name().cmp(rhs.name()));
      let bbox = if min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()
      {
        BoundingBox::new(
          min_x,
          min_y,
          (max_x - min_x).max(0.0),
          (max_y - min_y).max(0.0),
        )
      } else {
        BoundingBox::default()
      };

      hand_poses.push(HandPoseDetection::new(
        bbox,
        unsafe { obs.confidence() },
        map_hand_chirality(unsafe { obs.chirality() }),
        joints,
      ));
    }

    hand_poses
  }

  fn extract_person_instance_masks(&self) -> Vec<PersonInstanceMaskDetection> {
    let Some(results) = (unsafe { self.requests.person_instance_mask.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.person_instance_masks();

    let mut masks = Vec::new();
    for observation in results.iter() {
      let confidence = unsafe { observation.confidence() };
      if confidence < opts.min_confidence() {
        continue;
      }

      let instances = unsafe { observation.allInstances() };
      let mut instance_index = instances.firstIndex();
      let mut emitted = 0usize;
      while instance_index != NSNotFound as usize {
        if emitted >= opts.max_instances_per_observation() {
          break;
        }

        let selected_instances = NSIndexSet::indexSetWithIndex(instance_index);
        let Ok(mask_buffer) =
          (unsafe { observation.generateMaskForInstances_error(&selected_instances) })
        else {
          instance_index = instances.indexGreaterThanIndex(instance_index);
          continue;
        };

        let Some((bbox, dimensions, data)) = copy_instance_mask_buffer(&mask_buffer) else {
          instance_index = instances.indexGreaterThanIndex(instance_index);
          continue;
        };

        masks.push(PersonInstanceMaskDetection::new(
          bbox,
          confidence,
          u32::try_from(instance_index).unwrap_or_default(),
          dimensions,
          data,
        ));
        emitted += 1;

        instance_index = instances.indexGreaterThanIndex(instance_index);
      }
    }

    masks
  }

  fn extract_person_segmentation_masks(&self) -> Vec<PersonSegmentationMask> {
    let Some(results) = (unsafe { self.requests.person_segmentation.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.person_segmentation_masks();

    let mut masks = Vec::with_capacity(results.len());
    for observation in results.iter() {
      let confidence = unsafe { observation.confidence() };
      if confidence < opts.min_confidence() {
        continue;
      }

      let pixel_buffer = unsafe { observation.pixelBuffer() };
      let Some((bbox, dimensions, data)) = copy_instance_mask_buffer(&pixel_buffer) else {
        continue;
      };

      masks.push(PersonSegmentationMask::new(
        bbox, confidence, dimensions, data,
      ));
    }

    masks
  }

  fn extract_animal_subjects(&self) -> Vec<SubjectDetection> {
    unsafe {
      let Some(results) = self.requests.animals.results() else {
        return Vec::new();
      };

      let mut animals = Vec::new();
      for obs in results.iter() {
        let labels = obs.labels();
        for label in labels.iter() {
          let confidence = label.confidence();
          if confidence >= self.opts.animals().min_confidence() {
            let id = label.identifier().to_smolstr();
            if !id.is_empty() {
              let bbox = obs.boundingBox().standardize();
              animals.push(SubjectDetection::new(
                id,
                confidence,
                BoundingBox::new(
                  bbox.origin.x as f32,
                  bbox.origin.y as f32,
                  bbox.size.width as f32,
                  bbox.size.height as f32,
                ),
              ));
            }
          }
        }
      }

      animals
    }
  }

  fn extract_animal_body_poses(&self) -> Vec<BodyPoseDetection> {
    let Some(results) = (unsafe { self.requests.animal_body_pose.results() }) else {
      return Vec::new();
    };
    let Some(group_name) = (unsafe { VNAnimalBodyPoseObservationJointsGroupNameAll }) else {
      return Vec::new();
    };

    let mut body_poses = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let Ok(points_by_joint) =
        (unsafe { obs.recognizedPointsForJointsGroupName_error(group_name) })
      else {
        continue;
      };

      let (joint_names, points) = points_by_joint.to_vecs();
      let mut joints = Vec::with_capacity(points.len());
      let mut min_x = f32::INFINITY;
      let mut min_y = f32::INFINITY;
      let mut max_x = f32::NEG_INFINITY;
      let mut max_y = f32::NEG_INFINITY;

      for (joint_name, point) in joint_names.into_iter().zip(points) {
        let name = joint_name.to_smolstr();
        if name.is_empty() {
          continue;
        }

        let x = unsafe { point.x() } as f32;
        let y = unsafe { point.y() } as f32;
        let confidence = unsafe { point.confidence() };
        if confidence < self.opts.animal_pose().min_joint_confidence() {
          continue;
        }

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);

        joints.push(BodyPoseJoint::new(name, x, y, confidence));
      }

      if joints.is_empty() {
        continue;
      }

      joints.sort_by(|lhs, rhs| lhs.name().cmp(rhs.name()));
      let bbox = if min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()
      {
        BoundingBox::new(
          min_x,
          min_y,
          (max_x - min_x).max(0.0),
          (max_y - min_y).max(0.0),
        )
      } else {
        BoundingBox::default()
      };

      body_poses.push(BodyPoseDetection::new(
        bbox,
        unsafe { obs.confidence() },
        joints,
      ));
    }

    body_poses
  }

  fn extract_text_detections(&self) -> Vec<TextDetection> {
    let Some(results) = self.requests.text.results() else {
      return Vec::new();
    };

    let mut text_detections = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let candidates = obs.topCandidates(self.opts.text().max_candidates_per_observation());
      for candidate in candidates.iter() {
        let text = candidate.string().to_smolstr();
        if text.len() >= self.opts.text().min_text_len() {
          let bbox = unsafe { obs.boundingBox() }.standardize();
          text_detections.push(TextDetection::new(
            text,
            candidate.confidence(),
            BoundingBox::new(
              bbox.origin.x as f32,
              bbox.origin.y as f32,
              bbox.size.width as f32,
              bbox.size.height as f32,
            ),
          ));
        }
      }
    }
    text_detections
  }

  fn extract_barcodes(&self) -> Vec<BarcodeDetection> {
    let Some(results) = (unsafe { self.requests.barcodes.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.barcodes();

    let mut barcodes = Vec::with_capacity(results.len());
    for obs in results.iter() {
      let confidence = unsafe { obs.confidence() };
      if confidence < opts.min_confidence() {
        continue;
      }

      if let Some(payload) = unsafe { obs.payloadStringValue() } {
        let s = payload.to_smolstr();
        if s.len() >= opts.min_payload_len() {
          let bbox = unsafe { obs.boundingBox() }.standardize();
          let symbology = unsafe { obs.symbology() }.to_smolstr();
          barcodes.push(BarcodeDetection::new(
            s,
            symbology,
            confidence,
            BoundingBox::new(
              bbox.origin.x as f32,
              bbox.origin.y as f32,
              bbox.size.width as f32,
              bbox.size.height as f32,
            ),
          ));
        }
      }
    }
    barcodes
  }

  fn extract_attention_saliency(&self) -> Vec<SaliencyRegion> {
    self.extract_saliency_regions(
      unsafe { self.requests.attention_saliency.results() },
      self.opts.attention_saliency(),
    )
  }

  fn extract_objectness_saliency(&self) -> Vec<SaliencyRegion> {
    self.extract_saliency_regions(
      unsafe { self.requests.objectness_saliency.results() },
      self.opts.objectness_saliency(),
    )
  }

  fn extract_saliency_regions(
    &self,
    observations: Option<Retained<NSArray<VNSaliencyImageObservation>>>,
    opts: AppleVisionSaliencyOptions,
  ) -> Vec<SaliencyRegion> {
    let Some(observations) = observations else {
      return Vec::new();
    };

    let mut regions = Vec::new();
    for observation in observations.iter() {
      let Some(objects) = (unsafe { observation.salientObjects() }) else {
        continue;
      };
      for object in objects.iter().take(opts.max_regions()) {
        let confidence = unsafe { object.confidence() };
        if confidence < opts.min_confidence() {
          continue;
        }

        let bbox = unsafe { object.boundingBox() }.standardize();
        regions.push(SaliencyRegion::new(
          BoundingBox::new(
            bbox.origin.x as f32,
            bbox.origin.y as f32,
            bbox.size.width as f32,
            bbox.size.height as f32,
          ),
          confidence,
        ));
      }
    }
    regions
  }

  fn extract_horizon(&self) -> HorizonInfo {
    let Some(results) = (unsafe { self.requests.horizon.results() }) else {
      return HorizonInfo::default();
    };
    let Some(observation) = results.iter().next() else {
      return HorizonInfo::default();
    };
    let confidence = unsafe { observation.confidence() };
    if confidence < self.opts.horizon().min_confidence() {
      return HorizonInfo::default();
    }

    HorizonInfo::new(unsafe { observation.angle() } as f32, confidence)
  }

  fn extract_document_segments(&self) -> Vec<DocumentSegment> {
    let Some(results) = (unsafe { self.requests.document_segments.results() }) else {
      return Vec::new();
    };
    let opts = self.opts.document_segments();

    let mut segments = Vec::with_capacity(results.len());
    for observation in results.iter() {
      if segments.len() >= opts.max_segments() {
        break;
      }

      let confidence = unsafe { observation.confidence() };
      if confidence < opts.min_confidence() {
        continue;
      }

      let top_left = unsafe { observation.topLeft() };
      let top_right = unsafe { observation.topRight() };
      let bottom_left = unsafe { observation.bottomLeft() };
      let bottom_right = unsafe { observation.bottomRight() };

      segments.push(
        DocumentSegment::default()
          .with_top_left((top_left.x as f32, top_left.y as f32))
          .with_top_right((top_right.x as f32, top_right.y as f32))
          .with_bottom_left((bottom_left.x as f32, bottom_left.y as f32))
          .with_bottom_right((bottom_right.x as f32, bottom_right.y as f32))
          .with_confidence(confidence),
      );
    }

    segments
  }

  fn extract_aesthetics(&self) -> Aesthetics {
    let Some(results) = (unsafe { self.requests.aesthetics.results() }) else {
      return Aesthetics::default();
    };
    let Some(obs) = results.iter().next() else {
      return Aesthetics::default();
    };
    let overall_score = unsafe { obs.overallScore() };
    if overall_score < self.opts.aesthetics().min_overall_score() {
      return Aesthetics::default();
    }

    Aesthetics::new(overall_score, unsafe { obs.isUtility() })
  }

  fn extract_feature_print(&self) -> FeaturePrint {
    let Some(results) = (unsafe { self.requests.feature_print.results() }) else {
      return FeaturePrint::default();
    };
    let Some(obs) = results.iter().next() else {
      return FeaturePrint::default();
    };
    let count = unsafe { obs.elementCount() };
    if count < self.opts.feature_print().min_element_count() {
      return FeaturePrint::default();
    }

    let ns_data = unsafe { obs.data() };
    let len = ns_data.len();
    let ptr: *const std::ffi::c_void = unsafe { objc2::msg_send![&*ns_data, bytes] };
    if ptr.is_null() || len == 0 {
      return FeaturePrint::default();
    }

    let data = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }.to_vec();
    let element_type = u32::try_from(unsafe { obs.elementType() }.0).unwrap_or_default();
    FeaturePrint::new(Bytes::from(data), element_type)
  }
}

fn normalize_classification_label(label: SmolStr) -> SmolStr {
  label.trim().to_ascii_lowercase_smolstr()
}

fn extract_body_pose_3d_coordinates(
  point: &VNHumanBodyRecognizedPoint3D,
) -> Option<(f32, f32, f32)> {
  let transform: SimdFloat4x4 = unsafe { objc2::msg_send![point, position] };
  let translation = transform.columns.get(3)?;
  let x = translation.0[0];
  let y = translation.0[1];
  let z = translation.0[2];
  if !(x.is_finite() && y.is_finite() && z.is_finite()) {
    return None;
  }
  Some((x, y, z))
}

fn map_hand_chirality(chirality: VNChirality) -> HandChirality {
  match chirality {
    VNChirality::Left => HAND_CHIRALITY_LEFT,
    VNChirality::Right => HAND_CHIRALITY_RIGHT,
    _ => HAND_CHIRALITY_UNKNOWN,
  }
}

fn extract_face_landmark_regions(landmarks: &VNFaceLandmarks2D) -> Vec<FaceLandmarkRegion> {
  let mut regions = Vec::new();
  push_face_landmark_region(&mut regions, "allPoints", unsafe { landmarks.allPoints() });
  push_face_landmark_region(&mut regions, "faceContour", unsafe {
    landmarks.faceContour()
  });
  push_face_landmark_region(&mut regions, "leftEye", unsafe { landmarks.leftEye() });
  push_face_landmark_region(&mut regions, "rightEye", unsafe { landmarks.rightEye() });
  push_face_landmark_region(&mut regions, "leftEyebrow", unsafe {
    landmarks.leftEyebrow()
  });
  push_face_landmark_region(&mut regions, "rightEyebrow", unsafe {
    landmarks.rightEyebrow()
  });
  push_face_landmark_region(&mut regions, "nose", unsafe { landmarks.nose() });
  push_face_landmark_region(&mut regions, "noseCrest", unsafe { landmarks.noseCrest() });
  push_face_landmark_region(&mut regions, "medianLine", unsafe {
    landmarks.medianLine()
  });
  push_face_landmark_region(&mut regions, "outerLips", unsafe { landmarks.outerLips() });
  push_face_landmark_region(&mut regions, "innerLips", unsafe { landmarks.innerLips() });
  push_face_landmark_region(&mut regions, "leftPupil", unsafe { landmarks.leftPupil() });
  push_face_landmark_region(&mut regions, "rightPupil", unsafe {
    landmarks.rightPupil()
  });
  regions
}

fn push_face_landmark_region(
  regions: &mut Vec<FaceLandmarkRegion>,
  name: &'static str,
  region: Option<Retained<VNFaceLandmarkRegion2D>>,
) {
  let Some(region) = region else {
    return;
  };

  let point_count = unsafe { region.pointCount() };
  if point_count == 0 {
    return;
  }

  let points_ptr = unsafe { region.normalizedPoints() };
  if points_ptr.is_null() {
    return;
  }

  let points = unsafe { std::slice::from_raw_parts(points_ptr, point_count) };
  let points = points
    .iter()
    .map(|point| FaceLandmarkPoint::new(point.x as f32, point.y as f32))
    .collect::<Vec<_>>();
  if points.is_empty() {
    return;
  }

  regions.push(FaceLandmarkRegion::new(name, points));
}

fn map_body_pose_3d_height_estimation(
  estimation: VNHumanBodyPose3DObservationHeightEstimation,
) -> BodyPose3DHeightEstimation {
  if estimation == VNHumanBodyPose3DObservationHeightEstimation::Measured {
    BODY_POSE_3D_HEIGHT_ESTIMATION_MEASURED
  } else if estimation == VNHumanBodyPose3DObservationHeightEstimation::Reference {
    BODY_POSE_3D_HEIGHT_ESTIMATION_REFERENCE
  } else {
    BODY_POSE_3D_HEIGHT_ESTIMATION_UNKNOWN
  }
}

fn copy_instance_mask_buffer(
  pixel_buffer: &objc2_core_video::CVPixelBuffer,
) -> Option<(BoundingBox, Dimensions, Bytes)> {
  let lock_flags = CVPixelBufferLockFlags::ReadOnly;
  if unsafe { CVPixelBufferLockBaseAddress(pixel_buffer, lock_flags) } != kCVReturnSuccess {
    return None;
  }

  let result = copy_instance_mask_buffer_locked(pixel_buffer);
  let _ = unsafe { CVPixelBufferUnlockBaseAddress(pixel_buffer, lock_flags) };
  result
}

#[allow(non_upper_case_globals)]
fn copy_instance_mask_buffer_locked(
  pixel_buffer: &objc2_core_video::CVPixelBuffer,
) -> Option<(BoundingBox, Dimensions, Bytes)> {
  let width = CVPixelBufferGetWidth(pixel_buffer);
  let height = CVPixelBufferGetHeight(pixel_buffer);
  if width == 0 || height == 0 {
    return None;
  }

  let pixel_format = CVPixelBufferGetPixelFormatType(pixel_buffer);
  let bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
  let base_address = CVPixelBufferGetBaseAddress(pixel_buffer) as *const u8;
  if base_address.is_null() || bytes_per_row == 0 {
    return None;
  }

  match pixel_format {
    kCVPixelFormatType_OneComponent32Float => {
      let mut packed = vec![0u8; width * height * core::mem::size_of::<f32>()];
      let mut min_x = usize::MAX;
      let mut min_y = usize::MAX;
      let mut max_x = 0usize;
      let mut max_y = 0usize;
      let mut has_foreground = false;

      for row in 0..height {
        let src_row = unsafe { base_address.add(row * bytes_per_row) };
        let src_f32 =
          unsafe { std::slice::from_raw_parts(src_row as *const f32, bytes_per_row / 4) };
        let dst_row = &mut packed[row * width * 4..(row + 1) * width * 4];
        for col in 0..width {
          let value = *src_f32.get(col)?;
          dst_row[col * 4..(col + 1) * 4].copy_from_slice(&value.to_le_bytes());
          if value > 0.0 {
            has_foreground = true;
            min_x = min_x.min(col);
            min_y = min_y.min(row);
            max_x = max_x.max(col);
            max_y = max_y.max(row);
          }
        }
      }

      let bbox = if has_foreground {
        normalized_bbox_from_pixel_bounds(min_x, min_y, max_x, max_y, width, height)
      } else {
        BoundingBox::default()
      };

      Some((
        bbox,
        Dimensions::new(
          u16::try_from(width).unwrap_or(u16::MAX),
          u16::try_from(height).unwrap_or(u16::MAX),
        ),
        Bytes::from(packed),
      ))
    }
    kCVPixelFormatType_OneComponent8 => {
      let mut packed = vec![0u8; width * height];
      let mut min_x = usize::MAX;
      let mut min_y = usize::MAX;
      let mut max_x = 0usize;
      let mut max_y = 0usize;
      let mut has_foreground = false;

      for row in 0..height {
        let src_row = unsafe {
          std::slice::from_raw_parts(base_address.add(row * bytes_per_row), bytes_per_row)
        };
        let dst_row = &mut packed[row * width..(row + 1) * width];
        dst_row.copy_from_slice(&src_row[..width]);
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

      let bbox = if has_foreground {
        normalized_bbox_from_pixel_bounds(min_x, min_y, max_x, max_y, width, height)
      } else {
        BoundingBox::default()
      };

      Some((
        bbox,
        Dimensions::new(
          u16::try_from(width).unwrap_or(u16::MAX),
          u16::try_from(height).unwrap_or(u16::MAX),
        ),
        Bytes::from(packed),
      ))
    }
    _ => None,
  }
}

fn normalized_bbox_from_pixel_bounds(
  min_x: usize,
  min_y: usize,
  max_x: usize,
  max_y: usize,
  width: usize,
  height: usize,
) -> BoundingBox {
  let x = min_x as f32 / width as f32;
  let y = 1.0 - ((max_y + 1) as f32 / height as f32);
  let w = (max_x + 1 - min_x) as f32 / width as f32;
  let h = (max_y + 1 - min_y) as f32 / height as f32;
  BoundingBox::new(x, y, w, h)
}

