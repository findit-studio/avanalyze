#![allow(missing_docs)]

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

macro_rules! default_options {
  ($($name:ident),+$(,)?) => {
    $(
      impl Default for $name {
        #[cfg_attr(not(tarpaulin), inline(always))]
        fn default() -> Self {
          Self::new()
        }
      }
    )*
  };
}

default_options!(
  AppleVisionClassificationOptions,
  AppleVisionAnimalOptions,
  AppleVisionTextOptions,
  AppleVisionBodyPoseOptions,
  AppleVisionHandPoseOptions,
  AppleVisionAnimalPoseOptions,
  AppleVisionBodyPose3DOptions,
  AppleVisionFaceCaptureOptions,
  AppleVisionFaceRectangleOptions,
  AppleVisionFaceKeypointsOptions,
  AppleVisionFaceOptions,
  AppleVisionFaceLandmarkOptions,
  AppleVisionHumanSubjectOptions,
  AppleVisionBarcodeOptions,
  AppleVisionSaliencyOptions,
  AppleVisionHorizonOptions,
  AppleVisionDocumentSegmentationOptions,
  AppleVisionAestheticsOptions,
  AppleVisionPersonInstanceMaskOptions,
  AppleVisionPersonSegmentationOptions,
  AppleVisionBodyPoserOptions,
  AppleVisionPersonMaskerOptions,
  AnalyzeOptions,
);

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_classification_min_confidence() -> f32 {
  0.3
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_classification_max_results() -> usize {
  12
}
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionClassificationOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_classification_min_confidence")
  )]
  min_confidence: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_classification_max_results")
  )]
  max_results: usize,
}

impl AppleVisionClassificationOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_classification_min_confidence();
  /// Default [`max_results`](Self::max_results).
  pub const DEFAULT_MAX_RESULTS: usize = default_classification_max_results();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
      max_results: Self::DEFAULT_MAX_RESULTS,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_results(mut self, max_results: usize) -> Self {
    self.set_max_results(max_results);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_results(&mut self, max_results: usize) -> &mut Self {
    self.max_results = max_results;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_results(&self) -> usize {
    self.max_results
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_animal_min_confidence() -> f32 {
  0.3
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionAnimalOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_animal_min_confidence"))]
  min_confidence: f32,
}

impl AppleVisionAnimalOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_animal_min_confidence();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_text_min_len() -> usize {
  1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_text_max_candidates_per_observation() -> usize {
  1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_text_detect_language() -> bool {
  true
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_text_recognition_level() -> TextRecognitionLevel {
  TextRecognitionLevel::Accurate
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_text_language_correction() -> bool {
  true
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_text_min_text_height() -> f32 {
  0.0
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_text_min_confidence() -> f32 {
  0.0
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_text_revision() -> usize {
  3
}

/// How hard Vision works to read text — Apple's
/// `VNRequestTextRecognitionLevel`.
///
/// The level also decides which languages the request reads at all.
/// [`Accurate`](Self::Accurate) reads every language its revision lists;
/// [`Fast`](Self::Fast) reads only Latin-script ones (at revision 3:
/// English, French, Italian, German, Spanish and Portuguese), so a
/// Chinese or Japanese roster at `Fast` is refused by name rather than
/// left to read nothing.
///
/// In a config it is spelled `"accurate"` or `"fast"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum TextRecognitionLevel {
  /// Slower, and reads every language the request's revision lists.
  /// Apple's default.
  #[default]
  Accurate,
  /// Faster, and reads Latin-script languages only.
  Fast,
}

/// Everything [`TextRecognizer`](crate::TextRecognizer) reads.
///
/// Two kinds of option live here, and what separates them is when they
/// act.
///
/// **The request's own, set once by
/// [`TextRecognizer::new`](crate::TextRecognizer::new).** These are
/// `VNRecognizeTextRequest` properties. They follow the recognizer, not
/// the call: the options handed to a later
/// [`recognize`](crate::TextRecognizer::recognize) cannot move them.
///
/// | option | default | Vision property |
/// |---|---|---|
/// | [`revision`](Self::revision) | `3` | `revision` — refused unless this host implements it |
/// | [`recognition_level`](Self::recognition_level) | [`Accurate`](TextRecognitionLevel::Accurate) | `recognitionLevel` |
/// | [`languages`](Self::languages) | empty: Vision's own roster | `recognitionLanguages` — refused unless the request lists every tag |
/// | [`detect_language`](Self::detect_language) | `true` | `automaticallyDetectsLanguage` (macOS 13 and later) |
/// | [`language_correction`](Self::language_correction) | `true` | `usesLanguageCorrection` |
/// | [`custom_words`](Self::custom_words) | empty | `customWords` |
/// | [`min_text_height`](Self::min_text_height) | `0.0` | `minimumTextHeight` — refused outside `0..=1` |
///
/// **Gates on what comes back, read per call.**
///
/// | option | default | what it drops |
/// |---|---|---|
/// | [`min_text_len`](Self::min_text_len) | `1` | a reading shorter than this many UTF-8 bytes |
/// | [`max_candidates_per_observation`](Self::max_candidates_per_observation) | `1` | every candidate past this many per region (Apple caps the list at 10) |
/// | [`min_confidence`](Self::min_confidence) | `0.0` | a reading Vision scored below this — refused outside `0..=1` |
///
/// Every default on the request is Apple's own except one:
/// [`detect_language`](Self::detect_language) is `true` where Vision's is
/// `false`. Vision's default roster is English alone, and a request left
/// there reads a Chinese or Japanese picture as Latin letters or as
/// nothing at all.
///
/// It is `Clone` and not `Copy`: the language roster and the custom
/// words are lists.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionTextOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_text_min_len"))]
  min_text_len: usize,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_text_max_candidates_per_observation")
  )]
  max_candidates_per_observation: usize,
  #[cfg_attr(feature = "serde", serde(default))]
  languages: Vec<String>,
  #[cfg_attr(feature = "serde", serde(default = "default_text_detect_language"))]
  detect_language: bool,
  #[cfg_attr(feature = "serde", serde(default = "default_text_recognition_level"))]
  recognition_level: TextRecognitionLevel,
  #[cfg_attr(feature = "serde", serde(default = "default_text_language_correction"))]
  language_correction: bool,
  #[cfg_attr(feature = "serde", serde(default))]
  custom_words: Vec<String>,
  #[cfg_attr(feature = "serde", serde(default = "default_text_min_text_height"))]
  min_text_height: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_text_min_confidence"))]
  min_confidence: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_text_revision"))]
  revision: usize,
}

impl AppleVisionTextOptions {
  /// Default [`min_text_len`](Self::min_text_len).
  pub const DEFAULT_MIN_TEXT_LEN: usize = default_text_min_len();
  /// Default [`max_candidates_per_observation`](Self::max_candidates_per_observation).
  pub const DEFAULT_MAX_CANDIDATES_PER_OBSERVATION: usize =
    default_text_max_candidates_per_observation();
  /// Default [`detect_language`](Self::detect_language).
  pub const DEFAULT_DETECT_LANGUAGE: bool = default_text_detect_language();
  /// Default [`recognition_level`](Self::recognition_level).
  pub const DEFAULT_RECOGNITION_LEVEL: TextRecognitionLevel = default_text_recognition_level();
  /// Default [`language_correction`](Self::language_correction).
  pub const DEFAULT_LANGUAGE_CORRECTION: bool = default_text_language_correction();
  /// Default [`min_text_height`](Self::min_text_height).
  pub const DEFAULT_MIN_TEXT_HEIGHT: f32 = default_text_min_text_height();
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_text_min_confidence();
  /// Default [`revision`](Self::revision): Apple's
  /// `VNRecognizeTextRequestRevision3`, the newest text revision the SDK
  /// names.
  pub const DEFAULT_REVISION: usize = default_text_revision();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_text_len: Self::DEFAULT_MIN_TEXT_LEN,
      max_candidates_per_observation: Self::DEFAULT_MAX_CANDIDATES_PER_OBSERVATION,
      languages: Vec::new(),
      detect_language: Self::DEFAULT_DETECT_LANGUAGE,
      recognition_level: Self::DEFAULT_RECOGNITION_LEVEL,
      language_correction: Self::DEFAULT_LANGUAGE_CORRECTION,
      custom_words: Vec::new(),
      min_text_height: Self::DEFAULT_MIN_TEXT_HEIGHT,
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
      revision: Self::DEFAULT_REVISION,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_text_len(mut self, min_text_len: usize) -> Self {
    self.set_min_text_len(min_text_len);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_text_len(&mut self, min_text_len: usize) -> &mut Self {
    self.min_text_len = min_text_len;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_text_len(&self) -> usize {
    self.min_text_len
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_candidates_per_observation(
    mut self,
    max_candidates_per_observation: usize,
  ) -> Self {
    self.set_max_candidates_per_observation(max_candidates_per_observation);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_candidates_per_observation(
    &mut self,
    max_candidates_per_observation: usize,
  ) -> &mut Self {
    self.max_candidates_per_observation = max_candidates_per_observation;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_candidates_per_observation(&self) -> usize {
    self.max_candidates_per_observation
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_languages<I, S>(mut self, languages: I) -> Self
  where
    I: IntoIterator<Item = S>,
    S: Into<String>,
  {
    self.set_languages(languages);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_languages<I, S>(&mut self, languages: I) -> &mut Self
  where
    I: IntoIterator<Item = S>,
    S: Into<String>,
  {
    self.languages = languages.into_iter().map(Into::into).collect();
    self
  }

  /// The recognition language roster, in priority order — Vision's
  /// `recognitionLanguages`.
  ///
  /// Each tag is passed to Vision exactly as written, and each must be
  /// one the request itself lists for its
  /// [`revision`](Self::revision) and
  /// [`recognition_level`](Self::recognition_level)
  /// (`supportedRecognitionLanguages`): `zh-Hans`, `zh-Hant`, `ja-JP`,
  /// `ko-KR`, `en-US` and so on at revision 3. Anything else is refused
  /// by name when the recognizer is built — including a spelling Vision
  /// would quietly take, such as a bare `ja`, because what Vision does
  /// with a tag it does not list is not something it promises.
  ///
  /// Empty, the default, is never sent, so the request keeps the roster
  /// Vision built it with: English alone.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn languages(&self) -> &[String] {
    self.languages.as_slice()
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_detect_language(mut self, detect_language: bool) -> Self {
    self.set_detect_language(detect_language);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_detect_language(&mut self, detect_language: bool) -> &mut Self {
    self.detect_language = detect_language;
    self
  }

  /// Whether Vision works out the script and language of each region
  /// for itself — `automaticallyDetectsLanguage`. Default `true`; Apple's
  /// own default is `false`.
  ///
  /// It is what reads a picture whose language nobody named: with it on,
  /// the default English roster still reads Chinese and Japanese. Apple
  /// added it in macOS 13 (iOS 16, tvOS 16); on an older system the
  /// recognizer is built without it and, with the `tracing` feature,
  /// says so once. Apple documents it as a no-op before revision 3.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn detect_language(&self) -> bool {
    self.detect_language
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_recognition_level(mut self, recognition_level: TextRecognitionLevel) -> Self {
    self.set_recognition_level(recognition_level);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_recognition_level(
    &mut self,
    recognition_level: TextRecognitionLevel,
  ) -> &mut Self {
    self.recognition_level = recognition_level;
    self
  }

  /// Accurate or fast — `recognitionLevel`. See [`TextRecognitionLevel`]
  /// for what the fast level does not read.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn recognition_level(&self) -> TextRecognitionLevel {
    self.recognition_level
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_language_correction(mut self, language_correction: bool) -> Self {
    self.set_language_correction(language_correction);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_language_correction(&mut self, language_correction: bool) -> &mut Self {
    self.language_correction = language_correction;
    self
  }

  /// Whether Vision corrects its readings against a lexicon —
  /// `usesLanguageCorrection`. Default `true`, as Apple's is; off returns
  /// the raw readings, faster and less accurate.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn language_correction(&self) -> bool {
    self.language_correction
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_custom_words<I, S>(mut self, custom_words: I) -> Self
  where
    I: IntoIterator<Item = S>,
    S: Into<String>,
  {
    self.set_custom_words(custom_words);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_custom_words<I, S>(&mut self, custom_words: I) -> &mut Self
  where
    I: IntoIterator<Item = S>,
    S: Into<String>,
  {
    self.custom_words = custom_words.into_iter().map(Into::into).collect();
    self
  }

  /// Words Vision should prefer over its standard lexicon when it reads
  /// — `customWords`: names, brands, terms of art. Empty by default, and
  /// an empty list is never sent.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn custom_words(&self) -> &[String] {
    self.custom_words.as_slice()
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_text_height(mut self, min_text_height: f32) -> Self {
    self.set_min_text_height(min_text_height);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_text_height(&mut self, min_text_height: f32) -> &mut Self {
    self.min_text_height = min_text_height;
    self
  }

  /// The smallest text Vision should look for, as a fraction of the
  /// image height — `minimumTextHeight`. `0.0`, the default and Apple's,
  /// reads the image at full resolution; larger is faster and misses
  /// smaller text. A value outside `0..=1`, or not a number, is refused
  /// by name when the recognizer is built.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_text_height(&self) -> f32 {
    self.min_text_height
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  /// The lowest confidence a reading may carry and still be emitted.
  /// Default `0.0`, which drops nothing.
  ///
  /// Read per call: Vision scores every candidate it returns, and a
  /// candidate scored below this is dropped after the request ran. A
  /// value outside `0..=1`, or not a number, is refused by name — by
  /// [`TextRecognizer::new`](crate::TextRecognizer::new) and by every
  /// call that is handed it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_revision(mut self, revision: usize) -> Self {
    self.set_revision(revision);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_revision(&mut self, revision: usize) -> &mut Self {
    self.revision = revision;
    self
  }

  /// The text request revision — `revision`, Apple's
  /// `VNRecognizeTextRequestRevision*`. Default `3`.
  ///
  /// A revision is a different engine behind the same API, so it changes
  /// what is read without changing any signature;
  /// [`TextRecognizer::revision`](crate::TextRecognizer::revision) reports
  /// the one a recognizer runs. A revision this host's Vision does not
  /// implement (`supportedRevisions`) is refused by name when the
  /// recognizer is built, rather than by every call after it. Revision 3
  /// needs macOS 13; revisions 1 and 2 are deprecated since macOS 15.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn revision(&self) -> usize {
    self.revision
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_body_pose_min_joint_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionBodyPoseOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_body_pose_min_joint_confidence")
  )]
  min_joint_confidence: f32,
}

impl AppleVisionBodyPoseOptions {
  /// Default [`min_joint_confidence`](Self::min_joint_confidence).
  pub const DEFAULT_MIN_JOINT_CONFIDENCE: f32 = default_body_pose_min_joint_confidence();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_joint_confidence: Self::DEFAULT_MIN_JOINT_CONFIDENCE,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_joint_confidence(mut self, min_joint_confidence: f32) -> Self {
    self.set_min_joint_confidence(min_joint_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_joint_confidence(&mut self, min_joint_confidence: f32) -> &mut Self {
    self.min_joint_confidence = min_joint_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_joint_confidence(&self) -> f32 {
    self.min_joint_confidence
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_hand_pose_min_joint_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_hand_pose_maximum_hand_count() -> usize {
  2
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionHandPoseOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_hand_pose_min_joint_confidence")
  )]
  min_joint_confidence: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_hand_pose_maximum_hand_count")
  )]
  maximum_hand_count: usize,
}

impl AppleVisionHandPoseOptions {
  /// Default [`min_joint_confidence`](Self::min_joint_confidence).
  pub const DEFAULT_MIN_JOINT_CONFIDENCE: f32 = default_hand_pose_min_joint_confidence();
  /// Default [`maximum_hand_count`](Self::maximum_hand_count).
  pub const DEFAULT_MAXIMUM_HAND_COUNT: usize = default_hand_pose_maximum_hand_count();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_joint_confidence: Self::DEFAULT_MIN_JOINT_CONFIDENCE,
      maximum_hand_count: Self::DEFAULT_MAXIMUM_HAND_COUNT,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_joint_confidence(mut self, min_joint_confidence: f32) -> Self {
    self.set_min_joint_confidence(min_joint_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_joint_confidence(&mut self, min_joint_confidence: f32) -> &mut Self {
    self.min_joint_confidence = min_joint_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_joint_confidence(&self) -> f32 {
    self.min_joint_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_maximum_hand_count(mut self, maximum_hand_count: usize) -> Self {
    self.set_maximum_hand_count(maximum_hand_count);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_maximum_hand_count(&mut self, maximum_hand_count: usize) -> &mut Self {
    self.maximum_hand_count = maximum_hand_count;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn maximum_hand_count(&self) -> usize {
    self.maximum_hand_count
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_animal_pose_min_joint_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionAnimalPoseOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_animal_pose_min_joint_confidence")
  )]
  min_joint_confidence: f32,
}

impl AppleVisionAnimalPoseOptions {
  /// Default [`min_joint_confidence`](Self::min_joint_confidence).
  pub const DEFAULT_MIN_JOINT_CONFIDENCE: f32 = default_animal_pose_min_joint_confidence();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_joint_confidence: Self::DEFAULT_MIN_JOINT_CONFIDENCE,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_joint_confidence(mut self, min_joint_confidence: f32) -> Self {
    self.set_min_joint_confidence(min_joint_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_joint_confidence(&mut self, min_joint_confidence: f32) -> &mut Self {
    self.min_joint_confidence = min_joint_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_joint_confidence(&self) -> f32 {
    self.min_joint_confidence
  }
}

/// The 3-D body-pose section — deliberately empty.
///
/// It held a `min_joint_confidence` floor, which gated a reading
/// Apple's 3-D road does not report: the whole point hierarchy
/// (`VNPoint3D` → `VNRecognizedPoint3D` →
/// `VNHumanBodyRecognizedPoint3D`) declares `position`, `identifier`,
/// `localPosition` and `parentJoint` and no confidence at any level, so
/// the floor had nothing to compare a joint against. It never compared
/// one: the send that fetched the missing value raised instead, so
/// every 3-D joint was dropped in every released version and the gate
/// ran zero times. Removing it removes no behaviour anyone could have
/// observed.
///
/// The section stays. It is where a 3-D knob would land if Apple ever
/// bakes one into `VNDetectHumanBodyPose3DRequest`, and it keeps
/// [`AppleVisionBodyPoserOptions`]'s two-section shape. Like every
/// options type in this crate it is unknown-field tolerant, so a config
/// still naming `min_joint_confidence` under `pose_3d` parses and the
/// key is dropped.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionBodyPose3DOptions {}

impl AppleVisionBodyPose3DOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {}
  }
}

/// Everything [`BodyPoser`](crate::BodyPoser) reads — one section per
/// pose dimensionality, because
/// [`detect_2d`](crate::BodyPoser::detect_2d) and
/// [`detect_3d`](crate::BodyPoser::detect_3d) run different models, and
/// what each reports per joint is not the same. The 2-D section carries
/// a joint-confidence floor; the 3-D section carries nothing, because
/// its model reports nothing per joint.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionBodyPoserOptions {
  #[cfg_attr(feature = "serde", serde(default))]
  pose_2d: AppleVisionBodyPoseOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  pose_3d: AppleVisionBodyPose3DOptions,
}

impl AppleVisionBodyPoserOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      pose_2d: AppleVisionBodyPoseOptions::new(),
      pose_3d: AppleVisionBodyPose3DOptions::new(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn pose_2d(&self) -> AppleVisionBodyPoseOptions {
    self.pose_2d
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn pose_2d_mut(&mut self) -> &mut AppleVisionBodyPoseOptions {
    &mut self.pose_2d
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn pose_3d(&self) -> AppleVisionBodyPose3DOptions {
    self.pose_3d
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn pose_3d_mut(&mut self) -> &mut AppleVisionBodyPose3DOptions {
    &mut self.pose_3d
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_face_capture_min_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_face_capture_min_capture_quality() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionFaceCaptureOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_face_capture_min_confidence")
  )]
  min_confidence: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_face_capture_min_capture_quality")
  )]
  min_capture_quality: f32,
}

impl AppleVisionFaceCaptureOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_face_capture_min_confidence();
  /// Default [`min_capture_quality`](Self::min_capture_quality).
  pub const DEFAULT_MIN_CAPTURE_QUALITY: f32 = default_face_capture_min_capture_quality();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
      min_capture_quality: Self::DEFAULT_MIN_CAPTURE_QUALITY,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_capture_quality(mut self, min_capture_quality: f32) -> Self {
    self.set_min_capture_quality(min_capture_quality);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_capture_quality(&mut self, min_capture_quality: f32) -> &mut Self {
    self.min_capture_quality = min_capture_quality;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_capture_quality(&self) -> f32 {
    self.min_capture_quality
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_face_rectangle_min_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionFaceRectangleOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_face_rectangle_min_confidence")
  )]
  min_confidence: f32,
}

impl AppleVisionFaceRectangleOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_face_rectangle_min_confidence();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_face_keypoints_min_confidence() -> f32 {
  0.1
}

/// Gates for the landmark pass [`FaceDetector`](crate::FaceDetector)
/// reduces to [`FaceKeypoints`](crate::FaceKeypoints).
///
/// Separate from [`AppleVisionFaceLandmarkOptions`], which configures
/// the full thirteen-region [`FaceLandmarker`](crate::FaceLandmarker):
/// the two run different passes for different consumers and their
/// thresholds are not required to move together.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionFaceKeypointsOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_face_keypoints_min_confidence")
  )]
  min_confidence: f32,
}

impl AppleVisionFaceKeypointsOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_face_keypoints_min_confidence();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }
}

/// Everything [`FaceDetector`](crate::FaceDetector) reads — one
/// section per Vision pass it fuses.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionFaceOptions {
  #[cfg_attr(feature = "serde", serde(default))]
  rectangles: AppleVisionFaceRectangleOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  capture: AppleVisionFaceCaptureOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  keypoints: AppleVisionFaceKeypointsOptions,
}

impl AppleVisionFaceOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      rectangles: AppleVisionFaceRectangleOptions::new(),
      capture: AppleVisionFaceCaptureOptions::new(),
      keypoints: AppleVisionFaceKeypointsOptions::new(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rectangles(&self) -> AppleVisionFaceRectangleOptions {
    self.rectangles
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rectangles_mut(&mut self) -> &mut AppleVisionFaceRectangleOptions {
    &mut self.rectangles
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn capture(&self) -> AppleVisionFaceCaptureOptions {
    self.capture
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn capture_mut(&mut self) -> &mut AppleVisionFaceCaptureOptions {
    &mut self.capture
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn keypoints(&self) -> AppleVisionFaceKeypointsOptions {
    self.keypoints
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn keypoints_mut(&mut self) -> &mut AppleVisionFaceKeypointsOptions {
    &mut self.keypoints
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_face_landmark_min_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_face_landmark_min_region_count() -> usize {
  1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionFaceLandmarkOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_face_landmark_min_confidence")
  )]
  min_confidence: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_face_landmark_min_region_count")
  )]
  min_region_count: usize,
}

impl AppleVisionFaceLandmarkOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_face_landmark_min_confidence();
  /// Default [`min_region_count`](Self::min_region_count).
  pub const DEFAULT_MIN_REGION_COUNT: usize = default_face_landmark_min_region_count();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
      min_region_count: Self::DEFAULT_MIN_REGION_COUNT,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_region_count(mut self, min_region_count: usize) -> Self {
    self.set_min_region_count(min_region_count);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_region_count(&mut self, min_region_count: usize) -> &mut Self {
    self.min_region_count = min_region_count;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_region_count(&self) -> usize {
    self.min_region_count
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_human_subject_min_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionHumanSubjectOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_human_subject_min_confidence")
  )]
  min_confidence: f32,
}

impl AppleVisionHumanSubjectOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_human_subject_min_confidence();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_barcode_min_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_barcode_min_payload_len() -> usize {
  1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionBarcodeOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_barcode_min_confidence"))]
  min_confidence: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_barcode_min_payload_len"))]
  min_payload_len: usize,
}

impl AppleVisionBarcodeOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_barcode_min_confidence();
  /// Default [`min_payload_len`](Self::min_payload_len).
  pub const DEFAULT_MIN_PAYLOAD_LEN: usize = default_barcode_min_payload_len();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
      min_payload_len: Self::DEFAULT_MIN_PAYLOAD_LEN,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_payload_len(mut self, min_payload_len: usize) -> Self {
    self.set_min_payload_len(min_payload_len);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_payload_len(&mut self, min_payload_len: usize) -> &mut Self {
    self.min_payload_len = min_payload_len;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_payload_len(&self) -> usize {
    self.min_payload_len
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_saliency_min_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_saliency_max_regions() -> usize {
  16
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionSaliencyOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_saliency_min_confidence"))]
  min_confidence: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_saliency_max_regions"))]
  max_regions: usize,
}

impl AppleVisionSaliencyOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_saliency_min_confidence();
  /// Default [`max_regions`](Self::max_regions).
  pub const DEFAULT_MAX_REGIONS: usize = default_saliency_max_regions();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
      max_regions: Self::DEFAULT_MAX_REGIONS,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_regions(mut self, max_regions: usize) -> Self {
    self.set_max_regions(max_regions);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_regions(&mut self, max_regions: usize) -> &mut Self {
    self.max_regions = max_regions;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_regions(&self) -> usize {
    self.max_regions
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_horizon_min_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionHorizonOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_horizon_min_confidence"))]
  min_confidence: f32,
}

impl AppleVisionHorizonOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_horizon_min_confidence();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_document_segmentation_min_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_document_segmentation_max_segments() -> usize {
  16
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionDocumentSegmentationOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_document_segmentation_min_confidence")
  )]
  min_confidence: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_document_segmentation_max_segments")
  )]
  max_segments: usize,
}

impl AppleVisionDocumentSegmentationOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_document_segmentation_min_confidence();
  /// Default [`max_segments`](Self::max_segments).
  pub const DEFAULT_MAX_SEGMENTS: usize = default_document_segmentation_max_segments();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
      max_segments: Self::DEFAULT_MAX_SEGMENTS,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_segments(mut self, max_segments: usize) -> Self {
    self.set_max_segments(max_segments);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_segments(&mut self, max_segments: usize) -> &mut Self {
    self.max_segments = max_segments;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_segments(&self) -> usize {
    self.max_segments
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_aesthetics_min_overall_score() -> f32 {
  -1.0
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionAestheticsOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_aesthetics_min_overall_score")
  )]
  min_overall_score: f32,
}

impl AppleVisionAestheticsOptions {
  /// Default [`min_overall_score`](Self::min_overall_score).
  pub const DEFAULT_MIN_OVERALL_SCORE: f32 = default_aesthetics_min_overall_score();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_overall_score: Self::DEFAULT_MIN_OVERALL_SCORE,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_overall_score(mut self, min_overall_score: f32) -> Self {
    self.set_min_overall_score(min_overall_score);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_overall_score(&mut self, min_overall_score: f32) -> &mut Self {
    self.min_overall_score = min_overall_score;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_overall_score(&self) -> f32 {
    self.min_overall_score
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_person_instance_mask_min_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_person_instance_mask_max_instances_per_observation() -> usize {
  16
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionPersonInstanceMaskOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_person_instance_mask_min_confidence")
  )]
  min_confidence: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_person_instance_mask_max_instances_per_observation")
  )]
  max_instances_per_observation: usize,
}

impl AppleVisionPersonInstanceMaskOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_person_instance_mask_min_confidence();
  /// Default [`max_instances_per_observation`](Self::max_instances_per_observation).
  pub const DEFAULT_MAX_INSTANCES_PER_OBSERVATION: usize =
    default_person_instance_mask_max_instances_per_observation();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
      max_instances_per_observation: Self::DEFAULT_MAX_INSTANCES_PER_OBSERVATION,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_instances_per_observation(
    mut self,
    max_instances_per_observation: usize,
  ) -> Self {
    self.set_max_instances_per_observation(max_instances_per_observation);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_instances_per_observation(
    &mut self,
    max_instances_per_observation: usize,
  ) -> &mut Self {
    self.max_instances_per_observation = max_instances_per_observation;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_instances_per_observation(&self) -> usize {
    self.max_instances_per_observation
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_person_segmentation_min_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionPersonSegmentationOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_person_segmentation_min_confidence")
  )]
  min_confidence: f32,
}

impl AppleVisionPersonSegmentationOptions {
  /// Default [`min_confidence`](Self::min_confidence).
  pub const DEFAULT_MIN_CONFIDENCE: f32 = default_person_segmentation_min_confidence();

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }
}

/// Everything [`PersonMasker`](crate::PersonMasker) reads — one
/// section per mask kind, because
/// [`instance_masks`](crate::PersonMasker::instance_masks) and
/// [`segmentation_masks`](crate::PersonMasker::segmentation_masks)
/// run different models whose gates are not required to move
/// together.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionPersonMaskerOptions {
  #[cfg_attr(feature = "serde", serde(default))]
  instances: AppleVisionPersonInstanceMaskOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  segmentation: AppleVisionPersonSegmentationOptions,
}

impl AppleVisionPersonMaskerOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      instances: AppleVisionPersonInstanceMaskOptions::new(),
      segmentation: AppleVisionPersonSegmentationOptions::new(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn instances(&self) -> AppleVisionPersonInstanceMaskOptions {
    self.instances
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn instances_mut(&mut self) -> &mut AppleVisionPersonInstanceMaskOptions {
    &mut self.instances
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn segmentation(&self) -> AppleVisionPersonSegmentationOptions {
    self.segmentation
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn segmentation_mut(&mut self) -> &mut AppleVisionPersonSegmentationOptions {
    &mut self.segmentation
  }
}

#[cfg(feature = "serde")]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_num_workers() -> usize {
  AnalyzeOptions::DEFAULT_NUM_WORKERS
}

// Mirror the `with_workers` / `set_workers` 0 -> 1 coercion at the serde
// boundary so a `{"num_workers": 0}` config can't silently produce a
// zero-worker service.
#[cfg(feature = "serde")]
fn deserialize_num_workers<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
  D: serde::Deserializer<'de>,
{
  let n = usize::deserialize(deserializer)?;
  Ok(if n == 0 { 1 } else { n })
}

/// Everything [`VisionAnalyzer`](crate::VisionAnalyzer) reads — one
/// section per core detection, and nothing else.
///
/// The eight sections here are exactly the eight
/// [`Analysis`](crate::Analysis) slots. Text, barcodes, faces,
/// landmarks, poses and masks configure their own entry points
/// through their own options types, so a config that only enables
/// text no longer carries eleven sections that nothing reads.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnalyzeOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_num_workers",
      deserialize_with = "deserialize_num_workers"
    )
  )]
  num_workers: usize,
  #[cfg_attr(feature = "serde", serde(default))]
  classifications: AppleVisionClassificationOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  human_subjects: AppleVisionHumanSubjectOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  animals: AppleVisionAnimalOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  attention_saliency: AppleVisionSaliencyOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  objectness_saliency: AppleVisionSaliencyOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  horizon: AppleVisionHorizonOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  document_segments: AppleVisionDocumentSegmentationOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  aesthetics: AppleVisionAestheticsOptions,
}

impl AnalyzeOptions {
  /// Default [`num_workers`](Self::num_workers).
  pub const DEFAULT_NUM_WORKERS: usize = 1;

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      num_workers: Self::DEFAULT_NUM_WORKERS,
      classifications: AppleVisionClassificationOptions::new(),
      human_subjects: AppleVisionHumanSubjectOptions::new(),
      animals: AppleVisionAnimalOptions::new(),
      attention_saliency: AppleVisionSaliencyOptions::new(),
      objectness_saliency: AppleVisionSaliencyOptions::new(),
      horizon: AppleVisionHorizonOptions::new(),
      document_segments: AppleVisionDocumentSegmentationOptions::new(),
      aesthetics: AppleVisionAestheticsOptions::new(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_workers(mut self, num_workers: usize) -> Self {
    self.set_workers(num_workers);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_workers(&mut self, num_workers: usize) -> &mut Self {
    self.num_workers = if num_workers == 0 { 1 } else { num_workers };
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn num_workers(&self) -> usize {
    self.num_workers
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn classifications(&self) -> AppleVisionClassificationOptions {
    self.classifications
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn classifications_mut(&mut self) -> &mut AppleVisionClassificationOptions {
    &mut self.classifications
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn human_subjects(&self) -> AppleVisionHumanSubjectOptions {
    self.human_subjects
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn human_subjects_mut(&mut self) -> &mut AppleVisionHumanSubjectOptions {
    &mut self.human_subjects
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn animals(&self) -> AppleVisionAnimalOptions {
    self.animals
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn animals_mut(&mut self) -> &mut AppleVisionAnimalOptions {
    &mut self.animals
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn attention_saliency(&self) -> AppleVisionSaliencyOptions {
    self.attention_saliency
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn attention_saliency_mut(&mut self) -> &mut AppleVisionSaliencyOptions {
    &mut self.attention_saliency
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn objectness_saliency(&self) -> AppleVisionSaliencyOptions {
    self.objectness_saliency
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn objectness_saliency_mut(&mut self) -> &mut AppleVisionSaliencyOptions {
    &mut self.objectness_saliency
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn horizon(&self) -> AppleVisionHorizonOptions {
    self.horizon
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn horizon_mut(&mut self) -> &mut AppleVisionHorizonOptions {
    &mut self.horizon
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn document_segments(&self) -> AppleVisionDocumentSegmentationOptions {
    self.document_segments
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn document_segments_mut(&mut self) -> &mut AppleVisionDocumentSegmentationOptions {
    &mut self.document_segments
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn aesthetics(&self) -> AppleVisionAestheticsOptions {
    self.aesthetics
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn aesthetics_mut(&mut self) -> &mut AppleVisionAestheticsOptions {
    &mut self.aesthetics
  }
}
