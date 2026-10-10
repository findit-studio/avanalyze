//! The text request's options against the real framework: the request
//! carries what the options say, every refusal names what it refused,
//! and a picture with Chinese and Japanese on it is read in those
//! languages — where the request this crate built before the options
//! existed read English alone.
//!
//! The picture is set by Core Text, in the test, into the same kind of
//! Core Graphics bitmap the pixel-door tests draw into: no fixture to
//! vendor, and no doubt about what it says.
//!
//! One refusal has no law here. Below the recognizer's floor — macOS 13,
//! iOS 16, tvOS 16, visionOS 1 — `TextRecognizer::new` refuses with
//! `AnalyzeErrorKind::Unsupported` before anything is sent, and that
//! branch cannot run on a system this suite runs on: every one of them
//! is at or above the floor, as the laws below need it to be. It is not
//! faked.

use core::{convert::Infallible, ffi::c_void};

use objc2::{ClassType, msg_send, rc::Retained};
use objc2_core_foundation::{CFAttributedString, CFDictionary, CFString, CGPoint, CGRect, CGSize};
use objc2_core_graphics::{CGBitmapContextCreate, CGColorSpace, CGContext, CGImageAlphaInfo};
use objc2_core_text::{CTFont, CTLine, kCTFontAttributeName};
use objc2_foundation::{NSArray, NSIndexSet, NSNotFound, NSString};
use objc2_vision::{
  VNRecognizeTextRequest, VNRecognizeTextRequestRevision3, VNRequestTextRecognitionLevel,
};

use crate::{
  AnalyzeErrorKind, AppleVisionTextOptions, BoundingBox, PixelFormat, PixelPlane, TextDetection,
  TextRecognitionLevel, TextRecognizer,
};

/// The Chinese line: "hello, world".
const CHINESE: &str = "你好世界";

/// The Japanese line: "hello, world", in kana and kanji.
const JAPANESE: &str = "こんにちは世界";

/// The English line, which every request below reads.
const ENGLISH: &str = "Hello World 2026";

/// The page: each line in a face that carries its script, top to
/// bottom. A face missing on some host would not leave a blank: Core
/// Text falls back to one that has the glyphs.
const LINES: [(&str, &str); 3] = [
  ("PingFang SC", CHINESE),
  ("Hiragino Sans", JAPANESE),
  ("Helvetica", ENGLISH),
];

/// The page's width and height, in pixels.
const PAGE: (usize, usize) = (1400, 560);

// ----- a vocabulary that keeps what Vision read -----------------------------

/// A box that stores what it is given. The engine's own guards are
/// asserted elsewhere; this one refuses nothing, on purpose.
#[derive(Debug, Clone, Copy)]
struct Bbox {
  x: f32,
  y: f32,
  width: f32,
  height: f32,
}

impl BoundingBox for Bbox {
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

/// One reading: the string and the confidence Vision gave it.
#[derive(Debug, Clone)]
struct Reading {
  text: String,
  confidence: f32,
}

impl TextDetection for Reading {
  type Error = Infallible;
  type BoundingBox = Bbox;

  fn try_new(
    text: &str,
    confidence: f32,
    _bbox: Self::BoundingBox,
    _observation: usize,
    _rank: usize,
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      text: text.to_owned(),
      confidence,
    })
  }
}

// ----- the page -------------------------------------------------------------

/// Sets [`LINES`] in black on a blank [`PAGE`] and hands back packed
/// RGBA8 whose fourth byte is padding — [`PixelFormat::Rgba8`], whose
/// alpha the engine never reads.
fn page() -> Vec<u8> {
  let (width, height) = PAGE;
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
  CGContext::set_rgb_fill_color(Some(&context), 1.0, 1.0, 1.0, 1.0);
  CGContext::fill_rect(
    Some(&context),
    CGRect {
      origin: CGPoint { x: 0.0, y: 0.0 },
      size: CGSize {
        width: width as f64,
        height: height as f64,
      },
    },
  );
  // Core Text's default foreground is black, so each line names its
  // face and nothing else.
  let pitch = height as f64 / (LINES.len() as f64 + 1.0);
  for (index, (face, text)) in LINES.iter().enumerate() {
    // SAFETY: a null matrix is the documented identity.
    let font = unsafe { CTFont::with_name(&CFString::from_str(face), 72.0, core::ptr::null()) };
    // SAFETY: a constant Core Text exports, read and not written.
    let key: &CFString = unsafe { kCTFontAttributeName };
    let attributes = CFDictionary::<CFString, CTFont>::from_slices(&[key], &[&*font]);
    // SAFETY: the default allocator, a valid string, and a dictionary
    // whose one key maps to the type Core Text reads it as.
    let string = unsafe {
      CFAttributedString::new(
        None,
        Some(&CFString::from_str(text)),
        Some(attributes.as_opaque()),
      )
    }
    .expect("an attributed string");
    // SAFETY: a valid attributed string.
    let line = unsafe { CTLine::with_attributed_string(&string) };
    // Core Graphics counts up from the bottom; the lines go down from
    // the top.
    CGContext::set_text_position(
      Some(&context),
      60.0,
      height as f64 - pitch * (index as f64 + 1.0),
    );
    // SAFETY: a valid line and a live context.
    unsafe { line.draw(&context) };
  }
  drop(context);
  pixels
}

/// How many pixels on `rgba` carry ink.
fn ink(rgba: &[u8]) -> usize {
  rgba
    .as_chunks::<4>()
    .0
    .iter()
    .filter(|pixel| pixel[0] < 128)
    .count()
}

/// `pixels` as the plane [`page`] laid them out in.
fn plane(pixels: &[u8]) -> PixelPlane<'_> {
  let (width, height) = PAGE;
  PixelPlane::packed(pixels, width as u32, height as u32, PixelFormat::Rgba8).expect("the page")
}

/// What a recognizer built from `options` reads on `pixels`.
fn read(options: &AppleVisionTextOptions, pixels: &[u8]) -> Vec<Reading> {
  TextRecognizer::new(options)
    .expect("the options are ones the request can be")
    .recognize_pixels::<Reading>(&plane(pixels), options)
    .expect("the pixel door must return Ok")
}

/// Every reading, in order, for an assertion to search and print.
fn joined(readings: &[Reading]) -> String {
  readings
    .iter()
    .map(|reading| reading.text.as_str())
    .collect::<Vec<_>>()
    .join(" | ")
}

/// Kana and CJK ideographs — any of them in a reading means the
/// Chinese or Japanese line was read in its own script.
fn is_chinese_or_japanese(c: char) -> bool {
  matches!(
    c,
    '\u{3040}'..='\u{30FF}' | '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}' | '\u{F900}'..='\u{FAFF}'
  )
}

/// Whether some reading is a line read in its own script: more than half
/// of its characters, spaces aside, are kana or CJK ideographs, and it
/// carries every one of `marks`.
fn read_in_its_own_script(readings: &[Reading], marks: &[&str]) -> bool {
  readings.iter().any(|reading| {
    let (cjk, all) = reading
      .text
      .chars()
      .filter(|c| !c.is_whitespace())
      .fold((0usize, 0usize), |(cjk, all), c| {
        (cjk + usize::from(is_chinese_or_japanese(c)), all + 1)
      });
    cjk * 2 > all && marks.iter().all(|mark| reading.text.contains(mark))
  })
}

// ----- the request, read back ----------------------------------------------

/// The request's language roster.
fn roster(request: &VNRecognizeTextRequest) -> Vec<String> {
  // SAFETY: a plain property read on a live request.
  let roster = unsafe { request.recognitionLanguages() };
  roster.iter().map(|tag| tag.to_string()).collect()
}

/// The request's custom words, or `None` where Vision holds nil.
///
/// Read by hand rather than through the binding: the binding declares
/// `customWords` non-null, and its safe getter panics on the nil a
/// fresh request answers.
fn custom_words(request: &VNRecognizeTextRequest) -> Option<Vec<String>> {
  // SAFETY: the selector is the property's getter, which returns an
  // `NSArray<NSString *>` or nil and takes no argument.
  let words: Option<Retained<NSArray<NSString>>> = unsafe { msg_send![request, customWords] };
  words.map(|words| words.iter().map(|word| word.to_string()).collect())
}

/// The revisions the text request class implements on this host —
/// asked of the class itself, as [`TextRecognizer::new`] asks it.
fn implemented_revisions() -> Vec<usize> {
  // SAFETY: a class property getter, returning an `NSIndexSet`.
  let revisions: Retained<NSIndexSet> =
    unsafe { msg_send![VNRecognizeTextRequest::class(), supportedRevisions] };
  let not_found = NSNotFound as usize;
  let mut implemented = Vec::new();
  let mut index = revisions.firstIndex();
  while index != not_found {
    implemented.push(index);
    index = revisions.indexGreaterThanIndex(index);
  }
  implemented
}

/// The tags a text request at `revision` and `level` lists.
fn listed_languages(revision: usize, level: VNRequestTextRecognitionLevel) -> Vec<String> {
  let request = VNRecognizeTextRequest::new();
  // SAFETY: plain property writes and a documented query on a live
  // request.
  let listed = unsafe {
    request.setRevision(revision);
    request.setRecognitionLevel(level);
    request.supportedRecognitionLanguagesAndReturnError()
  }
  .expect("the request lists the languages it reads");
  listed.iter().map(|tag| tag.to_string()).collect()
}

/// `TextRecognizer::new(options)`'s refusal, which must be by name.
fn refused(options: &AppleVisionTextOptions) -> String {
  let error = TextRecognizer::new(options).expect_err("the options must be refused");
  assert_eq!(
    error.kind(),
    AnalyzeErrorKind::InvalidOptions,
    "a refusal of the options, not of the host: {error}"
  );
  error.message().to_owned()
}

// ----- the laws ------------------------------------------------------------

/// The defaults leave the request as Apple builds it, save the one
/// default this crate changes: language detection is on.
///
/// The fresh request is the comparison, not a list of Apple's values
/// written down here, so the law holds whatever an OS makes those
/// values — and fails if an option this crate calls "Apple's default"
/// ever is not.
#[test]
fn the_defaults_are_apples_own_except_language_detection() {
  let options = AppleVisionTextOptions::new();
  assert!(options.languages().is_empty());
  assert!(options.custom_words().is_empty());
  assert!(options.detect_language());
  assert_eq!(options.recognition_level(), TextRecognitionLevel::Accurate);
  assert!(options.language_correction());
  assert_eq!(options.min_text_height(), 0.0);
  assert_eq!(options.min_confidence(), 0.0);
  assert_eq!(options.revision(), VNRecognizeTextRequestRevision3);

  let recognizer = TextRecognizer::new(&options).expect("the defaults build");
  let request = recognizer.request();
  let fresh = VNRecognizeTextRequest::new();
  // SAFETY: a plain property write on a live request.
  unsafe { fresh.setRevision(VNRecognizeTextRequestRevision3) };

  assert_eq!(
    roster(request),
    roster(&fresh),
    "an empty roster leaves Vision's own"
  );
  assert_eq!(
    custom_words(request),
    custom_words(&fresh),
    "no custom words leaves Vision's own nil"
  );
  assert_eq!(request.recognitionLevel(), fresh.recognitionLevel());
  assert_eq!(
    request.usesLanguageCorrection(),
    fresh.usesLanguageCorrection()
  );
  assert_eq!(request.minimumTextHeight(), fresh.minimumTextHeight());
  assert_eq!(recognizer.revision(), VNRecognizeTextRequestRevision3);

  assert!(
    request.automaticallyDetectsLanguage(),
    "detection is on by default"
  );
  assert!(
    !fresh.automaticallyDetectsLanguage(),
    "Apple's own default is off, which is why this one is not"
  );
}

/// Every request property the options name is on the request, read
/// back through the request's own getters.
#[test]
fn the_request_carries_what_the_options_say() {
  let options = AppleVisionTextOptions::new()
    .with_languages(["zh-Hans", "ja-JP", "en-US"])
    .with_detect_language(false)
    .with_language_correction(false)
    .with_custom_words(["avanalyze", CHINESE])
    .with_min_text_height(0.25);
  let recognizer = TextRecognizer::new(&options).expect("listed languages build");
  let request = recognizer.request();
  assert_eq!(roster(request), ["zh-Hans", "ja-JP", "en-US"]);
  assert!(!request.automaticallyDetectsLanguage());
  assert_eq!(
    request.recognitionLevel(),
    VNRequestTextRecognitionLevel::Accurate
  );
  assert!(!request.usesLanguageCorrection());
  assert_eq!(
    custom_words(request),
    Some(vec!["avanalyze".to_owned(), CHINESE.to_owned()])
  );
  assert_eq!(request.minimumTextHeight(), 0.25);
  assert_eq!(recognizer.revision(), VNRecognizeTextRequestRevision3);

  // The fast level, with a roster it lists and detection left on.
  let fast = AppleVisionTextOptions::new()
    .with_recognition_level(TextRecognitionLevel::Fast)
    .with_languages(["fr-FR"])
    .with_min_text_height(1.0);
  let recognizer = TextRecognizer::new(&fast).expect("a language the fast level lists builds");
  let request = recognizer.request();
  assert_eq!(
    request.recognitionLevel(),
    VNRequestTextRecognitionLevel::Fast
  );
  assert_eq!(roster(request), ["fr-FR"]);
  assert!(request.automaticallyDetectsLanguage());
  assert!(request.usesLanguageCorrection());
  assert_eq!(request.minimumTextHeight(), 1.0);
}

/// Every revision the host implements is taken, and is the revision the
/// request then runs — read off the request, not off the options.
///
/// Below revision 3 the options also turn detection off, which those
/// revisions cannot do; that refusal has its own law.
#[test]
fn every_implemented_revision_is_the_one_the_request_runs() {
  let revisions = implemented_revisions();
  assert!(
    revisions.contains(&VNRecognizeTextRequestRevision3),
    "the default revision is implemented on any host this suite runs on: {revisions:?}"
  );
  for revision in revisions {
    let options = AppleVisionTextOptions::new()
      .with_revision(revision)
      .with_detect_language(revision >= VNRecognizeTextRequestRevision3);
    let recognizer = TextRecognizer::new(&options)
      .unwrap_or_else(|error| panic!("revision {revision} is implemented and must build: {error}"));
    assert_eq!(recognizer.revision(), revision);
    // SAFETY: a plain property read on a live request.
    assert_eq!(unsafe { recognizer.request().revision() }, revision);
  }
}

/// A revision the host does not implement is refused at construction,
/// by name. Vision itself takes the setter and fails every frame after.
#[test]
fn an_unimplemented_revision_is_refused_by_name() {
  let implemented = implemented_revisions()
    .iter()
    .map(ToString::to_string)
    .collect::<Vec<_>>()
    .join(", ");
  for revision in [0, 99, usize::MAX] {
    let message = refused(&AppleVisionTextOptions::new().with_revision(revision));
    assert!(
      message.contains(&format!("revision {revision} ")),
      "the refusal names the revision: {message}"
    );
    assert!(
      message.ends_with(&format!("it implements {implemented}")),
      "and the ones the host implements: {message}"
    );
  }
}

/// Language detection asked for where it cannot act is refused by name,
/// with what would be taken instead.
///
/// Below revision 3 Apple documents detection as a no-op, so Vision
/// would build the request and read on its English-only default roster:
/// the reading this option exists to end, returned as a success. It runs
/// here and in CI's lane, because both implement revision 2.
#[test]
fn language_detection_where_it_cannot_act_is_refused_by_name() {
  let implemented = implemented_revisions();
  let below: Vec<usize> = implemented
    .iter()
    .copied()
    .filter(|revision| *revision < VNRecognizeTextRequestRevision3)
    .collect();
  assert!(
    below.contains(&2),
    "revision 2 is implemented on any host this suite runs on: {implemented:?}"
  );
  for revision in below {
    for options in [
      AppleVisionTextOptions::new().with_revision(revision),
      AppleVisionTextOptions::new()
        .with_revision(revision)
        .with_languages(["en-US"]),
    ] {
      let message = refused(&options);
      assert!(
        message.contains(&format!(
          "detect_language is true, and revision {revision} cannot detect a language"
        )),
        "the refusal names the option and the revision: {message}"
      );
      assert!(
        message.ends_with(
          "set detect_language = false, or ask for revision 3, which this host implements"
        ),
        "and what would be taken: {message}"
      );
    }
  }
}

/// The other side of that refusal: without detection, revision 2 and a
/// roster it lists are taken, and the request runs at revision 2 and
/// reads the English line.
#[test]
fn revision_2_without_detection_and_with_a_roster_is_taken_and_reads_english() {
  let options = AppleVisionTextOptions::new()
    .with_revision(2)
    .with_detect_language(false)
    .with_languages(["en-US"]);
  let recognizer =
    TextRecognizer::new(&options).expect("revision 2, without detection, with a listed roster");
  assert_eq!(recognizer.revision(), 2);
  assert_eq!(roster(recognizer.request()), ["en-US"]);
  assert!(!recognizer.request().automaticallyDetectsLanguage());

  let pixels = page();
  let read = joined(
    &recognizer
      .recognize_pixels::<Reading>(&plane(&pixels), &options)
      .expect("the pixel door must return Ok"),
  );
  assert!(
    read.contains(ENGLISH),
    "revision 2 reads the English line: {read}"
  );
}

/// A language the request does not list, for its revision and its
/// recognition level, is refused by name — and every one it lists is
/// taken.
///
/// Vision would take any of the refused ones without a word: handed a
/// tag it does not list, the request reads as though the tag were not
/// there.
#[test]
fn a_language_the_request_does_not_list_is_refused_by_name() {
  let listed = listed_languages(
    VNRecognizeTextRequestRevision3,
    VNRequestTextRecognitionLevel::Accurate,
  );
  assert!(
    listed.iter().any(|tag| tag == "zh-Hans") && listed.iter().any(|tag| tag == "ja-JP"),
    "revision 3 reads Chinese and Japanese at the accurate level: {listed:?}"
  );
  for tag in &listed {
    TextRecognizer::new(&AppleVisionTextOptions::new().with_languages([tag.as_str()]))
      .unwrap_or_else(|error| panic!("{tag} is listed and must be taken: {error}"));
  }

  // A tag nothing lists.
  let message = refused(&AppleVisionTextOptions::new().with_languages(["zh-Hans", "xx-YY"]));
  assert!(
    message.contains("\"xx-YY\""),
    "the refusal names the tag: {message}"
  );
  assert!(
    message.ends_with(&format!("it reads {}", listed.join(", "))),
    "and lists what the request reads: {message}"
  );

  // An empty tag is a tag nothing lists.
  let message = refused(&AppleVisionTextOptions::new().with_languages([""]));
  assert!(message.contains("\"\""), "{message}");

  // The match is exact: Vision lists `ja-JP`, so a bare `ja` is refused
  // and the refusal shows the spelling it lists.
  if !listed.iter().any(|tag| tag == "ja") {
    let message = refused(&AppleVisionTextOptions::new().with_languages(["ja"]));
    assert!(
      message.contains("\"ja\"") && message.contains("ja-JP"),
      "{message}"
    );
  }

  // A tag the accurate level lists and the fast level does not.
  let fast = listed_languages(
    VNRecognizeTextRequestRevision3,
    VNRequestTextRecognitionLevel::Fast,
  );
  assert!(
    !fast.iter().any(|tag| tag == "zh-Hans"),
    "the fast level reads no Chinese: {fast:?}"
  );
  let message = refused(
    &AppleVisionTextOptions::new()
      .with_recognition_level(TextRecognitionLevel::Fast)
      .with_languages(["zh-Hans"]),
  );
  assert!(
    message.contains("\"zh-Hans\"") && message.contains("fast"),
    "the refusal names the tag and the level: {message}"
  );
}

/// A confidence or a text height that is not a fraction — outside
/// `0..=1`, or not a number — is refused by name at construction; both
/// ends of the interval are taken.
#[test]
fn a_fraction_outside_the_unit_interval_is_refused_by_name() {
  for value in [-0.01, 1.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
    let message = refused(&AppleVisionTextOptions::new().with_min_confidence(value));
    assert!(
      message.contains(&format!("min_confidence {value} ")),
      "{message}"
    );
    let message = refused(&AppleVisionTextOptions::new().with_min_text_height(value));
    assert!(
      message.contains(&format!("min_text_height {value} ")),
      "{message}"
    );
  }
  for value in [0.0, 0.5, 1.0] {
    TextRecognizer::new(&AppleVisionTextOptions::new().with_min_confidence(value))
      .unwrap_or_else(|error| panic!("min_confidence {value} is a fraction: {error}"));
    TextRecognizer::new(&AppleVisionTextOptions::new().with_min_text_height(value))
      .unwrap_or_else(|error| panic!("min_text_height {value} is a fraction: {error}"));
  }
}

/// `min_confidence` is a per-call gate, so a call handed one that is not
/// a fraction refuses — through both doors, before the input is looked
/// at.
#[test]
fn a_per_call_confidence_outside_the_unit_interval_is_refused_by_name() {
  let recognizer = TextRecognizer::new(&AppleVisionTextOptions::new()).expect("the defaults build");
  let pixels = page();
  for value in [-0.01, 1.01, f32::NAN] {
    let options = AppleVisionTextOptions::new().with_min_confidence(value);
    for error in [
      recognizer
        .recognize_pixels::<Reading>(&plane(&pixels), &options)
        .expect_err("the pixel door refuses the gate"),
      // Empty bytes: the gate is checked before the input, so this is a
      // refusal of the gate and not of the bytes.
      recognizer
        .recognize::<Reading>(&[], &options)
        .expect_err("the jpeg door refuses the gate"),
    ] {
      assert_eq!(error.kind(), AnalyzeErrorKind::InvalidOptions, "{error}");
      assert!(
        error.message().starts_with(&format!(
          "TextRecognizer::recognize: min_confidence {value} "
        )),
        "{error}"
      );
    }
  }
}

/// The issue, as a law.
///
/// Under the request this crate built before these options existed —
/// revision 3 and nothing else set — Vision reads English alone: the
/// English line comes back, and nothing in Chinese or Japanese does.
/// Under language detection, now the default, each CJK line comes back
/// in its own script; so does each line under a roster that names its
/// language.
///
/// The CJK lines are compared by character class and by the characters
/// that tell the two lines apart — 你好 and こんにちは, each with 世界 —
/// not as exact strings. That is what Vision promises; how it transcribes
/// a line differs from system to system, and the law has to hold on
/// every one at or above the floor. The English line's transcription is
/// stable, and is compared exactly.
///
/// Each language is named alone. A roster is processed in its order:
/// with detection off, macOS 26.6 (a CI runner) read `["zh-Hans", "ja-JP"]`
/// as the Chinese line and left the Japanese line unread, where macOS 27
/// read both.
#[test]
fn chinese_and_japanese_are_read_under_detection_or_a_roster_and_were_not_before() {
  const CHINESE_MARKS: [&str; 2] = ["你好", "世界"];
  const JAPANESE_MARKS: [&str; 2] = ["こんにちは", "世界"];

  let pixels = page();
  assert!(ink(&pixels) > 5_000, "Core Text set the lines on the page");

  let detected = read(&AppleVisionTextOptions::new(), &pixels);
  assert!(
    read_in_its_own_script(&detected, &CHINESE_MARKS)
      && read_in_its_own_script(&detected, &JAPANESE_MARKS),
    "with detection on, each CJK line is read in its own script: {}",
    joined(&detected)
  );
  assert!(
    joined(&detected).contains(ENGLISH),
    "and the English line is read: {}",
    joined(&detected)
  );

  for (language, marks) in [("zh-Hans", CHINESE_MARKS), ("ja-JP", JAPANESE_MARKS)] {
    let rostered = read(
      &AppleVisionTextOptions::new()
        .with_detect_language(false)
        .with_languages([language]),
      &pixels,
    );
    assert!(
      read_in_its_own_script(&rostered, &marks),
      "with {language} named, its line is read in its own script: {}",
      joined(&rostered)
    );
  }

  let before = joined(&read(
    &AppleVisionTextOptions::new().with_detect_language(false),
    &pixels,
  ));
  assert!(
    before.contains(ENGLISH),
    "the old request still reads English: {before}"
  );
  assert!(
    !before.chars().any(is_chinese_or_japanese),
    "and nothing in Chinese or Japanese: {before}"
  );
}

/// `min_confidence` drops every reading Vision scored below it, after
/// the request ran, and keeps every reading it scored at or above.
#[test]
fn min_confidence_drops_every_reading_scored_below_it() {
  let pixels = page();
  let options = AppleVisionTextOptions::new().with_max_candidates_per_observation(3);
  let recognizer = TextRecognizer::new(&options).expect("the options build");
  let ungated = recognizer
    .recognize_pixels::<Reading>(&plane(&pixels), &options)
    .expect("the pixel door must return Ok");
  let top = ungated
    .iter()
    .map(|reading| reading.confidence)
    .fold(f32::NEG_INFINITY, f32::max);
  assert!(
    ungated.iter().any(|reading| reading.confidence < top),
    "the page must be read at more than one confidence for the gate to have anything to drop: \
     {ungated:?}"
  );

  let gated = recognizer
    .recognize_pixels::<Reading>(&plane(&pixels), &options.clone().with_min_confidence(top))
    .expect("the pixel door must return Ok");
  assert!(
    gated.iter().all(|reading| reading.confidence >= top),
    "nothing below the gate survives it: {gated:?}"
  );
  assert_eq!(
    gated.len(),
    ungated
      .iter()
      .filter(|reading| reading.confidence >= top)
      .count(),
    "and nothing at or above it is dropped: {gated:?} of {ungated:?}"
  );
}
