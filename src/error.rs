use core::fmt;
use std::borrow::Cow;

/// What kind of failure stopped an analysis.
///
/// The engine's failure surface is deliberately tiny: everything that
/// *can* degrade does so silently (a refused detection is simply not
/// emitted), so an `Err` means no analysis happened at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AnalyzeErrorKind {
  /// The frame was refused before or during the Vision pass: the input
  /// exceeded the engine's byte ceiling, or Apple's batched
  /// `performRequests` reported an error.
  RequestFailed,
  /// Apple's Vision framework is not available on this platform.
  Unsupported,
  /// Apple's native stack raised instead of returning: an Objective-C
  /// or C++ exception escaped Vision, CoreML, or the Neural Engine
  /// layers beneath them, and this crate's barrier caught it.
  ///
  /// This is a refusal by the HOST, not by the frame. The reproducible
  /// case is a machine whose Neural Engine is denied — a sandbox that
  /// refuses the ANE device, a policy that withholds it — where
  /// building the 3-D body-pose request raises `EspressoPlanFailure`
  /// while loading its model, before any image exists to blame. The
  /// same call on the same host refuses again; a sibling entry point
  /// whose model does load is unaffected, which is why this is
  /// per-call and not a platform-wide [`Unsupported`](Self::Unsupported).
  ///
  /// It exists because the alternative is not an error at all. A raise
  /// that crosses into Rust unguarded takes the whole process down —
  /// `fatal runtime error: Rust cannot catch foreign exceptions` — and
  /// a daemon that indexes media must refuse one frame, not die.
  ///
  /// Reaching this variant requires `panic = "unwind"`, the default. A
  /// raise has to cross one Rust frame to reach the barrier that names
  /// it, and `panic = "abort"` puts an abort-on-unwind shim on exactly
  /// that boundary — the same reason
  /// [`objc2::exception::catch`](https://docs.rs/objc2/latest/objc2/exception/fn.catch.html)
  /// documents itself as unable to catch there. Under that setting a
  /// consumer gets what they had before this variant existed.
  Environment,
}

impl fmt::Display for AnalyzeErrorKind {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::RequestFailed => f.write_str("apple-vision request failed"),
      Self::Unsupported => f.write_str("apple-vision unavailable"),
      Self::Environment => f.write_str("apple-vision raised a native exception"),
    }
  }
}

/// A refused analysis: a [kind](AnalyzeErrorKind) plus a message.
///
/// Unlike the aggregate-shaped error this crate used to return, this
/// one implements [`Display`](fmt::Display) and [`core::error::Error`],
/// so callers can propagate it with `?` instead of reassembling a
/// string from its parts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AnalyzeError {
  kind: AnalyzeErrorKind,
  message: Cow<'static, str>,
}

impl AnalyzeError {
  /// Builds an error. A `&'static str` message costs no allocation.
  #[inline]
  pub fn new(kind: AnalyzeErrorKind, message: impl Into<Cow<'static, str>>) -> Self {
    Self {
      kind,
      message: message.into(),
    }
  }

  /// The failure kind.
  #[inline]
  pub const fn kind(&self) -> AnalyzeErrorKind {
    self.kind
  }

  /// The human-readable detail. Never empty.
  #[inline]
  pub fn message(&self) -> &str {
    &self.message
  }

  /// Returns this error with a different kind.
  #[inline]
  pub fn with_kind(mut self, kind: AnalyzeErrorKind) -> Self {
    self.set_kind(kind);
    self
  }

  /// Replaces the kind in place.
  #[inline]
  pub const fn set_kind(&mut self, kind: AnalyzeErrorKind) -> &mut Self {
    self.kind = kind;
    self
  }

  /// Returns this error with a different message.
  #[inline]
  pub fn with_message(mut self, message: impl Into<Cow<'static, str>>) -> Self {
    self.set_message(message);
    self
  }

  /// Replaces the message in place.
  #[inline]
  pub fn set_message(&mut self, message: impl Into<Cow<'static, str>>) -> &mut Self {
    self.message = message.into();
    self
  }
}

/// The single refusal every entry point returns off Apple: Vision.framework
/// does not exist here, so nothing was analysed.
///
/// One function rather than one literal per stub, so the kind and the
/// message cannot drift between entry points.
#[cfg(not(target_vendor = "apple"))]
pub(crate) fn unsupported<T>() -> Result<T, AnalyzeError> {
  Err(AnalyzeError::new(
    AnalyzeErrorKind::Unsupported,
    "Apple Vision.framework is only available on macOS",
  ))
}

impl fmt::Display for AnalyzeError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}: {}", self.kind, self.message)
  }
}

impl core::error::Error for AnalyzeError {}
