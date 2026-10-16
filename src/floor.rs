//! The crate's system floor: the oldest systems every entry point is
//! built on, and the one check each constructor makes before anything
//! else.
//!
//! The floor is written once, in [`floor!`]. The public
//! [`SYSTEM_FLOOR`] text, the `available!` check and the version the
//! test seam compares against each expand from that list, so the three
//! cannot disagree.

#[cfg(target_vendor = "apple")]
use core::sync::atomic::{AtomicU8, Ordering};

#[cfg(target_vendor = "apple")]
use crate::{AnalyzeError, AnalyzeErrorKind};

/// Hands `$then!` the crate's floor: each Apple system's oldest version,
/// as `system major minor`.
///
/// This is the only place the floor is written. Raising it means editing
/// this list and nothing else.
macro_rules! floor {
  ($then:ident) => {
    $then! { macos 15 0, ios 18 0, tvos 18 0, visionos 2 0 }
  };
}

/// The floor as the text a refusal names.
macro_rules! floor_text {
  (
    macos $macos:tt $macos_minor:tt,
    ios $ios:tt $ios_minor:tt,
    tvos $tvos:tt $tvos_minor:tt,
    visionos $visionos:tt $visionos_minor:tt
  ) => {
    concat!(
      "macOS ",
      $macos,
      ".",
      $macos_minor,
      ", iOS ",
      $ios,
      ".",
      $ios_minor,
      ", tvOS ",
      $tvos,
      ".",
      $tvos_minor,
      " or visionOS ",
      $visionos,
      ".",
      $visionos_minor
    )
  };
}

/// The floor as the literals `objc2::available!` takes.
#[cfg(target_vendor = "apple")]
macro_rules! floor_available {
  (
    macos $macos:tt $macos_minor:tt,
    ios $ios:tt $ios_minor:tt,
    tvos $tvos:tt $tvos_minor:tt,
    visionos $visionos:tt $visionos_minor:tt
  ) => {
    objc2::available!(
      macos = $macos.$macos_minor,
      ios = $ios.$ios_minor,
      tvos = $tvos.$tvos_minor,
      visionos = $visionos.$visionos_minor
    )
  };
}

/// The floor as the `(major, minor)` this target's system version is
/// compared against. A system outside the four is never at it.
#[cfg(all(test, target_vendor = "apple"))]
macro_rules! floor_version {
  (
    macos $macos:tt $macos_minor:tt,
    ios $ios:tt $ios_minor:tt,
    tvos $tvos:tt $tvos_minor:tt,
    visionos $visionos:tt $visionos_minor:tt
  ) => {
    if cfg!(target_os = "macos") {
      ($macos, $macos_minor)
    } else if cfg!(target_os = "ios") {
      ($ios, $ios_minor)
    } else if cfg!(target_os = "tvos") {
      ($tvos, $tvos_minor)
    } else if cfg!(target_os = "visionos") {
      ($visionos, $visionos_minor)
    } else {
      (u16::MAX, u16::MAX)
    }
  };
}

/// The oldest systems this crate is built on: macOS 15, iOS 18, tvOS 18
/// and visionOS 2.
///
/// Below them, every entry point's constructor refuses with
/// [`AnalyzeErrorKind::BelowFloor`](crate::AnalyzeErrorKind::BelowFloor).
/// It refuses before it sends a message or resolves a Vision class, and
/// its message names this floor.
///
/// The floor is where the newest Vision class an entry point builds
/// arrives. [`VisionAnalyzer`](crate::VisionAnalyzer) always builds
/// `VNCalculateImageAestheticsScoresRequest`, which
/// `VNCalculateImageAestheticsScoresRequest.h` declares
/// `API_AVAILABLE(macos(15.0), ios(18.0), tvos(18.0), visionos(2.0))`. Every
/// other class the nine constructors build is older. One floor for the
/// crate, rather than one per entry point, means a consumer has one
/// question to ask and the crate has one answer to keep true.
///
/// Below a class's own floor, resolving it panics. One check, made first,
/// therefore makes every message a constructor sends safe to send.
pub const SYSTEM_FLOOR: &str = floor!(floor_text);

/// The floor's comparable version on this target, for the test seam.
#[cfg(all(test, target_vendor = "apple"))]
pub(crate) const FLOOR_VERSION: (u16, u16) = floor!(floor_version);

/// Whether this system is at or above [`SYSTEM_FLOOR`].
///
/// objc2 is asked once (it reads the version as Apple's own availability
/// check does, Mac Catalyst and the simulators included), and the answer
/// is kept for the life of the process. Under the test seam the
/// simulated version is compared against the floor instead, and nothing
/// is kept.
#[cfg(target_vendor = "apple")]
fn system_at_floor() -> bool {
  #[cfg(test)]
  if let Some(version) = seam::simulated() {
    return version >= FLOOR_VERSION;
  }
  /// Not asked yet.
  const UNKNOWN: u8 = 0;
  /// At or above the floor.
  const AT: u8 = 1;
  /// Below it.
  const BELOW: u8 = 2;
  // Relaxed is enough: two threads that race to ask both get the same
  // answer and store the same value.
  static ANSWER: AtomicU8 = AtomicU8::new(UNKNOWN);
  match ANSWER.load(Ordering::Relaxed) {
    AT => true,
    BELOW => false,
    _ => {
      let at = floor!(floor_available);
      ANSWER.store(if at { AT } else { BELOW }, Ordering::Relaxed);
      at
    }
  }
}

/// Refuses `site` below [`SYSTEM_FLOOR`] with
/// [`AnalyzeErrorKind::BelowFloor`], naming the entry point and the
/// floor.
///
/// It is the first statement of every entry point's constructor, before
/// options are checked and before any class is resolved.
#[cfg(target_vendor = "apple")]
pub(crate) fn require_floor(site: &'static str) -> Result<(), AnalyzeError> {
  if system_at_floor() {
    Ok(())
  } else {
    Err(AnalyzeError::new(
      AnalyzeErrorKind::BelowFloor,
      format!("{site}: this system is older than {SYSTEM_FLOOR}, the oldest avanalyze is built on"),
    ))
  }
}

/// The test seam: a simulated system version for the calling thread.
///
/// It is thread-local, so tests that run side by side never see each
/// other's simulation. It is scoped, so a simulation ends when its guard
/// drops, a panicking test's included.
#[cfg(all(test, target_vendor = "apple"))]
pub(crate) mod seam {
  use std::cell::Cell;

  thread_local! {
    static SIMULATED: Cell<Option<(u16, u16)>> = const { Cell::new(None) };
  }

  /// The calling thread's simulated `(major, minor)`, if any.
  pub(crate) fn simulated() -> Option<(u16, u16)> {
    SIMULATED.with(Cell::get)
  }

  /// A simulation in force. Dropping it restores what was simulated
  /// before.
  #[must_use = "the simulation ends when the guard drops"]
  pub(crate) struct Simulated {
    previous: Option<(u16, u16)>,
  }

  /// Simulates `major.minor` as the calling thread's system version until
  /// the guard drops.
  pub(crate) fn simulate(major: u16, minor: u16) -> Simulated {
    Simulated {
      previous: SIMULATED.with(|cell| cell.replace(Some((major, minor)))),
    }
  }

  impl Drop for Simulated {
    fn drop(&mut self) {
      SIMULATED.with(|cell| cell.set(self.previous));
    }
  }
}
