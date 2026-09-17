//! The Vision request revisions a detector or analyzer pinned at
//! construction, published rather than logged.

use core::fmt;

/// The pinned Vision request revisions one entry point holds, one entry
/// per request it constructed.
///
/// [`Display`](fmt::Display) renders every entry as
/// `<request>@<revision>`, comma separated, in construction order — the
/// same order [`IntoIterator`] walks. A producer
/// (e.g. [`VisionAnalyzer::revisions`](crate::VisionAnalyzer::revisions))
/// also exposes its own requests by name, so a caller who already knows
/// which entry point it holds never has to parse the rendered form.
///
/// `N` is the number of requests the producing entry point owns. It is
/// what lets several producers share this one type rather than each
/// growing its own — [`VisionAnalyzer`](crate::VisionAnalyzer) returns
/// `Revisions<8>`, [`FaceDetector`](crate::FaceDetector) returns
/// `Revisions<3>`, [`BodyPoser`](crate::BodyPoser) returns
/// `Revisions<2>` — and it is why the named getters live beside each
/// producer rather than on this type directly: they are inherent methods
/// on that producer's own `N`. Two producers that ever pinned the same
/// number of requests would need to keep sharing their named getters
/// here, or one of them would need a rename; nothing in this crate
/// collides today.
///
/// Every value comes from reading the request object's own
/// `-[VNRequest revision]` back after `setRevision` — never from a
/// second constant kept beside the setter — so a value this type reports
/// can never drift from what the request itself would answer.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Revisions<const N: usize> {
  pub(crate) entries: [(&'static str, usize); N],
}

// Built only where a producer exists to call it: every `revisions()` /
// `revision()` reader lives behind `#[cfg(target_vendor = "apple")]`, so
// off Apple nothing in the crate ever calls this — and a `pub(crate)`
// function nothing calls is `dead_code`, which this crate's own CI
// denies as a warning.
#[cfg(target_vendor = "apple")]
impl<const N: usize> Revisions<N> {
  /// Builds a revision report from its named entries, in the order they
  /// should render and iterate.
  pub(crate) const fn new(entries: [(&'static str, usize); N]) -> Self {
    Self { entries }
  }
}

impl<const N: usize> fmt::Display for Revisions<N> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for (index, (name, revision)) in self.entries.iter().enumerate() {
      if index > 0 {
        write!(f, ",")?;
      }
      write!(f, "{name}@{revision}")?;
    }
    Ok(())
  }
}

impl<const N: usize> IntoIterator for Revisions<N> {
  type Item = (&'static str, usize);
  type IntoIter = core::array::IntoIter<(&'static str, usize), N>;

  fn into_iter(self) -> Self::IntoIter {
    self.entries.into_iter()
  }
}
