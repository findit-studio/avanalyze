//! The Vision request revisions a detector or analyzer pinned at
//! construction, published rather than logged.

use core::{fmt, marker::PhantomData};

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
/// # Why a producer marker, and not the request count alone
///
/// `P` names the entry point this roster came from and `N` is how many
/// requests that entry point owns. The named getters are inherent
/// methods on one exact `Revisions<P, N>`, so a getter belongs to its
/// producer rather than to an arity.
///
/// `N` alone would not do. [`BodyPoser`](crate::BodyPoser) and
/// [`PersonMasker`](crate::PersonMasker) each pin exactly two requests.
/// Keyed by count alone, one type would carry both rosters' getters, a
/// mask roster would answer `body_pose()` with a person-segmentation
/// revision, and no later release could withdraw that method without
/// breaking the published API. With the marker they are
/// `Revisions<BodyPoserRevisions, 2>` and
/// `Revisions<PersonMaskerRevisions, 2>` — distinct types that share
/// [`Display`](fmt::Display), [`IntoIterator`], and nothing else.
///
/// Every value is read back from the request object's own
/// `-[VNRequest revision]` inside the constructor that called
/// `setRevision` on it — never from a second constant kept beside that
/// setter — so a value this type reports can never drift from what the
/// request itself would answer.
#[non_exhaustive]
pub struct Revisions<P, const N: usize> {
  pub(crate) entries: [(&'static str, usize); N],
  pub(crate) producer: PhantomData<P>,
}

// Built only where a producer exists to call it: on Apple each
// constructor reads every revision back under its own exception
// barrier, and off Apple no reader ever returns a roster, so nothing
// calls this — and a `pub(crate)` function nothing calls is
// `dead_code`, which this crate's own CI denies as a warning.
#[cfg(target_vendor = "apple")]
impl<P, const N: usize> Revisions<P, N> {
  /// Builds a revision report from its named entries, in the order they
  /// should render and iterate.
  pub(crate) const fn new(entries: [(&'static str, usize); N]) -> Self {
    Self {
      entries,
      producer: PhantomData,
    }
  }
}

// The five impls below are written out rather than derived: a derive
// would put a `P: Trait` bound on each, and `P` is a marker that is
// never held, compared or printed. `PhantomData<P>` satisfies all of
// them for every `P`, so those bounds would only leak into the API.
impl<P, const N: usize> fmt::Debug for Revisions<P, N> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Revisions")
      .field("entries", &self.entries)
      .finish()
  }
}

impl<P, const N: usize> Clone for Revisions<P, N> {
  fn clone(&self) -> Self {
    *self
  }
}

impl<P, const N: usize> Copy for Revisions<P, N> {}

impl<P, const N: usize> PartialEq for Revisions<P, N> {
  fn eq(&self, other: &Self) -> bool {
    self.entries == other.entries
  }
}

impl<P, const N: usize> Eq for Revisions<P, N> {}

impl<P, const N: usize> fmt::Display for Revisions<P, N> {
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

impl<P, const N: usize> IntoIterator for Revisions<P, N> {
  type Item = (&'static str, usize);
  type IntoIter = core::array::IntoIter<(&'static str, usize), N>;

  fn into_iter(self) -> Self::IntoIter {
    self.entries.into_iter()
  }
}

/// Marks the roster
/// [`VisionAnalyzer::revisions`](crate::VisionAnalyzer::revisions)
/// returns, so its eight named getters belong to that analyzer alone.
///
/// A marker is only ever a type argument to [`Revisions`]; it is never
/// constructed, and its one private field is what keeps it that way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionAnalyzerRevisions(());

/// Marks the roster
/// [`FaceDetector::revisions`](crate::FaceDetector::revisions) returns,
/// so its three named getters belong to that detector alone.
///
/// A marker is only ever a type argument to [`Revisions`]; it is never
/// constructed, and its one private field is what keeps it that way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceDetectorRevisions(());

/// Marks the roster [`BodyPoser::revisions`](crate::BodyPoser::revisions)
/// returns, so its two named getters belong to that poser alone.
///
/// A marker is only ever a type argument to [`Revisions`]; it is never
/// constructed, and its one private field is what keeps it that way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BodyPoserRevisions(());

/// Marks the roster
/// [`PersonMasker::revisions`](crate::PersonMasker::revisions) returns,
/// so its two named getters belong to that masker alone.
///
/// It owns the same number of requests as [`BodyPoserRevisions`] and
/// shares none of its getters — which is the whole reason the producer
/// is a type parameter and not a request count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersonMaskerRevisions(());
