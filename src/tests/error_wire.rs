//! The wire form of [`AnalyzeErrorKind`]: the serde derive numbers a unit
//! variant by its position, so an ordinal-based format reads a kind back as
//! whatever now sits at that position. Every kind that existed in 0.7 keeps
//! its 0.7 index, and a kind added since comes after all of them.

use core::fmt;

use serde::{
  Serialize,
  ser::{self, Impossible, Serializer},
};

use crate::AnalyzeErrorKind;

/// The kinds 0.7 shipped, in their 0.7 declaration order.
const SHIPPED_IN_0_7: [(&str, u32, AnalyzeErrorKind); 4] = [
  ("RequestFailed", 0, AnalyzeErrorKind::RequestFailed),
  ("Unsupported", 1, AnalyzeErrorKind::Unsupported),
  ("Environment", 2, AnalyzeErrorKind::Environment),
  ("InvalidOptions", 3, AnalyzeErrorKind::InvalidOptions),
];

/// Forces a new kind into this law: the match is exhaustive inside the
/// crate, so a variant added without a line here does not compile.
fn since_0_7(kind: AnalyzeErrorKind) -> bool {
  match kind {
    AnalyzeErrorKind::RequestFailed
    | AnalyzeErrorKind::Unsupported
    | AnalyzeErrorKind::Environment
    | AnalyzeErrorKind::InvalidOptions => false,
    AnalyzeErrorKind::BelowFloor => true,
  }
}

#[derive(Debug)]
struct NotAUnitVariant;

impl fmt::Display for NotAUnitVariant {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("only a unit variant is recorded")
  }
}

impl core::error::Error for NotAUnitVariant {}

impl ser::Error for NotAUnitVariant {
  fn custom<T: fmt::Display>(_: T) -> Self {
    Self
  }
}

/// A serializer that answers the one question the law asks: which name and
/// which ordinal the derive hands `serialize_unit_variant`.
struct Ordinal;

macro_rules! refuse {
  ($($method:ident($($arg:ty),*) -> $ret:ty;)*) => {
    $(
      fn $method(self, $(_: $arg),*) -> Result<$ret, NotAUnitVariant> {
        Err(NotAUnitVariant)
      }
    )*
  };
}

impl Serializer for Ordinal {
  type Ok = (&'static str, u32);
  type Error = NotAUnitVariant;
  type SerializeSeq = Impossible<Self::Ok, NotAUnitVariant>;
  type SerializeTuple = Impossible<Self::Ok, NotAUnitVariant>;
  type SerializeTupleStruct = Impossible<Self::Ok, NotAUnitVariant>;
  type SerializeTupleVariant = Impossible<Self::Ok, NotAUnitVariant>;
  type SerializeMap = Impossible<Self::Ok, NotAUnitVariant>;
  type SerializeStruct = Impossible<Self::Ok, NotAUnitVariant>;
  type SerializeStructVariant = Impossible<Self::Ok, NotAUnitVariant>;

  fn serialize_unit_variant(
    self,
    _name: &'static str,
    variant_index: u32,
    variant: &'static str,
  ) -> Result<Self::Ok, NotAUnitVariant> {
    Ok((variant, variant_index))
  }

  refuse! {
    serialize_bool(bool) -> Self::Ok;
    serialize_i8(i8) -> Self::Ok;
    serialize_i16(i16) -> Self::Ok;
    serialize_i32(i32) -> Self::Ok;
    serialize_i64(i64) -> Self::Ok;
    serialize_u8(u8) -> Self::Ok;
    serialize_u16(u16) -> Self::Ok;
    serialize_u32(u32) -> Self::Ok;
    serialize_u64(u64) -> Self::Ok;
    serialize_f32(f32) -> Self::Ok;
    serialize_f64(f64) -> Self::Ok;
    serialize_char(char) -> Self::Ok;
    serialize_str(&str) -> Self::Ok;
    serialize_bytes(&[u8]) -> Self::Ok;
    serialize_none() -> Self::Ok;
    serialize_unit() -> Self::Ok;
    serialize_unit_struct(&'static str) -> Self::Ok;
    serialize_seq(Option<usize>) -> Self::SerializeSeq;
    serialize_tuple(usize) -> Self::SerializeTuple;
    serialize_tuple_struct(&'static str, usize) -> Self::SerializeTupleStruct;
    serialize_tuple_variant(&'static str, u32, &'static str, usize) -> Self::SerializeTupleVariant;
    serialize_map(Option<usize>) -> Self::SerializeMap;
    serialize_struct(&'static str, usize) -> Self::SerializeStruct;
    serialize_struct_variant(&'static str, u32, &'static str, usize) -> Self::SerializeStructVariant;
  }

  fn serialize_some<T: ?Sized + Serialize>(self, _: &T) -> Result<Self::Ok, NotAUnitVariant> {
    Err(NotAUnitVariant)
  }

  fn serialize_newtype_struct<T: ?Sized + Serialize>(
    self,
    _: &'static str,
    _: &T,
  ) -> Result<Self::Ok, NotAUnitVariant> {
    Err(NotAUnitVariant)
  }

  fn serialize_newtype_variant<T: ?Sized + Serialize>(
    self,
    _: &'static str,
    _: u32,
    _: &'static str,
    _: &T,
  ) -> Result<Self::Ok, NotAUnitVariant> {
    Err(NotAUnitVariant)
  }
}

fn recorded(kind: AnalyzeErrorKind) -> (&'static str, u32) {
  kind.serialize(Ordinal).expect("a unit variant serializes")
}

/// Every kind 0.7 shipped serializes at its 0.7 index, under its 0.7 name.
#[test]
fn every_kind_from_0_7_keeps_its_index() {
  for (name, index, kind) in SHIPPED_IN_0_7 {
    assert!(!since_0_7(kind), "{name} is listed as shipped in 0.7");
    assert_eq!(
      recorded(kind),
      (name, index),
      "{name} must serialize at its 0.7 index: an ordinal-based format would read a 0.7 value back as another kind"
    );
  }
}

/// A kind added since 0.7 comes after every kind 0.7 shipped.
#[test]
fn a_kind_added_since_0_7_comes_after_all_of_them() {
  let last_shipped = SHIPPED_IN_0_7
    .iter()
    .map(|(_, index, _)| *index)
    .max()
    .unwrap();
  let (name, index) = recorded(AnalyzeErrorKind::BelowFloor);
  assert_eq!(name, "BelowFloor");
  assert!(
    index > last_shipped,
    "BelowFloor is at index {index}, not after the last 0.7 kind (index {last_shipped})"
  );
}

/// The JSON names are the variant names, unchanged.
#[test]
fn the_json_names_are_unchanged() {
  for (name, _, kind) in SHIPPED_IN_0_7 {
    assert_eq!(serde_json::to_string(&kind).unwrap(), format!("\"{name}\""));
    let back: AnalyzeErrorKind = serde_json::from_str(&format!("\"{name}\"")).unwrap();
    assert_eq!(back, kind);
  }
  assert_eq!(
    serde_json::to_string(&AnalyzeErrorKind::BelowFloor).unwrap(),
    "\"BelowFloor\""
  );
}
