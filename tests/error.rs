//! What the error type promises a caller: one type across the whole crate, a
//! message that names the culprit, and a source chain where one survived.

use burn_setfit::{Bundle, ClassifierConfig, Result, SetFitError};
use std::error::Error;
use std::io::ErrorKind;

#[test]
fn an_io_failure_reaches_the_caller_as_one_error_type() {
    // The point of the `From` impl: a function that both reads a file and parses
    // what it read needs one error type, not a `Box<dyn Error>`.
    fn load(path: &str) -> Result<Bundle> {
        let bytes = std::fs::read(path)?;
        Bundle::unpack(&bytes)
    }

    let err = load("no/such/bundle.setfit").expect_err("the file does not exist");
    assert!(matches!(&err, SetFitError::Io(e) if e.kind() == ErrorKind::NotFound));
}

#[test]
fn the_underlying_io_error_survives_rather_than_becoming_a_string() {
    // Flattening to a string would leave a caller unable to distinguish "not
    // found" from "permission denied" without parsing prose.
    let err: SetFitError = std::io::Error::new(ErrorKind::PermissionDenied, "nope").into();

    let SetFitError::Io(inner) = &err else {
        panic!("io errors convert to the Io variant, got {err:?}");
    };
    assert_eq!(inner.kind(), ErrorKind::PermissionDenied);

    let source = err.source().expect("an io error has a source to expose");
    assert_eq!(source.to_string(), "nope");
}

#[test]
fn the_variants_built_from_foreign_errors_report_no_source() {
    // Deliberate, and documented on `source`: those messages are folded into the
    // variant at the point of failure, where this crate can say which file or
    // which example was at fault.
    let err = Bundle::unpack(b"not a bundle").expect_err("not a bundle");
    assert!(err.source().is_none());
}

#[test]
fn display_names_the_kind_and_the_culprit() {
    // A message that says only "config error" sends the reader back to the code.
    let err = ClassifierConfig::new(["spam", "ham", "spam"])
        .validate()
        .expect_err("duplicate labels are ambiguous");

    let message = err.to_string();
    assert!(message.starts_with("config error: "), "got: {message}");
    assert!(message.contains("spam"), "got: {message}");

    let io = SetFitError::from(std::io::Error::new(ErrorKind::UnexpectedEof, "truncated"));
    assert_eq!(io.to_string(), "io error: truncated");
}

#[test]
fn the_error_type_is_usable_as_a_boxed_std_error() {
    // `main() -> Result<(), Box<dyn Error>>` is how most callers start out, so
    // the type has to survive being boxed and printed.
    fn fails() -> std::result::Result<(), Box<dyn Error>> {
        Bundle::unpack(b"")?;
        Ok(())
    }

    let err = fails().expect_err("an empty slice is not a bundle");
    assert!(err.to_string().contains("bundle error"));
}
