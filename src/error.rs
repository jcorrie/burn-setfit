//! Error type for the crate.

use std::fmt;

/// Anything that can go wrong loading, training, or running a SetFit model.
///
/// Every variant carries a message that names what was wrong, rather than only
/// what kind of thing was wrong:
///
/// ```
/// use burn_setfit::{Bundle, SetFitError};
///
/// let err = Bundle::unpack(b"not a bundle").unwrap_err();
/// assert!(matches!(err, SetFitError::Bundle(_)));
/// assert_eq!(err.to_string(), "bundle error: not a .setfit bundle");
/// ```
#[derive(Debug)]
pub enum SetFitError {
    /// Weight (de)serialisation failed.
    Store(String),
    /// A config or manifest could not be parsed.
    Config(String),
    /// The tokenizer could not be built, or failed on an input.
    Tokenizer(String),
    /// A model bundle was malformed or had a version we cannot read.
    Bundle(String),
    /// Downloading a checkpoint failed (native only).
    Download(String),
    /// Training was given data it cannot learn from.
    Training(String),
}

impl fmt::Display for SetFitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SetFitError::Store(m) => write!(f, "store error: {m}"),
            SetFitError::Config(m) => write!(f, "config error: {m}"),
            SetFitError::Tokenizer(m) => write!(f, "tokenizer error: {m}"),
            SetFitError::Bundle(m) => write!(f, "bundle error: {m}"),
            SetFitError::Download(m) => write!(f, "download error: {m}"),
            SetFitError::Training(m) => write!(f, "training error: {m}"),
        }
    }
}

impl std::error::Error for SetFitError {}

/// Convenience alias.
pub type Result<T> = core::result::Result<T, SetFitError>;
