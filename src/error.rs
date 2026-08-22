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
///
/// # Reading a model from disk
///
/// Loading a bundle is two failures wearing one coat: the file might not be
/// readable, and its contents might not be a bundle. [`SetFitError::Io`] carries
/// the first, so `?` composes across both and a caller needs one error type
/// rather than a `Box<dyn Error>`:
///
/// ```no_run
/// use burn::backend::NdArray;
/// use burn_setfit::{Classifier, Result};
///
/// fn load(path: &str) -> Result<Classifier<NdArray<f32>>> {
///     let bytes = std::fs::read(path)?;             // io::Error
///     Classifier::from_bundle(&bytes, Default::default()) // SetFitError
/// }
/// ```
///
/// The underlying [`std::io::Error`] is kept rather than flattened into a
/// string, so a caller can still ask what kind of failure it was:
///
/// ```
/// use std::error::Error;
/// use std::io::ErrorKind;
/// use burn_setfit::SetFitError;
///
/// let err: SetFitError = std::io::Error::new(ErrorKind::NotFound, "no such file").into();
///
/// // Matched by kind, for a caller that wants to treat "missing" differently...
/// assert!(matches!(&err, SetFitError::Io(e) if e.kind() == ErrorKind::NotFound));
/// // ...or followed as a source chain, for one that just wants to print it.
/// assert!(err.source().is_some());
/// ```
///
/// # Adding variants
///
/// The enum is `#[non_exhaustive]`: matching on it needs a `_` arm, and gaining
/// a variant is therefore not a breaking change. `Io` was added after the fact
/// and would otherwise have broken every downstream `match`.
#[derive(Debug)]
#[non_exhaustive]
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
    /// Reading or writing failed underneath us.
    ///
    /// Produced by `?` on [`std::io`] operations rather than by this crate,
    /// which reports its own file failures through [`SetFitError::Download`]
    /// with the path attached. The source error is kept intact.
    Io(std::io::Error),
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
            SetFitError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for SetFitError {
    /// The underlying error, where one survived.
    ///
    /// Only [`SetFitError::Io`] has one to give. The rest are built from foreign
    /// error types — `burn_store`, `tokenizers`, `hf_hub` — whose messages are
    /// folded into the variant's own at the point of failure, because the
    /// context this crate can add there (which file, which tensor, which
    /// example) is worth more than the chain.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SetFitError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SetFitError {
    fn from(e: std::io::Error) -> Self {
        SetFitError::Io(e)
    }
}

/// Convenience alias.
pub type Result<T> = core::result::Result<T, SetFitError>;
