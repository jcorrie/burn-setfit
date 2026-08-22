//! SetFit for Burn: few-shot text classification that runs natively and in the browser.
//!
//! SetFit is two stages. First a sentence-transformer body is contrastively
//! fine-tuned on pairs drawn from a handful of labelled examples; then a light
//! classification head is fitted on the frozen body's embeddings. Both stages
//! run here, on any Burn backend, including `wasm32-unknown-unknown`.
//!
//! The inference path additionally handles documents of unbounded length by
//! windowing them into chunks the encoder can actually see and reducing the
//! per-chunk scores back to a document verdict. See [`chunk`] and [`reduce`].

pub mod bundle;
pub mod checkpoint;
pub mod chunk;
pub mod config;
pub mod error;
pub mod head;
pub mod infer;
pub mod minilm;
pub mod model;
pub mod reduce;
pub mod tokenize;
#[cfg(feature = "train")]
pub mod train;

pub use checkpoint::Checkpoint;
pub use config::ClassifierConfig;
pub use error::{Result, SetFitError};
pub use minilm::{EMBEDDING_DIM, MiniLmVariant, TRAINED_SEQ_LEN};
