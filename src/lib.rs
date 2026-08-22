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
//!
//! # The shape of the thing
//!
//! Four types carry the whole workflow, and they hand off in one direction:
//!
//! ```text
//! Checkpoint ──> Trainer ──> Vec<u8> ──> Classifier ──> Prediction
//!  pretrained     both        a .setfit    loaded        scores, labels,
//!  MiniLM body    stages      bundle       model         and the passages
//!                                                        that caused them
//! ```
//!
//! [`Checkpoint`] is the pretrained body, however it arrived: downloaded from
//! HuggingFace natively, or handed over as bytes in a browser. [`Trainer`] runs
//! both SetFit stages one optimiser step at a time. Its output is a
//! [`Bundle`] — weights, tokenizer and configuration in a single file — and
//! [`Classifier`] turns those bytes back into a working model.
//!
//! [`ClassifierConfig`] is the one place behaviour is decided: labels, task
//! mode, windowing, reduction. Every entry point that consumes one validates
//! it, so a mistake is reported where it is used rather than becoming a model
//! that merely behaves oddly.
//!
//! # Training
//!
//! ```no_run
//! use burn::backend::{Autodiff, NdArray};
//! use burn_setfit::{Checkpoint, ClassifierConfig, Example, MiniLmVariant, TrainConfig, Trainer};
//!
//! # fn main() -> burn_setfit::Result<()> {
//! let checkpoint = Checkpoint::download(MiniLmVariant::L6, None)?;
//!
//! let mut trainer = Trainer::<Autodiff<NdArray<f32>>>::new(
//!     &checkpoint,
//!     ClassifierConfig::new(["billing", "outage"]),
//!     vec![
//!         Example::single("I was charged twice this month.", 0),
//!         Example::single("The dashboard is returning 503s.", 1),
//!         // ...eight or so per class
//!     ],
//!     TrainConfig::default(),
//!     Default::default(),
//! )?;
//!
//! trainer.fit_with(|p| eprintln!("{:?} {}/{}", p.stage, p.step, p.total_steps))?;
//!
//! // The trainer already knows the checkpoint, the labels and the task, so a
//! // finished bundle needs no further arguments.
//! let bundle: Vec<u8> = trainer.finish()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Classifying
//!
//! From those bytes alone — the same code path natively and in a browser:
//!
//! ```no_run
//! use burn::backend::NdArray;
//! use burn_setfit::{Classifier, Reducer};
//!
//! # fn main() -> burn_setfit::Result<()> {
//! # let bundle: Vec<u8> = Vec::new();
//! # let very_long_document = "";
//! let classifier = Classifier::<NdArray<f32>>::from_bundle(&bundle, Default::default())?;
//! let prediction = classifier.classify(very_long_document)?;
//!
//! for label in prediction.labels(classifier.manifest()) {
//!     println!("{label}");
//! }
//!
//! // Reduction, hierarchy and threshold are decode-time choices: switching one
//! // needs no retraining and no repacking.
//! let stricter = classifier.with_reducer(Reducer::MaxLogits);
//! # Ok(())
//! # }
//! ```
//!
//! # Documents longer than the encoder
//!
//! MiniLM sees [`TRAINED_SEQ_LEN`] tokens. Longer input is windowed at sentence
//! boundaries, each window is scored, and the per-window scores are folded into
//! one verdict by a [`Reducer`]. Two results are worth knowing before choosing
//! one, both measured rather than argued:
//!
//! - A single-label softmax head **cannot abstain**. Every chunk is forced into
//!   some class, so filler votes as confidently as signal.
//! - [`Reducer::NoisyOr`] **saturates**: at a realistic `p ≈ 0.6` for background
//!   text, eight chunks reach `1 - 0.4⁸ ≈ 0.999` and every label fires.
//!
//! For long documents, use multi-label with [`Reducer::MaxLogits`] and train a
//! background class. See the [`reduce`] module and `examples/long_document.rs`.
//!
//! # Feature flags
//!
//! | Feature | Purpose | wasm |
//! | ------- | ------- | ---- |
//! | `ndarray` | CPU backend | yes |
//! | `wgpu` | GPU backend (WebGPU in browsers) | yes |
//! | `train` | Both training stages | yes |
//! | `native` | HuggingFace download, filesystem | **no** |
//!
//! `native` is the only feature that cannot go to wasm, and it is confined to
//! [`Checkpoint::download`] for that reason. Everything else — including both
//! training stages — is byte-oriented and target-agnostic.
//!
//! Items that need a feature are labelled as such in the rendered docs, so an
//! item that appears to be missing is a feature that is off rather than an API
//! that does not exist.

// docs.rs builds with `--cfg docsrs` (see Cargo.toml), which turns on the
// feature labels. Nothing here changes for an ordinary build.
#![cfg_attr(docsrs, feature(doc_cfg))]

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
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub mod train;

// The types a caller names in ordinary use, re-exported so that using this
// crate is one `use` line rather than six. The modules stay public: anything
// deeper — accumulators, the chunker, the head — is still reachable at its own
// path, and this list is deliberately not everything.
pub use bundle::{Bundle, Manifest};
pub use checkpoint::Checkpoint;
pub use chunk::ChunkConfig;
pub use config::ClassifierConfig;
pub use error::{Result, SetFitError};
pub use head::TaskMode;
pub use infer::{Classifier, Evidence, Prediction};
pub use minilm::{EMBEDDING_DIM, MiniLmVariant, TRAINED_SEQ_LEN};
pub use reduce::Reducer;
pub use tokenize::Tokenizer;
#[cfg(feature = "train")]
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub use train::{Example, Progress, Stage, TrainConfig, Trainer};
