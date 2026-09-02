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
//! | `wgpu` | GPU backend | native, or opt in below |
//! | `wgpu-wasm-unverified` | `wgpu` on `wasm32`, never executed | yes |
//! | `train` | Both training stages | yes |
//! | `native` | HuggingFace download, filesystem | **no** |
//!
//! `native` is filesystem and network, and is confined to
//! [`Checkpoint::download`] for that reason. Everything else — including both
//! training stages — is byte-oriented and target-agnostic.
//!
//! # A GPU backend in a browser
//!
//! WebGPU has no synchronous readback. A browser cannot block a thread on a GPU
//! buffer map, so Burn's `Tensor::into_data()` polls the map future once, finds
//! it pending and panics. Lazy `WgpuDevice` acquisition goes through the same
//! `block_on`, so the trap arrives even earlier — while a tensor is being
//! built, before anything is read.
//!
//! Every method here that reads a tensor therefore has an `_async` twin:
//! [`Classifier::classify_async`], [`Classifier::classify_stream_async`],
//! [`Classifier::embed_async`], [`Trainer::step_async`],
//! [`Trainer::fit_with_async`]. Those are the implementations. The synchronous
//! methods are wrappers that drive them to completion, which natively always
//! works and on `wasm32` succeeds only for a backend whose reads finish
//! immediately — `ndarray` does, a GPU backend does not. Where the wrapper
//! cannot wait it returns [`SetFitError::Readback`] naming the `_async` method,
//! rather than panicking inside Burn.
//!
//! Two things remain the caller's job on that target. The device must be
//! brought up before any tensor exists, with
//! `burn::backend::wgpu::init_setup_async`, because the lazy path cannot work.
//! And `wasm32` with `wgpu` must be asked for explicitly, through
//! `wgpu-wasm-unverified`: the combination compiles and the API it needs is
//! here, but it has never been run on a GPU in a browser, and the feature is
//! named so that enabling it cannot be mistaken for evidence that it has. See
//! [#5](https://github.com/jcorrie/burn-setfit/issues/5).
//!
//! Items that need a feature are labelled as such in the rendered docs, so an
//! item that appears to be missing is a feature that is off rather than an API
//! that does not exist.

// docs.rs builds with `--cfg docsrs` (see Cargo.toml), which turns on the
// feature labels. Nothing here changes for an ordinary build.
#![cfg_attr(docsrs, feature(doc_cfg))]

// A GPU backend on wasm32 is reachable only through the `_async` methods, and
// only after the caller has brought the device up by hand. Nothing fails to
// compile if neither happens — the combination builds cleanly and then traps in
// the browser with no usable message — so it is gated on an opt-in whose name
// says what is being opted into. See the feature table above.
#[cfg(all(
    feature = "wgpu",
    target_arch = "wasm32",
    not(feature = "wgpu-wasm-unverified")
))]
compile_error!(
    "`wgpu` on wasm32 needs the `wgpu-wasm-unverified` feature. WebGPU readback \
     and device init are both async, so on this target the backend works only \
     through the `_async` methods, and only if you call \
     `burn::backend::wgpu::init_setup_async` before building any tensor; the \
     blocking methods report `SetFitError::Readback` instead. The combination \
     has never been executed on a GPU. Build wasm with `ndarray` for the tested \
     path. See https://github.com/jcorrie/burn-setfit/issues/5"
);

pub mod bundle;
pub mod checkpoint;
pub mod chunk;
pub mod config;
pub mod error;
pub mod head;
pub mod infer;
pub mod minilm;
pub mod model;
pub mod quantize;
mod readback;
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
pub use infer::{ChunkPrediction, Classifier, Evidence, Prediction};
pub use minilm::{EMBEDDING_DIM, MiniLmVariant, TRAINED_SEQ_LEN};
pub use quantize::Quantization;
pub use reduce::Reducer;
pub use tokenize::Tokenizer;
#[cfg(feature = "train")]
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub use train::{Example, Progress, Stage, TrainConfig, Trainer};
