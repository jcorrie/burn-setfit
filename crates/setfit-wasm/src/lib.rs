//! Browser bindings for `burn-setfit`.
//!
//! Two things shape this API.
//!
//! **Bytes, not paths.** Nothing here touches a filesystem or a network stack.
//! JavaScript fetches the bundle and hands over an `ArrayBuffer`; every code path
//! below that is the same one the native build uses.
//!
//! **Training yields.** [`Trainer::step`] performs exactly one optimiser step and
//! returns. A browser drives it from a loop that awaits a frame between calls, so
//! a minute of fine-tuning does not become a minute of frozen tab. The equivalent
//! `fit()` convenience is deliberately absent — it would only ever be the wrong
//! thing to call from the main thread.
//!
//! # Why the backend is `NdArray`
//!
//! Not a default anyone settled for: a GPU backend cannot be driven through the
//! synchronous methods used below. WebGPU has no blocking readback, so on
//! `wasm32` those return `SetFitError::Readback`, and lazy `WgpuDevice`
//! acquisition traps earlier still, while the first tensor is being built.
//!
//! `burn-setfit` grew an `_async` twin for every method that reads a tensor, so
//! the library side of that is solved. Reaching it from here is a separate
//! piece of work: the constructors and `step` would have to become
//! `Promise`-returning through `wasm-bindgen-futures`, and something would need
//! to call `burn::backend::wgpu::init_setup_async` before the first tensor
//! exists. Until that is written *and run against a real GPU*, this binding
//! stays on the backend that is actually tested. See
//! [#5](https://github.com/jcorrie/burn-setfit/issues/5).

use burn::backend::{Autodiff, NdArray};
use burn_setfit::{
    Checkpoint, Classifier as CoreClassifier, ClassifierConfig, Example, MiniLmVariant,
    TrainConfig, Trainer as CoreTrainer,
};
use wasm_bindgen::prelude::*;

type B = NdArray<f32>;
type AB = Autodiff<B>;

/// Install a panic hook that reports Rust panics to the browser console.
///
/// Without it a panic surfaces as `unreachable executed`, which says nothing.
#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

fn js_err(e: impl core::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// A document-level prediction, as returned to JavaScript.
#[derive(serde::Serialize)]
struct JsPrediction {
    /// Names of the predicted labels.
    labels: Vec<String>,
    /// Probability per label, in model label order.
    scores: Vec<f32>,
    /// Label names in model order, so `scores` can be read without a second call.
    label_names: Vec<String>,
    /// How many chunks the document produced.
    chunks_seen: usize,
    /// Passages supporting the predicted labels.
    evidence: Vec<JsEvidence>,
}

#[derive(serde::Serialize)]
struct JsEvidence {
    label: String,
    score: f32,
    /// Byte offsets into the classified text.
    start: usize,
    end: usize,
}

/// A loaded model.
#[wasm_bindgen]
pub struct Classifier {
    inner: CoreClassifier<B>,
}

#[wasm_bindgen]
impl Classifier {
    /// Load from `.setfit` bundle bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(bundle: &[u8]) -> Result<Classifier, JsValue> {
        Ok(Classifier {
            inner: CoreClassifier::from_bundle(bundle, Default::default()).map_err(js_err)?,
        })
    }

    /// Classify a document of any length. Returns JSON.
    ///
    /// Long inputs are windowed and reduced internally, so there is no length
    /// limit here beyond what the caller can hold as a string.
    pub fn classify(&self, text: &str) -> Result<String, JsValue> {
        let p = self.inner.classify(text).map_err(js_err)?;
        let manifest = self.inner.manifest();

        let out = JsPrediction {
            labels: p.labels(manifest).into_iter().map(str::to_string).collect(),
            scores: p.scores.clone(),
            label_names: manifest.labels().to_vec(),
            chunks_seen: p.chunks_seen,
            evidence: p
                .evidence
                .iter()
                .map(|e| JsEvidence {
                    label: manifest
                        .labels()
                        .get(e.label)
                        .cloned()
                        .unwrap_or_else(|| e.label.to_string()),
                    score: e.score,
                    start: e.byte_range.start,
                    end: e.byte_range.end,
                })
                .collect(),
        };
        serde_json::to_string(&out).map_err(js_err)
    }

    /// The model's label names, in order.
    pub fn labels(&self) -> Vec<String> {
        self.inner.labels().to_vec()
    }
}

/// One labelled example, as supplied from JavaScript.
#[derive(serde::Deserialize)]
struct JsExample {
    text: String,
    /// Label indices. One entry for single-label, any number for multi-label.
    labels: Vec<usize>,
}

/// Everything needed to start training, supplied as one JSON object.
#[derive(serde::Deserialize)]
struct JsTrainRequest {
    labels: Vec<String>,
    examples: Vec<JsExample>,
    #[serde(default)]
    multi_label: bool,
    /// Multi-label decision threshold. Ignored when `multi_label` is false.
    #[serde(default)]
    threshold: Option<f32>,
    #[serde(default)]
    num_iterations: Option<usize>,
    #[serde(default)]
    head_epochs: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
}

/// Progress from a single training step.
#[derive(serde::Serialize)]
struct JsProgress {
    stage: String,
    step: usize,
    total_steps: usize,
    loss: f32,
    fraction: f32,
    done: bool,
}

/// Few-shot training, one step per call.
#[wasm_bindgen]
pub struct Trainer {
    inner: Option<CoreTrainer<AB>>,
}

#[wasm_bindgen]
impl Trainer {
    /// Start training from a pretrained MiniLM checkpoint.
    ///
    /// `config_json`, `weights` and `tokenizer_json` are the three files from the
    /// HuggingFace repo, fetched by the caller.
    #[wasm_bindgen(constructor)]
    pub fn new(
        config_json: &str,
        weights: Vec<u8>,
        tokenizer_json: Vec<u8>,
        request_json: &str,
    ) -> Result<Trainer, JsValue> {
        let request: JsTrainRequest = serde_json::from_str(request_json).map_err(js_err)?;

        let checkpoint =
            Checkpoint::from_files(MiniLmVariant::L6, config_json, weights, tokenizer_json)
                .map_err(js_err)?;

        let mut classifier = ClassifierConfig::new(request.labels);
        if request.multi_label {
            classifier = match request.threshold {
                Some(t) => classifier.multi_label_at(t),
                None => classifier.multi_label(),
            };
        }

        let mut config = TrainConfig::default();
        if let Some(v) = request.num_iterations {
            config.num_iterations = v;
        }
        if let Some(v) = request.head_epochs {
            config.head_epochs = v;
        }
        if let Some(v) = request.seed {
            config.seed = v;
        }

        let examples: Vec<Example> = request
            .examples
            .into_iter()
            .map(|e| Example {
                text: e.text,
                labels: e.labels,
            })
            .collect();

        let inner = CoreTrainer::new(
            &checkpoint,
            classifier,
            examples,
            config,
            Default::default(),
        )
        .map_err(js_err)?;

        Ok(Trainer { inner: Some(inner) })
    }

    /// Steps this training run will take in total, known before it starts.
    #[wasm_bindgen(js_name = totalSteps)]
    pub fn total_steps(&self) -> usize {
        self.inner.as_ref().map(|t| t.total_steps()).unwrap_or(0)
    }

    /// Perform one optimiser step. Returns progress JSON with `done: true`
    /// once there is nothing left to do.
    ///
    /// Call this from a loop that yields between iterations:
    ///
    /// ```js
    /// let p;
    /// do {
    ///   p = JSON.parse(trainer.step());
    ///   render(p.fraction);
    ///   await new Promise(requestAnimationFrame);
    /// } while (!p.done);
    /// ```
    pub fn step(&mut self) -> Result<String, JsValue> {
        let Some(trainer) = self.inner.as_mut() else {
            return Err(JsValue::from_str("trainer already finished"));
        };

        let progress = trainer.step().map_err(js_err)?;
        let out = match progress {
            Some(p) => JsProgress {
                stage: format!("{:?}", p.stage),
                step: p.step,
                total_steps: p.total_steps,
                loss: p.loss,
                fraction: p.fraction(),
                done: false,
            },
            None => JsProgress {
                stage: "Done".into(),
                step: trainer.total_steps(),
                total_steps: trainer.total_steps(),
                loss: 0.0,
                fraction: 1.0,
                done: true,
            },
        };
        serde_json::to_string(&out).map_err(js_err)
    }

    /// Finish training and return a `.setfit` bundle.
    ///
    /// Consumes the trainer; calling [`Self::step`] afterwards fails. Fails if
    /// training has not run to completion.
    pub fn finish(&mut self) -> Result<Vec<u8>, JsValue> {
        let Some(trainer) = self.inner.take() else {
            return Err(JsValue::from_str("trainer already finished"));
        };
        trainer.finish().map_err(js_err)
    }
}
