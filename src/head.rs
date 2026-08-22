//! Classification head.
//!
//! Upstream SetFit defaults to a scikit-learn `LogisticRegression` fitted on the
//! frozen body's embeddings, and reaches for one-vs-rest when the task is
//! multi-label. This is SetFit's other supported head — `use_differentiable_head`
//! — a single `Linear` layer trained with Adam. Same model class, same decision
//! boundary, but it trains inside Burn, runs on every backend, and needs no LBFGS
//! implementation. Single-label and multi-label differ only in loss and decoding,
//! so both task modes share one set of weights.

use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::{BinaryCrossEntropyLossConfig, CrossEntropyLossConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Whether a document gets exactly one label or any number of them.
///
/// The decision threshold lives inside the multi-label variant because that is the
/// only place it means anything: an argmax has nothing to threshold. Carrying it as
/// a sibling field invites a single-label config with a carefully tuned threshold
/// that silently does nothing.
///
/// ```
/// use burn_setfit::TaskMode;
/// use burn_setfit::infer::{decode, to_probabilities};
///
/// let logits = [2.0, 1.0, -3.0];
///
/// // Single-label: a softmax over the label set, decoded by argmax. Exactly
/// // one label, always -- even when nothing fits.
/// assert_eq!(decode(&to_probabilities(&logits, TaskMode::SingleLabel), TaskMode::SingleLabel), vec![0]);
///
/// // Multi-label: independent sigmoids, decoded by threshold. Both of these
/// // logits are positive, so both clear 0.5.
/// let multi = TaskMode::multi_label();
/// assert_eq!(decode(&to_probabilities(&logits, multi), multi), vec![0, 1]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum TaskMode {
    /// Exactly one label per document. Softmax over classes, argmax to decode.
    #[default]
    SingleLabel,
    /// Zero or more labels per document. Independent sigmoids, threshold to decode.
    MultiLabel {
        /// Probability at or above which a class counts as present.
        threshold: f32,
    },
}

impl TaskMode {
    /// Multi-label at the conventional 0.5 threshold.
    pub fn multi_label() -> Self {
        TaskMode::MultiLabel { threshold: 0.5 }
    }

    /// Whether this is the multi-label mode.
    pub fn is_multi_label(&self) -> bool {
        matches!(self, TaskMode::MultiLabel { .. })
    }

    /// Reject a threshold that could never fire, or that always would.
    ///
    /// A threshold of 0 labels every document with every class; one above 1 labels
    /// nothing, whatever the model learned. Both are configuration mistakes that
    /// look like model failures.
    ///
    /// ```
    /// use burn_setfit::TaskMode;
    ///
    /// assert!(TaskMode::MultiLabel { threshold: 0.5 }.validate().is_ok());
    /// assert!(TaskMode::MultiLabel { threshold: 0.0 }.validate().is_err());
    /// assert!(TaskMode::MultiLabel { threshold: 1.0 }.validate().is_err());
    /// assert!(TaskMode::MultiLabel { threshold: f32::NAN }.validate().is_err());
    /// ```
    pub fn validate(&self) -> crate::Result<()> {
        match self {
            TaskMode::SingleLabel => Ok(()),
            TaskMode::MultiLabel { threshold } => {
                if !threshold.is_finite() || *threshold <= 0.0 || *threshold >= 1.0 {
                    Err(crate::SetFitError::Config(format!(
                        "multi-label threshold must lie strictly between 0 and 1, got {threshold}"
                    )))
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Head configuration.
#[derive(Config, Debug)]
pub struct SetFitHeadConfig {
    /// Embedding width of the body — 384 for both MiniLM variants.
    pub input_dim: usize,
    /// Number of classes.
    pub num_labels: usize,
}

/// A linear classification head over sentence embeddings.
#[derive(Module, Debug)]
pub struct SetFitHead<B: Backend> {
    /// `[input_dim] -> [num_labels]`.
    pub linear: Linear<B>,
}

impl SetFitHeadConfig {
    /// Initialise with random weights, drawn from Burn's global RNG.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SetFitHead<B> {
        SetFitHead {
            linear: LinearConfig::new(self.input_dim, self.num_labels).init(device),
        }
    }

    /// Initialise from a caller-supplied RNG, so a run is reproducible.
    ///
    /// [`Self::init`] draws from Burn's global RNG, which no per-run seed can
    /// reach: two runs with the same seed would still start from different heads
    /// and end at different weights. Reproducing Burn's documented scheme here —
    /// `U(-k, k)` with `k = sqrt(1 / input_dim)`, the same distribution
    /// `LinearConfig` uses — keeps that guarantee without mutating global state,
    /// which a library has no business doing on someone else's behalf.
    #[cfg(feature = "train")]
    #[cfg_attr(docsrs, doc(cfg(feature = "train")))]
    pub fn init_seeded<B: Backend, R: rand::Rng>(
        &self,
        device: &B::Device,
        rng: &mut R,
    ) -> SetFitHead<B> {
        use burn::module::Param;

        let k = 1.0 / (self.input_dim as f32).sqrt();
        let mut draw = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.random_range(-k..k)).collect() };

        let weight =
            Tensor::<B, 1>::from_data(draw(self.input_dim * self.num_labels).as_slice(), device)
                .reshape([self.input_dim, self.num_labels]);
        let bias = Tensor::<B, 1>::from_data(draw(self.num_labels).as_slice(), device);

        SetFitHead {
            linear: Linear {
                weight: Param::from_tensor(weight),
                bias: Some(Param::from_tensor(bias)),
            },
        }
    }
}

impl<B: Backend> SetFitHead<B> {
    /// Embeddings `[batch, input_dim]` to logits `[batch, num_labels]`.
    pub fn forward(&self, embeddings: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(embeddings)
    }

    /// Number of classes this head predicts.
    pub fn num_labels(&self) -> usize {
        self.linear.weight.dims()[1]
    }

    /// Cross-entropy loss against class indices `[batch]`.
    pub fn single_label_loss(
        &self,
        logits: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let device = logits.device();
        CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits, targets)
    }

    /// Per-class binary cross-entropy against a `[batch, num_labels]` 0/1 matrix.
    ///
    /// Each class is scored independently, which is what makes a document able to
    /// carry several labels at once — or none.
    pub fn multi_label_loss(
        &self,
        logits: Tensor<B, 2>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let device = logits.device();
        BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&device)
            .forward(logits, targets)
    }

    /// Convert logits to probabilities for the given task mode.
    pub fn probabilities(logits: Tensor<B, 2>, task: TaskMode) -> Tensor<B, 2> {
        match task {
            TaskMode::SingleLabel => softmax(logits, 1),
            TaskMode::MultiLabel { .. } => sigmoid(logits),
        }
    }
}
