//! Few-shot training, both stages.
//!
//! SetFit's insight is that you do not need many labelled examples if you first
//! teach the encoder what "similar" means for *your* labels. Stage one does that
//! contrastively: pairs drawn from the labelled examples, positive when they share
//! a label, negative when they do not, optimised so cosine similarity matches.
//! Stage two freezes that body and fits a linear classifier on its embeddings.
//!
//! Training here is a **state machine, not a loop**. [`Trainer::step`] performs one
//! optimiser step and returns; the caller decides when to take the next. Natively
//! that is a `while` loop, but in a browser it is what lets training yield to the
//! event loop between steps instead of freezing the tab for a minute.

use crate::bundle::{Bundle, Manifest};
use crate::checkpoint::Checkpoint;
use crate::config::ClassifierConfig;
use crate::error::{Result, SetFitError};
use crate::head::{SetFitHead, SetFitHeadConfig, TaskMode};
use crate::minilm::{MiniLmConfig, MiniLmModel, MiniLmVariant, check_sequence_budget};
use crate::model::{SetFitModule, embed_body};
use crate::tokenize::pad_batch;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::{Int, Tensor};
use rand::prelude::*;
use rand::rngs::StdRng;

/// One labelled training example.
///
/// A single-label example carries exactly one index; a multi-label example carries
/// any number, including none.
///
/// ```
/// use burn_setfit::Example;
///
/// // Indices point into `ClassifierConfig::labels`, in order.
/// let single = Example::single("I was charged twice this month.", 0);
/// assert_eq!(single.labels, vec![0]);
///
/// // Multi-label examples may carry several labels, or none at all -- which is
/// // how a background or "none of the above" example is expressed.
/// let both = Example::multi("Billing is broken and the site is down.", vec![0, 1]);
/// let neither = Example::multi("The quarterly review ran to time.", vec![]);
/// assert_eq!(neither.labels, Vec::<usize>::new());
/// ```
#[derive(Debug, Clone)]
pub struct Example {
    /// The text.
    pub text: String,
    /// Indices into the label set.
    pub labels: Vec<usize>,
}

impl Example {
    /// A single-label example.
    pub fn single(text: impl Into<String>, label: usize) -> Self {
        Self {
            text: text.into(),
            labels: vec![label],
        }
    }

    /// A multi-label example.
    pub fn multi(text: impl Into<String>, labels: Vec<usize>) -> Self {
        Self {
            text: text.into(),
            labels,
        }
    }

    /// Whether two examples count as a positive pair: any label in common.
    ///
    /// For single-label data this is just label equality. For multi-label data
    /// "shares a label" is the useful relation — requiring identical label *sets*
    /// would leave almost nothing positive to learn from at few-shot sizes.
    fn is_positive_with(&self, other: &Example) -> bool {
        self.labels.iter().any(|l| other.labels.contains(l))
    }
}

/// Training hyperparameters.
///
/// Defaults follow upstream SetFit, which are tuned for tens of examples per
/// class rather than thousands.
///
/// ```
/// use burn_setfit::TrainConfig;
///
/// let config = TrainConfig {
///     num_iterations: 10,   // pairs generated per example, SetFit's `R`
///     head_epochs: 40,
///     seed: 7,              // covers pair sampling, shuffling and head init
///     ..Default::default()
/// };
/// config.validate()?;
///
/// // Hyperparameters that would train nothing are refused, not adjusted.
/// assert!(TrainConfig { head_epochs: 0, ..Default::default() }.validate().is_err());
/// # Ok::<(), burn_setfit::SetFitError>(())
/// ```
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct TrainConfig {
    /// Positive/negative pairs generated per example. SetFit calls this `R`.
    pub num_iterations: usize,
    /// Passes over the generated pairs.
    pub body_epochs: usize,
    /// Body learning rate. Transformer fine-tuning territory, so small.
    pub body_lr: f64,
    /// Pairs per body step.
    pub body_batch_size: usize,
    /// Passes over the embedded examples when fitting the head.
    pub head_epochs: usize,
    /// Head learning rate. The head trains from scratch, so much larger.
    pub head_lr: f64,
    /// Examples per head step.
    pub head_batch_size: usize,
    /// Weight decay on the head — the counterpart to logistic regression's L2.
    pub head_weight_decay: f32,
    /// Longest training sequence; longer examples are truncated.
    pub max_tokens: usize,
    /// Seed for pair sampling and shuffling.
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            num_iterations: 20,
            body_epochs: 1,
            body_lr: 2e-5,
            body_batch_size: 16,
            head_epochs: 50,
            head_lr: 1e-2,
            head_batch_size: 16,
            head_weight_decay: 1e-2,
            max_tokens: crate::TRAINED_SEQ_LEN,
            seed: 42,
        }
    }
}

impl TrainConfig {
    /// Reject hyperparameters that would train nothing, or diverge.
    pub fn validate(&self) -> Result<()> {
        let positive = |name: &str, v: usize| -> Result<()> {
            if v == 0 {
                Err(SetFitError::Config(format!("{name} must be at least 1")))
            } else {
                Ok(())
            }
        };
        positive("num_iterations", self.num_iterations)?;
        positive("body_epochs", self.body_epochs)?;
        positive("head_epochs", self.head_epochs)?;
        positive("body_batch_size", self.body_batch_size)?;
        positive("head_batch_size", self.head_batch_size)?;
        positive("max_tokens", self.max_tokens)?;

        for (name, lr) in [("body_lr", self.body_lr), ("head_lr", self.head_lr)] {
            if !lr.is_finite() || lr <= 0.0 {
                return Err(SetFitError::Config(format!(
                    "{name} must be finite and positive, got {lr}"
                )));
            }
        }
        if !self.head_weight_decay.is_finite() || self.head_weight_decay < 0.0 {
            return Err(SetFitError::Config(format!(
                "head_weight_decay must be finite and non-negative, got {}",
                self.head_weight_decay
            )));
        }
        Ok(())
    }
}

/// A contrastive training pair: two example indices and a target similarity.
#[derive(Debug, Clone, Copy)]
struct Pair {
    left: usize,
    right: usize,
    /// 1.0 when the pair shares a label, 0.0 when it does not.
    target: f32,
}

/// Which stage the trainer is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Contrastively fine-tuning the body.
    Body,
    /// Fitting the classification head on frozen embeddings.
    Head,
    /// Nothing left to do.
    Done,
}

/// What one step accomplished.
///
/// ```no_run
/// # use burn::backend::{Autodiff, NdArray};
/// # use burn_setfit::{Stage, Trainer};
/// # fn drive(trainer: &mut Trainer<Autodiff<NdArray<f32>>>) -> burn_setfit::Result<()> {
/// while let Some(p) = trainer.step()? {
///     // `total_steps` is known before training starts, so a progress bar needs
///     // no guesswork.
///     eprintln!("{:?} {}/{} loss {:.4} ({:.0}%)",
///               p.stage, p.step, p.total_steps, p.loss, p.fraction() * 100.0);
///     if p.stage == Stage::Head {
///         // The body is frozen from here; only the classifier is still moving.
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Progress {
    /// The stage this step belonged to.
    pub stage: Stage,
    /// Steps completed across both stages.
    pub step: usize,
    /// Steps in total, known up front.
    pub total_steps: usize,
    /// Loss for this step.
    pub loss: f32,
}

impl Progress {
    /// Completion in `0.0..=1.0`.
    pub fn fraction(&self) -> f32 {
        if self.total_steps == 0 {
            1.0
        } else {
            self.step as f32 / self.total_steps as f32
        }
    }
}

/// Drives both SetFit stages, one optimiser step at a time.
///
/// Natively, run it to completion:
///
/// ```no_run
/// use burn::backend::{Autodiff, NdArray};
/// use burn_setfit::{Checkpoint, ClassifierConfig, Example, MiniLmVariant, TrainConfig, Trainer};
///
/// # fn main() -> burn_setfit::Result<()> {
/// let checkpoint = Checkpoint::download(MiniLmVariant::L6, None)?;
///
/// let mut trainer = Trainer::<Autodiff<NdArray<f32>>>::new(
///     &checkpoint,
///     ClassifierConfig::new(["billing", "outage"]),
///     vec![
///         Example::single("I was charged twice this month.", 0),
///         Example::single("The dashboard is returning 503s.", 1),
///         // ...eight or so per class
///     ],
///     TrainConfig::default(),
///     Default::default(),
/// )?;
///
/// trainer.fit_with(|p| eprintln!("{:?} {}/{}", p.stage, p.step, p.total_steps))?;
/// let bundle: Vec<u8> = trainer.finish()?;
/// # Ok(())
/// # }
/// ```
///
/// In a browser, drive [`Trainer::step`] yourself and yield between calls. That
/// is what this being a state machine rather than a `fit()` loop buys: a minute
/// of fine-tuning does not become a minute of frozen tab.
///
/// ```no_run
/// # use burn::backend::{Autodiff, NdArray};
/// # use burn_setfit::Trainer;
/// # fn drive(trainer: &mut Trainer<Autodiff<NdArray<f32>>>) -> burn_setfit::Result<()> {
/// while let Some(progress) = trainer.step()? {
///     render(progress.fraction());
///     // ...and in wasm, await a frame here before the next step.
/// }
/// # Ok(())
/// # }
/// # fn render(_: f32) {}
/// ```
pub struct Trainer<B: AutodiffBackend> {
    body: MiniLmModel<B>,
    head: SetFitHead<B>,
    body_optim: burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, MiniLmModel<B>, B>,
    head_optim: burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, SetFitHead<B>, B>,

    /// Token ids per example, computed once and reused across every pair.
    encoded: Vec<Vec<u32>>,
    examples: Vec<Example>,
    num_labels: usize,
    task: TaskMode,
    config: TrainConfig,
    device: B::Device,
    pad_id: u32,
    hidden_size: usize,

    /// Captured at construction so `finish` can emit a bundle unaided.
    variant: MiniLmVariant,
    body_config: MiniLmConfig,
    tokenizer_json: Vec<u8>,
    classifier: ClassifierConfig,

    pairs: Vec<Pair>,
    /// Example embeddings from the tuned body; filled when stage two begins.
    embeddings: Vec<f32>,
    /// Shuffled example order for the current head epoch.
    head_order: Vec<usize>,

    stage: Stage,
    cursor: usize,
    epoch: usize,
    step: usize,
    rng: StdRng,
}

impl<B: AutodiffBackend> Trainer<B> {
    /// Prepare training from a checkpoint and a classifier configuration.
    ///
    /// Everything needed to emit a finished bundle is captured here — the label
    /// set, the body architecture, the tokenizer — so [`Self::finish`] needs no
    /// further arguments and cannot be handed metadata that disagrees with what
    /// was actually trained.
    ///
    /// Fails if the data cannot support contrastive learning: fewer than two
    /// labels, or a label with no example that shares it and none that does not.
    pub fn new(
        checkpoint: &Checkpoint,
        classifier: ClassifierConfig,
        examples: Vec<Example>,
        config: TrainConfig,
        device: B::Device,
    ) -> Result<Self> {
        checkpoint.validate()?;
        classifier.validate()?;
        config.validate()?;
        // Both budgets, before any work: training pads to `config.max_tokens`,
        // and the packed model will chunk to `classifier.chunk.max_tokens`.
        // Catching the latter here rather than at `finish` saves discovering it
        // after the training run it invalidates.
        check_sequence_budget(&checkpoint.config, config.max_tokens, "training sequences")?;
        check_sequence_budget(
            &checkpoint.config,
            classifier.chunk.max_tokens,
            "chunk windows",
        )?;

        let num_labels = classifier.num_labels();
        if examples.len() < 2 {
            return Err(SetFitError::Training(
                "need at least two examples to form a pair".into(),
            ));
        }

        // A label index outside the configured set is a data-preparation mistake,
        // and every way of tolerating it is worse than stopping: dropping it trains
        // on an example that means nothing, and clamping it trains the wrong class.
        for (i, example) in examples.iter().enumerate() {
            if let Some(&bad) = example.labels.iter().find(|&&l| l >= num_labels) {
                return Err(SetFitError::Training(format!(
                    "example {i} carries label index {bad}, but only {num_labels} labels \
                     are configured ({:?})",
                    classifier.labels
                )));
            }
            if !classifier.task.is_multi_label() && example.labels.len() != 1 {
                return Err(SetFitError::Training(format!(
                    "example {i} carries {} labels, but a single-label task needs exactly one; \
                     use ClassifierConfig::multi_label for zero-or-more",
                    example.labels.len()
                )));
            }
        }

        let tokenizer = checkpoint.tokenizer()?;
        let mut encoded = Vec::with_capacity(examples.len());
        for e in &examples {
            encoded.push(tokenizer.encode_full(&e.text, config.max_tokens)?);
        }

        let mut rng = StdRng::seed_from_u64(config.seed);
        let pairs = generate_pairs(&examples, config.num_iterations, &mut rng)?;

        let body = checkpoint.body::<B>(&device)?;
        let hidden_size = body.hidden_size();
        // Seeded, so the whole run is reproducible and not just the sampling.
        let head = SetFitHeadConfig::new(hidden_size, num_labels).init_seeded(&device, &mut rng);

        Ok(Self {
            body,
            head,
            body_optim: AdamWConfig::new().init(),
            head_optim: AdamWConfig::new()
                .with_weight_decay(config.head_weight_decay)
                .init(),
            encoded,
            examples,
            num_labels,
            task: classifier.task,
            config,
            device: device.clone(),
            pad_id: tokenizer.special_tokens().pad,
            hidden_size,
            variant: checkpoint.variant,
            body_config: checkpoint.config.clone(),
            tokenizer_json: checkpoint.tokenizer_json.clone(),
            classifier,
            pairs,
            embeddings: Vec::new(),
            head_order: Vec::new(),
            stage: Stage::Body,
            cursor: 0,
            epoch: 0,
            step: 0,
            rng,
        })
    }

    /// Total optimiser steps across both stages, known before training starts.
    pub fn total_steps(&self) -> usize {
        let body =
            div_ceil(self.pairs.len(), self.config.body_batch_size) * self.config.body_epochs;
        let head =
            div_ceil(self.examples.len(), self.config.head_batch_size) * self.config.head_epochs;
        body + head
    }

    /// The current stage.
    pub fn stage(&self) -> Stage {
        self.stage
    }

    /// Perform one optimiser step.
    ///
    /// Returns `None` once training is complete. Between calls the caller is free
    /// to do anything — yield to a browser's event loop, report progress, stop early.
    pub fn step(&mut self) -> Result<Option<Progress>> {
        match self.stage {
            Stage::Body => self.body_step(),
            Stage::Head => self.head_step(),
            Stage::Done => Ok(None),
        }
    }

    /// One contrastive step over a batch of pairs.
    fn body_step(&mut self) -> Result<Option<Progress>> {
        if self.cursor >= self.pairs.len() {
            self.epoch += 1;
            self.cursor = 0;
            if self.epoch >= self.config.body_epochs {
                self.begin_head_stage()?;
                return self.step();
            }
            self.pairs.shuffle(&mut self.rng);
        }

        let end = (self.cursor + self.config.body_batch_size).min(self.pairs.len());
        let batch = &self.pairs[self.cursor..end];
        self.cursor = end;

        // Both sides go through the encoder as one batch, then split apart. One
        // forward pass instead of two, and identical padding for both halves.
        let n = batch.len();
        let mut sequences = Vec::with_capacity(n * 2);
        sequences.extend(batch.iter().map(|p| self.encoded[p.left].clone()));
        sequences.extend(batch.iter().map(|p| self.encoded[p.right].clone()));

        let (input_ids, attention_mask) = pad_batch::<B>(&sequences, self.pad_id, &self.device);
        let embeddings = embed_body(&self.body, input_ids, attention_mask);

        let dim = embeddings.dims()[1];
        let left = embeddings.clone().slice([0..n, 0..dim]);
        let right = embeddings.slice([n..n * 2, 0..dim]);

        // Embeddings are L2-normalised, so the cosine similarity is just a dot product.
        let cosine = (left * right).sum_dim(1).reshape([n]);

        let targets = Tensor::<B, 1>::from_data(
            batch
                .iter()
                .map(|p| p.target)
                .collect::<Vec<_>>()
                .as_slice(),
            &self.device,
        );

        // SetFit's default objective: drive cosine similarity toward 1 for pairs
        // that share a label and toward 0 for pairs that do not.
        let loss = (cosine - targets).powf_scalar(2.0).mean();
        let loss_value = scalar(&loss);

        let grads = GradientsParams::from_grads(loss.backward(), &self.body);
        self.body = self
            .body_optim
            .step(self.config.body_lr, self.body.clone(), grads);

        self.step += 1;
        Ok(Some(Progress {
            stage: Stage::Body,
            step: self.step,
            total_steps: self.total_steps(),
            loss: loss_value,
        }))
    }

    /// Freeze the body and embed every example once, ready for the head.
    ///
    /// This is the expensive transition: one forward pass over the whole training
    /// set. It happens inside a single `step()` call, so a browser sees one long
    /// step rather than a silent stall.
    fn begin_head_stage(&mut self) -> Result<()> {
        let frozen = self.body.valid();
        let mut embeddings = Vec::with_capacity(self.examples.len() * self.hidden_size);

        for batch in self.encoded.chunks(self.config.head_batch_size) {
            let (input_ids, attention_mask) =
                pad_batch::<B::InnerBackend>(batch, self.pad_id, &self.device);
            let embedded = embed_body(&frozen, input_ids, attention_mask);
            embeddings.extend(
                embedded
                    .into_data()
                    .into_vec::<f32>()
                    .map_err(|e| SetFitError::Training(format!("{e:?}")))?,
            );
        }

        self.embeddings = embeddings;
        self.head_order = (0..self.examples.len()).collect();
        self.head_order.shuffle(&mut self.rng);
        self.stage = Stage::Head;
        self.cursor = 0;
        self.epoch = 0;
        Ok(())
    }

    /// One supervised step over a batch of frozen embeddings.
    fn head_step(&mut self) -> Result<Option<Progress>> {
        if self.cursor >= self.head_order.len() {
            self.epoch += 1;
            self.cursor = 0;
            if self.epoch >= self.config.head_epochs {
                self.stage = Stage::Done;
                return Ok(None);
            }
            self.head_order.shuffle(&mut self.rng);
        }

        let end = (self.cursor + self.config.head_batch_size).min(self.head_order.len());
        let indices = &self.head_order[self.cursor..end];
        self.cursor = end;

        let n = indices.len();
        let dim = self.hidden_size;
        let mut features = Vec::with_capacity(n * dim);
        for &i in indices {
            features.extend_from_slice(&self.embeddings[i * dim..(i + 1) * dim]);
        }

        let features =
            Tensor::<B, 1>::from_data(features.as_slice(), &self.device).reshape([n, dim]);
        let logits = self.head.forward(features);

        let loss = match self.task {
            TaskMode::SingleLabel => {
                let targets: Vec<i64> = indices
                    .iter()
                    // Validated at construction: exactly one label, in range.
                    .map(|&i| self.examples[i].labels[0] as i64)
                    .collect();
                let targets = Tensor::<B, 1, Int>::from_data(targets.as_slice(), &self.device);
                self.head.single_label_loss(logits, targets)
            }
            TaskMode::MultiLabel { .. } => {
                let mut targets = vec![0i64; n * self.num_labels];
                for (row, &i) in indices.iter().enumerate() {
                    for &l in &self.examples[i].labels {
                        targets[row * self.num_labels + l] = 1;
                    }
                }
                let targets = Tensor::<B, 1, Int>::from_data(targets.as_slice(), &self.device)
                    .reshape([n, self.num_labels]);
                self.head.multi_label_loss(logits, targets)
            }
        };
        let loss_value = scalar(&loss);

        let grads = GradientsParams::from_grads(loss.backward(), &self.head);
        self.head = self
            .head_optim
            .step(self.config.head_lr, self.head.clone(), grads);

        self.step += 1;
        Ok(Some(Progress {
            stage: Stage::Head,
            step: self.step,
            total_steps: self.total_steps(),
            loss: loss_value,
        }))
    }

    /// Run every remaining step, reporting progress as it goes.
    ///
    /// The blocking convenience wrapper. In a browser, drive [`Self::step`] yourself.
    pub fn fit_with<F: FnMut(Progress)>(&mut self, mut on_progress: F) -> Result<()> {
        while let Some(p) = self.step()? {
            on_progress(p);
        }
        Ok(())
    }

    /// Take the trained model, detached from the autodiff graph.
    ///
    /// Available at any point; a partially trained model is a legitimate thing to
    /// inspect. [`Self::finish`] is the one that insists training completed.
    pub fn into_model(self) -> SetFitModule<B::InnerBackend> {
        SetFitModule {
            body: self.body.valid(),
            head: self.head.valid(),
        }
    }

    /// Pack the trained model into a `.setfit` bundle.
    ///
    /// Refuses to run before training completes. Packing early would produce a
    /// well-formed bundle containing a head that had never been fitted — a model
    /// that loads cleanly and predicts noise, which is far harder to diagnose than
    /// an error here.
    ///
    /// The bundle is everything needed to classify — weights, tokenizer and
    /// configuration — so it is one file to ship and one `fetch` to load. Use
    /// [`Self::into_model`] instead to inspect a partially trained model
    /// deliberately.
    pub fn finish(self) -> Result<Vec<u8>> {
        if self.stage != Stage::Done {
            return Err(SetFitError::Training(format!(
                "training is still in the {:?} stage at step {} of {}; \
                 call step() until it returns None, or use into_model() deliberately",
                self.stage,
                self.step,
                self.total_steps()
            )));
        }

        let manifest = Manifest::new(
            self.variant,
            self.body_config.clone(),
            self.classifier.clone(),
        );
        let tokenizer_json = self.tokenizer_json.clone();
        Bundle::pack(&self.into_model(), &manifest, &tokenizer_json)
    }
}

/// Read a scalar loss back to the host.
fn scalar<B: AutodiffBackend>(loss: &Tensor<B, 1>) -> f32 {
    loss.clone().into_scalar().to_f32()
}

fn div_ceil(a: usize, b: usize) -> usize {
    if b == 0 { 0 } else { a.div_ceil(b) }
}

/// Generate contrastive pairs, following SetFit's sampling scheme.
///
/// For each example, `num_iterations` positive partners and the same number of
/// negative ones, sampled with replacement. The result is balanced by
/// construction: exactly as many positives as negatives.
fn generate_pairs(
    examples: &[Example],
    num_iterations: usize,
    rng: &mut StdRng,
) -> Result<Vec<Pair>> {
    let n = examples.len();
    let mut pairs = Vec::with_capacity(n * num_iterations * 2);
    let mut lonely: Vec<usize> = Vec::new();

    for (i, example) in examples.iter().enumerate() {
        let positives: Vec<usize> = (0..n)
            .filter(|&j| j != i && example.is_positive_with(&examples[j]))
            .collect();
        let negatives: Vec<usize> = (0..n)
            .filter(|&j| j != i && !example.is_positive_with(&examples[j]))
            .collect();

        if positives.is_empty() || negatives.is_empty() {
            lonely.push(i);
            continue;
        }

        for _ in 0..num_iterations {
            pairs.push(Pair {
                left: i,
                right: positives[rng.random_range(0..positives.len())],
                target: 1.0,
            });
            pairs.push(Pair {
                left: i,
                right: negatives[rng.random_range(0..negatives.len())],
                target: 0.0,
            });
        }
    }

    if pairs.is_empty() {
        return Err(SetFitError::Training(
            "no usable pairs: every example needs at least one example sharing a label \
             and one not sharing any"
                .into(),
        ));
    }
    // Some examples being unusable is survivable; silently dropping them is not.
    if !lonely.is_empty() {
        // Contributing no pairs means contributing nothing to stage one. The
        // examples still train the head, so this is a warning, not an error.
        #[cfg(feature = "native")]
        eprintln!(
            "burn-setfit: {} example(s) formed no contrastive pairs and did not \
             contribute to body fine-tuning (indices: {:?})",
            lonely.len(),
            lonely
        );
    }

    pairs.shuffle(rng);
    Ok(pairs)
}
