//! Classifying documents, including ones too long to encode in one pass.
//!
//! The pipeline is: window the document, encode windows in batches, fold each
//! window's logits into a running accumulator, then decode once at the end.
//! Nothing accumulates per chunk except bounded evidence, so classifying a
//! gigabyte costs the same memory as classifying a paragraph.

use crate::bundle::{Bundle, Manifest};
use crate::chunk::{Chunk, Chunker};
use crate::config::ClassifierConfig;
use crate::error::Result;
use crate::head::TaskMode;
use crate::model::SetFitModule;
use crate::reduce::{DocAccumulator, Reducer};
use crate::tokenize::{Tokenizer, pad_batch};
use burn::tensor::backend::Backend;
use core::ops::Range;

/// How many chunks are encoded per forward pass.
///
/// Larger batches amortise the encoder better, at proportionally more peak memory.
pub const DEFAULT_BATCH_SIZE: usize = 8;

/// A chunk that contributed strongly to a predicted label.
///
/// Kept so a label can be traced back to the passage that caused it — the part of
/// long-document classification that is otherwise entirely opaque.
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Where this chunk sits in the source document.
    pub byte_range: Range<usize>,
    /// The label this chunk is evidence *for*.
    pub label: usize,
    /// That label's probability for this chunk alone.
    pub score: f32,
}

/// The outcome of classifying one document.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Probability per label, in manifest label order.
    pub scores: Vec<f32>,
    /// Decoded labels: one index for single-label, zero or more for multi-label.
    pub predicted: Vec<usize>,
    /// How many chunks the document produced.
    pub chunks_seen: usize,
    /// Chunks supporting the predicted labels, strongest first.
    ///
    /// Only ever mentions labels that were actually predicted: a chunk that was
    /// confident about some *other* label is not evidence for this verdict.
    pub evidence: Vec<Evidence>,
}

impl Prediction {
    /// Resolve predicted indices to label names.
    pub fn labels<'a>(&self, manifest: &'a Manifest) -> Vec<&'a str> {
        self.predicted
            .iter()
            .filter_map(|&i| manifest.labels().get(i).map(String::as_str))
            .collect()
    }

    /// The single highest-scoring label, whatever the task mode.
    pub fn top(&self) -> Option<(usize, f32)> {
        self.scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, &s)| (i, s))
    }
}

/// A loaded SetFit model, ready to classify.
pub struct Classifier<B: Backend> {
    module: SetFitModule<B>,
    tokenizer: Tokenizer,
    manifest: Manifest,
    device: B::Device,
    batch_size: usize,
    evidence_limit: usize,
}

impl<B: Backend> Classifier<B> {
    /// Load from packed bundle bytes — a file read natively, a `fetch` in a browser.
    pub fn from_bundle(bytes: &[u8], device: B::Device) -> Result<Self> {
        let bundle = Bundle::unpack(bytes)?;
        Ok(Self {
            module: bundle.load_module(&device)?,
            tokenizer: bundle.load_tokenizer()?,
            manifest: bundle.manifest,
            device,
            batch_size: DEFAULT_BATCH_SIZE,
            evidence_limit: 5,
        })
    }

    /// Assemble from already-loaded parts.
    pub fn new(
        module: SetFitModule<B>,
        tokenizer: Tokenizer,
        manifest: Manifest,
        device: B::Device,
    ) -> Result<Self> {
        manifest.validate()?;
        Ok(Self {
            module,
            tokenizer,
            manifest,
            device,
            batch_size: DEFAULT_BATCH_SIZE,
            evidence_limit: 5,
        })
    }

    /// Set how many chunks are encoded per forward pass.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Override how per-chunk scores collapse into a document verdict.
    ///
    /// Reduction happens after the head, so it is a decode-time choice: switching
    /// it needs no retraining and no repacking. Handy for tuning a long-document
    /// policy against a validation set.
    pub fn with_reducer(mut self, reducer: Reducer) -> Self {
        self.manifest.classifier.reducer = reducer;
        self
    }

    /// Reduce chunks in blocks of `fanout`, recursively, rather than all at once.
    ///
    /// Like [`Self::with_reducer`], a decode-time choice.
    pub fn with_hierarchy(mut self, fanout: usize) -> Self {
        self.manifest.classifier.hierarchy_fanout = Some(fanout);
        self
    }

    /// A copy of this classifier with a different decision threshold.
    ///
    /// Cheap: the weights are shared, only the configuration is copied. Useful for
    /// sweeping a threshold over a validation set without reloading the model.
    pub fn clone_with_threshold(&self, threshold: f32) -> Self
    where
        SetFitModule<B>: Clone,
    {
        Self {
            module: self.module.clone(),
            tokenizer: self.tokenizer.clone(),
            manifest: self.manifest.clone(),
            device: self.device.clone(),
            batch_size: self.batch_size,
            evidence_limit: self.evidence_limit,
        }
        .with_threshold(threshold)
    }

    /// Override the multi-label decision threshold.
    ///
    /// No effect on a single-label model, which has nothing to threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        if let TaskMode::MultiLabel { .. } = self.manifest.classifier.task {
            self.manifest.classifier.task = TaskMode::MultiLabel { threshold };
        }
        self
    }

    /// Set how many contributing chunks to retain. Zero disables the bookkeeping.
    pub fn with_evidence_limit(mut self, limit: usize) -> Self {
        self.evidence_limit = limit;
        self
    }

    /// The model's metadata, including the checkpoint it was built on.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// How this model classifies.
    pub fn cfg(&self) -> &ClassifierConfig {
        &self.manifest.classifier
    }

    /// Label names in head-output order.
    pub fn labels(&self) -> &[String] {
        &self.manifest.classifier.labels
    }

    /// Classify an in-memory document of any length.
    pub fn classify(&self, text: &str) -> Result<Prediction> {
        self.classify_stream(core::iter::once(text.to_string()))
    }

    /// Classify a document arriving in pieces.
    ///
    /// The source is consumed lazily and may be arbitrarily long: chunks are
    /// encoded and folded as they are produced, and only the current batch is
    /// ever resident.
    pub fn classify_stream<I: Iterator<Item = String>>(&self, source: I) -> Result<Prediction> {
        let num_labels = self.cfg().num_labels();
        let mut accumulator =
            DocAccumulator::new(self.cfg().reducer, self.cfg().hierarchy_fanout, num_labels);
        // Tracked per label, because which label matters is not known until the
        // document is fully reduced. Bounded at limit x labels either way.
        let mut evidence: Vec<Vec<Evidence>> = vec![Vec::new(); num_labels];

        let chunker = Chunker::new(source, &self.tokenizer, self.cfg().chunk);
        let mut batch: Vec<Chunk> = Vec::with_capacity(self.batch_size);

        for chunk in chunker {
            batch.push(chunk?);
            if batch.len() == self.batch_size {
                self.run_batch(&batch, &mut accumulator, &mut evidence)?;
                batch.clear();
            }
        }
        if !batch.is_empty() {
            self.run_batch(&batch, &mut accumulator, &mut evidence)?;
        }

        let logits = accumulator.finish();
        let scores = to_probabilities(&logits, self.cfg().task);
        let predicted = decode(&scores, self.cfg().task);

        // Keep only the evidence for labels the document actually got.
        let mut evidence: Vec<Evidence> = predicted
            .iter()
            .flat_map(|&l| evidence.get(l).cloned().unwrap_or_default())
            .collect();
        evidence.sort_by(|a, b| b.score.total_cmp(&a.score));
        evidence.truncate(self.evidence_limit);

        Ok(Prediction {
            scores,
            predicted,
            chunks_seen: accumulator.count(),
            evidence,
        })
    }

    /// Encode one batch of chunks and fold their scores in.
    fn run_batch(
        &self,
        batch: &[Chunk],
        accumulator: &mut DocAccumulator,
        evidence: &mut [Vec<Evidence>],
    ) -> Result<()> {
        let sequences: Vec<Vec<u32>> = batch.iter().map(|c| c.ids.clone()).collect();
        let (input_ids, attention_mask) = pad_batch::<B>(
            &sequences,
            self.tokenizer.special_tokens().pad,
            &self.device,
        );

        let logits = self.module.forward(input_ids, attention_mask);
        let data = logits
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| crate::SetFitError::Store(format!("{e:?}")))?;

        let num_labels = self.cfg().num_labels();
        for (i, chunk) in batch.iter().enumerate() {
            let row = &data[i * num_labels..(i + 1) * num_labels];
            accumulator.push(chunk.token_count as f32, row);
            self.record_evidence(chunk, row, evidence);
        }

        Ok(())
    }

    /// Record this chunk against every label it scores well on.
    ///
    /// One list per label, each capped, so the winning label's supporting passages
    /// are available whichever way the document ends up being decided.
    fn record_evidence(&self, chunk: &Chunk, logits: &[f32], evidence: &mut [Vec<Evidence>]) {
        if self.evidence_limit == 0 {
            return;
        }
        let probs = to_probabilities(logits, self.cfg().task);

        for (label, &score) in probs.iter().enumerate() {
            let slot = &mut evidence[label];
            if slot.len() == self.evidence_limit
                && score <= slot.last().map(|e| e.score).unwrap_or(f32::MIN)
            {
                continue;
            }
            let at = slot.partition_point(|e| e.score > score);
            slot.insert(
                at,
                Evidence {
                    byte_range: chunk.byte_range.clone(),
                    label,
                    score,
                },
            );
            slot.truncate(self.evidence_limit);
        }
    }

    /// Embed short texts directly, bypassing chunking.
    ///
    /// Inputs longer than the configured window are truncated, matching what
    /// `sentence-transformers` does. Long documents belong in [`Self::classify`].
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // An empty batch would reach the encoder as a zero-row tensor and panic
        // inside the backend rather than returning nothing.
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let max_tokens = self.cfg().chunk.max_tokens;
        let mut sequences = Vec::with_capacity(texts.len());
        for t in texts {
            sequences.push(self.tokenizer.encode_full(t, max_tokens)?);
        }

        let (input_ids, attention_mask) = pad_batch::<B>(
            &sequences,
            self.tokenizer.special_tokens().pad,
            &self.device,
        );
        let embeddings = self.module.embed(input_ids, attention_mask);
        let dim = embeddings.dims()[1];
        let data = embeddings
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| crate::SetFitError::Store(format!("{e:?}")))?;

        Ok(data.chunks(dim).map(<[f32]>::to_vec).collect())
    }
}

/// Logits to probabilities, in the space the task mode implies.
pub fn to_probabilities(logits: &[f32], task: TaskMode) -> Vec<f32> {
    match task {
        TaskMode::SingleLabel => {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum).collect()
        }
        TaskMode::MultiLabel { .. } => logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect(),
    }
}

/// Turn probabilities into predicted label indices.
pub fn decode(scores: &[f32], task: TaskMode) -> Vec<usize> {
    match task {
        TaskMode::SingleLabel => scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| vec![i])
            .unwrap_or_default(),
        // Genuinely zero or more: a document that matches nothing returns nothing,
        // rather than being forced into its least-bad class.
        TaskMode::MultiLabel { threshold } => scores
            .iter()
            .enumerate()
            .filter(|(_, s)| **s >= threshold)
            .map(|(i, _)| i)
            .collect(),
    }
}
