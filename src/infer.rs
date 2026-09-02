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
use crate::readback::{self, blocking};
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

/// One chunk's verdict, before anything is reduced.
///
/// [`Classifier::classify`] answers "what is this document about", which needs a
/// [`Reducer`] to collapse many chunks into one answer — and over a long
/// document that collapse is lossy in ways
/// [#4](https://github.com/jcorrie/burn-setfit/issues/4) measures. This is the
/// same work with the collapse left out: one row per passage, scored
/// independently, for a caller who would rather aggregate it themselves or show
/// it as-is.
#[derive(Debug, Clone)]
pub struct ChunkPrediction {
    /// Where this chunk sits in the source document.
    pub byte_range: Range<usize>,
    /// Content tokens, excluding specials — this chunk's weight in a reduction.
    pub token_count: usize,
    /// Per-label probabilities for this chunk alone.
    pub scores: Vec<f32>,
    /// What this chunk on its own decodes to, background class included.
    pub predicted: Vec<usize>,
}

/// The outcome of classifying one document.
///
/// ```no_run
/// use burn::backend::NdArray;
/// use burn_setfit::Classifier;
///
/// # fn main() -> burn_setfit::Result<()> {
/// # let classifier = Classifier::<NdArray<f32>>::from_bundle(&[], Default::default())?;
/// # let document = "";
/// let prediction = classifier.classify(document)?;
///
/// // `predicted` is one index for single-label, zero or more for multi-label.
/// for name in prediction.labels(classifier.manifest()) {
///     println!("{name}");
/// }
///
/// // Every predicted label can be traced back to the passage that caused it.
/// for evidence in &prediction.evidence {
///     let passage = &document[evidence.byte_range.clone()];
///     println!("{:.3}  {passage}", evidence.score);
/// }
/// # Ok(())
/// # }
/// ```
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
///
/// ```no_run
/// use burn::backend::NdArray;
/// use burn_setfit::{Classifier, Reducer};
///
/// # fn main() -> burn_setfit::Result<()> {
/// // `?` composes: the file read and the unpack share one error type.
/// let bundle = std::fs::read("support.setfit")?;
/// let classifier = Classifier::<NdArray<f32>>::from_bundle(&bundle, Default::default())?;
///
/// let prediction = classifier.classify("I was charged twice this month.")?;
/// println!("{:?} from {} chunks", prediction.labels(classifier.manifest()), prediction.chunks_seen);
///
/// // Reduction, hierarchy and threshold are decode-time choices, so sweeping
/// // them needs no retraining and no repacking.
/// let strict = classifier
///     .with_reducer(Reducer::TopKMeanLogits { k: 3 })
///     .with_hierarchy(8);
/// # Ok(())
/// # }
/// ```
///
/// The bundle carries its own error handling: reading a file that is not one is
/// refused rather than misinterpreted.
///
/// ```
/// use burn::backend::NdArray;
/// use burn_setfit::Classifier;
///
/// assert!(Classifier::<NdArray<f32>>::from_bundle(b"not a bundle", Default::default()).is_err());
/// ```
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

    /// [`Self::classify`], awaiting the device instead of blocking on it.
    ///
    /// Same result, and the same work: this is the implementation, and the
    /// synchronous method above is a wrapper that drives it to completion.
    /// Reach for this one on `wasm32` with a GPU backend, where blocking is
    /// not available and the wrapper can only report that — see
    /// [`SetFitError::Readback`](crate::SetFitError::Readback).
    pub async fn classify_async(&self, text: &str) -> Result<Prediction> {
        self.classify_stream_async(core::iter::once(text.to_string()))
            .await
    }

    /// Classify a document arriving in pieces.
    ///
    /// The source is consumed lazily and may be arbitrarily long: chunks are
    /// encoded and folded as they are produced, and only the current batch is
    /// ever resident.
    ///
    /// ```no_run
    /// use burn::backend::NdArray;
    /// use burn_setfit::Classifier;
    /// use std::io::{BufRead, BufReader};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let classifier = Classifier::<NdArray<f32>>::from_bundle(&[], Default::default())?;
    /// // A file of any size, never held in memory as a whole.
    /// let file = BufReader::new(std::fs::File::open("transcript.txt")?);
    /// let lines = file.lines().map_while(std::result::Result::ok).map(|l| l + "\n");
    ///
    /// let prediction = classifier.classify_stream(lines)?;
    /// println!("{} chunks", prediction.chunks_seen);
    /// # Ok(())
    /// # }
    /// ```
    pub fn classify_stream<I: Iterator<Item = String>>(&self, source: I) -> Result<Prediction> {
        blocking("classifying a document", self.classify_stream_async(source))
    }

    /// [`Self::classify_stream`], awaiting the device instead of blocking on it.
    pub async fn classify_stream_async<I: Iterator<Item = String>>(
        &self,
        source: I,
    ) -> Result<Prediction> {
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
                self.run_batch(&batch, &mut accumulator, &mut evidence)
                    .await?;
                batch.clear();
            }
        }
        if !batch.is_empty() {
            self.run_batch(&batch, &mut accumulator, &mut evidence)
                .await?;
        }

        let logits = accumulator.finish();
        let scores = to_probabilities(&logits, self.cfg().task);
        let predicted = decode_against_background(&scores, self.cfg().task, self.cfg().background);

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

    /// Classify each passage on its own, without reducing to a document verdict.
    ///
    /// The reducer is the lossy step over a long document, and
    /// [#4](https://github.com/jcorrie/burn-setfit/issues/4) is largely about
    /// how lossy. This hands back what the model actually saw — one row per
    /// chunk, with the byte range it came from — and leaves the aggregating to
    /// the caller. Useful for showing *where* in a document a label came from,
    /// and for choosing a reduction after the fact rather than before.
    ///
    /// Memory is proportional to the number of chunks, unlike
    /// [`Self::classify_stream`], which folds as it goes and stays constant. A
    /// genuinely unbounded input wants that one.
    ///
    /// ```no_run
    /// # use burn::backend::NdArray;
    /// # use burn_setfit::Classifier;
    /// # fn main() -> burn_setfit::Result<()> {
    /// # let classifier = Classifier::<NdArray<f32>>::from_bundle(&[], Default::default())?;
    /// # let document = "";
    /// for chunk in classifier.classify_chunks(document)? {
    ///     let labels = classifier.labels();
    ///     for label in &chunk.predicted {
    ///         println!("{:?}: {}", chunk.byte_range, labels[*label]);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn classify_chunks(&self, text: &str) -> Result<Vec<ChunkPrediction>> {
        blocking(
            "classifying a document by chunk",
            self.classify_chunks_async(text),
        )
    }

    /// [`Self::classify_chunks`], awaiting the device instead of blocking on it.
    pub async fn classify_chunks_async(&self, text: &str) -> Result<Vec<ChunkPrediction>> {
        let chunker = Chunker::new(
            core::iter::once(text.to_string()),
            &self.tokenizer,
            self.cfg().chunk,
        );

        let mut out = Vec::new();
        let mut batch: Vec<Chunk> = Vec::with_capacity(self.batch_size);

        for chunk in chunker {
            batch.push(chunk?);
            if batch.len() == self.batch_size {
                self.push_chunk_predictions(&batch, &mut out).await?;
                batch.clear();
            }
        }
        if !batch.is_empty() {
            self.push_chunk_predictions(&batch, &mut out).await?;
        }

        Ok(out)
    }

    /// Score one batch and append a row per chunk.
    async fn push_chunk_predictions(
        &self,
        batch: &[Chunk],
        out: &mut Vec<ChunkPrediction>,
    ) -> Result<()> {
        let rows = self.forward_rows(batch).await?;
        for (chunk, row) in batch.iter().zip(&rows) {
            let scores = to_probabilities(row, self.cfg().task);
            // Decoded per chunk with the same rules the document gets, so a
            // caller comparing the two is comparing like with like.
            let predicted =
                decode_against_background(&scores, self.cfg().task, self.cfg().background);
            out.push(ChunkPrediction {
                byte_range: chunk.byte_range.clone(),
                token_count: chunk.token_count,
                scores,
                predicted,
            });
        }
        Ok(())
    }

    /// Encode one batch of chunks and read their logits back, one row each.
    ///
    /// The whole device-touching part of inference, shared by the reducing path
    /// and the per-chunk one so they cannot disagree about what a chunk scores.
    async fn forward_rows(&self, batch: &[Chunk]) -> Result<Vec<Vec<f32>>> {
        let sequences: Vec<Vec<u32>> = batch.iter().map(|c| c.ids.clone()).collect();
        let (input_ids, attention_mask) = pad_batch::<B>(
            &sequences,
            self.tokenizer.special_tokens().pad,
            &self.device,
        );

        let logits = self.module.forward(input_ids, attention_mask);
        let data = readback::floats(logits).await?;

        let num_labels = self.cfg().num_labels();
        Ok(data.chunks(num_labels).map(<[f32]>::to_vec).collect())
    }

    /// Encode one batch of chunks and fold their scores in.
    async fn run_batch(
        &self,
        batch: &[Chunk],
        accumulator: &mut DocAccumulator,
        evidence: &mut [Vec<Evidence>],
    ) -> Result<()> {
        let rows = self.forward_rows(batch).await?;
        for (chunk, row) in batch.iter().zip(&rows) {
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
    ///
    /// ```no_run
    /// # use burn::backend::NdArray;
    /// # use burn_setfit::Classifier;
    /// # fn main() -> burn_setfit::Result<()> {
    /// # let classifier = Classifier::<NdArray<f32>>::from_bundle(&[], Default::default())?;
    /// let vectors = classifier.embed(&["The weather is lovely today.", "It's so sunny outside!"])?;
    ///
    /// // L2-normalised, so cosine similarity is a dot product.
    /// let similarity: f32 = vectors[0].iter().zip(&vectors[1]).map(|(a, b)| a * b).sum();
    /// println!("{similarity:.4}");
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        blocking("embedding a batch", self.embed_async(texts))
    }

    /// [`Self::embed`], awaiting the device instead of blocking on it.
    pub async fn embed_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
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
        let data = readback::floats(embeddings).await?;

        Ok(data.chunks(dim).map(<[f32]>::to_vec).collect())
    }
}

/// Logits to probabilities, in the space the task mode implies.
///
/// ```
/// use burn_setfit::TaskMode;
/// use burn_setfit::infer::to_probabilities;
///
/// // Single-label scores are a softmax, so they sum to one: raising every
/// // logit changes nothing, only the gaps between them matter.
/// let softmax = to_probabilities(&[2.0, 1.0, 0.0], TaskMode::SingleLabel);
/// assert!((softmax.iter().sum::<f32>() - 1.0).abs() < 1e-6);
///
/// // Multi-label scores are independent sigmoids, so they need not.
/// let sigmoids = to_probabilities(&[2.0, 1.0, 0.0], TaskMode::multi_label());
/// assert!(sigmoids.iter().sum::<f32>() > 1.0);
/// assert_eq!(sigmoids[2], 0.5);
/// ```
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
///
/// ```
/// use burn_setfit::TaskMode;
/// use burn_setfit::infer::decode;
///
/// let scores = [0.2, 0.7, 0.6];
///
/// // Single-label always returns exactly one index.
/// assert_eq!(decode(&scores, TaskMode::SingleLabel), vec![1]);
///
/// // Multi-label returns genuinely zero or more: a document that matches
/// // nothing returns nothing, rather than its least-bad class.
/// assert_eq!(decode(&scores, TaskMode::MultiLabel { threshold: 0.5 }), vec![1, 2]);
/// assert_eq!(decode(&scores, TaskMode::MultiLabel { threshold: 0.9 }), Vec::<usize>::new());
/// ```
pub fn decode(scores: &[f32], task: TaskMode) -> Vec<usize> {
    decode_against_background(scores, task, None)
}

/// [`decode`], with one label acting as "none of the above".
///
/// The background class is never predicted. What it does instead is set the bar:
/// a label counts only if it is stronger than "nothing in particular".
///
/// - **Multi-label** — a label must clear the threshold *and* outscore the
///   background. The second test is the one that matters over long documents,
///   where scores compress toward the middle and a fixed threshold stops
///   discriminating long before the ordering does.
/// - **Single-label** — the argmax still wins, but if it *is* the background
///   class the document predicts nothing. That is how a softmax abstains: not by
///   scoring low, which it cannot do, but by having somewhere to put the mass.
///
/// ```
/// use burn_setfit::TaskMode;
/// use burn_setfit::infer::decode_against_background;
///
/// // Index 2 is "other". Multi-label: `a` beats it, `b` does not.
/// let scores = [0.61, 0.30, 0.45];
/// let multi = TaskMode::MultiLabel { threshold: 0.4 };
/// assert_eq!(decode_against_background(&scores, multi, Some(2)), vec![0]);
///
/// // Without a background class, `b` clears 0.4 and is predicted too.
/// assert_eq!(decode_against_background(&scores, multi, None), vec![0, 2]);
///
/// // Single-label abstains when the background class wins outright.
/// let filler = [0.20, 0.15, 0.65];
/// let single = TaskMode::SingleLabel;
/// assert!(decode_against_background(&filler, single, Some(2)).is_empty());
/// assert_eq!(decode_against_background(&filler, single, None), vec![2]);
/// ```
pub fn decode_against_background(
    scores: &[f32],
    task: TaskMode,
    background: Option<usize>,
) -> Vec<usize> {
    let Some(background) = background else {
        return decode_plain(scores, task);
    };
    let floor = scores.get(background).copied().unwrap_or(f32::NEG_INFINITY);

    match task {
        // The argmax is unchanged; what changes is that landing on the
        // background class now means "none", not "this document is filler".
        TaskMode::SingleLabel => match decode_plain(scores, task).first() {
            Some(&winner) if winner != background => vec![winner],
            _ => Vec::new(),
        },
        TaskMode::MultiLabel { threshold } => scores
            .iter()
            .enumerate()
            .filter(|(i, s)| *i != background && **s >= threshold && **s > floor)
            .map(|(i, _)| i)
            .collect(),
    }
}

fn decode_plain(scores: &[f32], task: TaskMode) -> Vec<usize> {
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
