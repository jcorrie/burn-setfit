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
use crate::tokenize::{Tokenizer, pad_batch, pad_batch_to};
use burn::tensor::Device;
use core::num::NonZeroUsize;
use core::ops::Range;

/// How many chunks are encoded per forward pass.
///
/// Larger batches amortise the encoder better, at proportionally more peak memory.
pub const DEFAULT_BATCH_SIZE: usize = 8;

/// Most chunks in one forward pass of [`Classifier::classify_many`].
///
/// Many short documents are what that method is for, and a short document is
/// one short chunk, so its passes hold far more rows than
/// [`DEFAULT_BATCH_SIZE`]. [`MANY_MAX_TOKENS`] is what bounds them when the
/// chunks are long.
pub const MANY_MAX_ROWS: usize = 64;

/// Most padded tokens (rows × padded length) in one forward pass of
/// [`Classifier::classify_many`].
///
/// Sixteen full 256-token windows, or 64 short ones. Capping by tokens rather
/// than rows alone is what keeps peak memory flat whatever the mix of lengths.
pub const MANY_MAX_TOKENS: usize = 4096;

/// [`Classifier::classify_many`] pads each pass to a multiple of this many
/// tokens.
///
/// Padding is masked out, so it changes no score. It bounds how many distinct
/// shapes the encoder sees, which matters on a GPU backend: each new shape
/// compiles kernels, and that costs far more than a few padded columns.
pub const LENGTH_BUCKET: usize = 8;

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
/// use burn_setfit::Classifier;
///
/// # fn main() -> burn_setfit::Result<()> {
/// # let classifier = Classifier::from_bundle(&[], Default::default())?;
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
/// use burn_setfit::{Classifier, Reducer};
///
/// # fn main() -> burn_setfit::Result<()> {
/// // `?` composes: the file read and the unpack share one error type.
/// let bundle = std::fs::read("support.setfit")?;
/// let classifier = Classifier::from_bundle(&bundle, Default::default())?;
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
/// use burn_setfit::Classifier;
///
/// assert!(Classifier::from_bundle(b"not a bundle", Default::default()).is_err());
/// ```
pub struct Classifier {
    module: SetFitModule,
    tokenizer: Tokenizer,
    manifest: Manifest,
    device: Device,
    batch_size: usize,
    evidence_limit: usize,
}

impl Classifier {
    /// Load from packed bundle bytes — a file read natively, a `fetch` in a browser.
    pub fn from_bundle(bytes: &[u8], device: Device) -> Result<Self> {
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
        module: SetFitModule,
        tokenizer: Tokenizer,
        manifest: Manifest,
        device: Device,
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

    /// Decide per chunk and predict labels that at least `min_chunks` chunks vote
    /// for. See [`ClassifierConfig::with_chunk_votes`].
    ///
    /// A decode-time choice like [`Self::with_reducer`]. Pass `None` to go back
    /// to decoding the reduced scores.
    pub fn with_chunk_votes(mut self, min_chunks: Option<NonZeroUsize>) -> Self {
        self.manifest.classifier.chunk_votes = min_chunks;
        self
    }

    /// A copy of this classifier with a different decision threshold.
    ///
    /// Cheap: the weights are shared, only the configuration is copied. Useful for
    /// sweeping a threshold over a validation set without reloading the model.
    pub fn clone_with_threshold(&self, threshold: f32) -> Self
    where
        SetFitModule: Clone,
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
    /// use burn_setfit::Classifier;
    /// use std::io::{BufRead, BufReader};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let classifier = Classifier::from_bundle(&[], Default::default())?;
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
        let mut fold = self.new_fold();
        let chunker = Chunker::new(source, &self.tokenizer, self.cfg().chunk);
        let mut batch: Vec<Chunk> = Vec::with_capacity(self.batch_size);

        for chunk in chunker {
            batch.push(chunk?);
            if batch.len() == self.batch_size {
                self.run_batch(&batch, &mut fold).await?;
                batch.clear();
            }
        }
        if !batch.is_empty() {
            self.run_batch(&batch, &mut fold).await?;
        }

        Ok(self.finish_fold(fold))
    }

    /// Classify many documents together, one [`Prediction`] each, in order.
    ///
    /// Each prediction is the one [`Self::classify`] gives that document alone.
    /// What changes is how the work is laid out: every document is chunked
    /// first, and chunks from all of them share forward passes, shortest
    /// first, so a column of short texts costs a few passes rather than one
    /// each. A forward pass has a large fixed cost on a GPU, which for a short
    /// text is nearly all of it.
    ///
    /// Passes hold at most [`MANY_MAX_ROWS`] chunks and [`MANY_MAX_TOKENS`]
    /// padded tokens; [`Self::with_batch_size`] does not apply here. Every
    /// chunk's logits are kept until the end, so memory grows with the total
    /// number of chunks (a few floats each) — this is for many documents of
    /// ordinary length, and an unbounded one belongs in
    /// [`Self::classify_stream`].
    ///
    /// ```no_run
    /// # use burn_setfit::Classifier;
    /// # fn main() -> burn_setfit::Result<()> {
    /// # let classifier = Classifier::from_bundle(&[], Default::default())?;
    /// let cells = ["Refund please", "Where is my parcel?", "Love the new app"];
    /// for (cell, prediction) in cells.iter().zip(classifier.classify_many(&cells)?) {
    ///     println!("{cell}: {:?}", prediction.labels(classifier.manifest()));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn classify_many(&self, texts: &[&str]) -> Result<Vec<Prediction>> {
        blocking("classifying documents", self.classify_many_async(texts))
    }

    /// [`Self::classify_many`], awaiting the device instead of blocking on it.
    pub async fn classify_many_async(&self, texts: &[&str]) -> Result<Vec<Prediction>> {
        // Chunk everything first, remembering whose each chunk is. Documents
        // are chunked in order, so each one's chunks stay in document order.
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut owner: Vec<usize> = Vec::new();
        for (doc, text) in texts.iter().enumerate() {
            let source = core::iter::once((*text).to_string());
            for chunk in Chunker::new(source, &self.tokenizer, self.cfg().chunk) {
                chunks.push(chunk?);
                owner.push(doc);
            }
        }

        let lengths: Vec<usize> = chunks.iter().map(|c| c.ids.len()).collect();
        let mut rows: Vec<Vec<f32>> = vec![Vec::new(); chunks.len()];
        for pass in plan_passes(&lengths, MANY_MAX_ROWS, MANY_MAX_TOKENS) {
            let sequences: Vec<Vec<u32>> = pass.iter().map(|&i| chunks[i].ids.clone()).collect();
            let longest = pass.iter().map(|&i| lengths[i]).max().unwrap_or(0);
            let pad_len = bucket_len(longest).min(self.cfg().chunk.max_tokens);
            let pass_rows = self.forward_sequences(&sequences, pad_len).await?;
            for (&i, row) in pass.iter().zip(pass_rows) {
                rows[i] = row;
            }
        }

        // Fold in chunk order, which is each document's own order, so every
        // accumulator sees exactly the sequence `classify` would feed it.
        let mut folds: Vec<DocFold> = texts.iter().map(|_| self.new_fold()).collect();
        for ((chunk, row), &doc) in chunks.iter().zip(&rows).zip(&owner) {
            self.fold_row(&mut folds[doc], chunk, row);
        }
        Ok(folds.into_iter().map(|f| self.finish_fold(f)).collect())
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
    /// # use burn_setfit::Classifier;
    /// # fn main() -> burn_setfit::Result<()> {
    /// # let classifier = Classifier::from_bundle(&[], Default::default())?;
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
        self.forward_sequences(&sequences, 0).await
    }

    /// [`Self::forward_rows`] on token sequences, padded to at least `pad_len`.
    async fn forward_sequences(
        &self,
        sequences: &[Vec<u32>],
        pad_len: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let (input_ids, attention_mask) = pad_batch_to(
            sequences,
            self.tokenizer.special_tokens().pad,
            pad_len,
            &self.device,
        );

        let logits = self.module.forward(input_ids, attention_mask);
        let data = readback::floats(logits).await?;

        let num_labels = self.cfg().num_labels();
        Ok(data.chunks(num_labels).map(<[f32]>::to_vec).collect())
    }

    /// Encode one batch of chunks and fold their scores in.
    async fn run_batch(&self, batch: &[Chunk], fold: &mut DocFold) -> Result<()> {
        let rows = self.forward_rows(batch).await?;
        for (chunk, row) in batch.iter().zip(&rows) {
            self.fold_row(fold, chunk, row);
        }
        Ok(())
    }

    /// An empty fold for one document.
    fn new_fold(&self) -> DocFold {
        let num_labels = self.cfg().num_labels();
        DocFold {
            accumulator: DocAccumulator::new(
                self.cfg().reducer,
                self.cfg().hierarchy_fanout,
                num_labels,
            ),
            evidence: vec![Vec::new(); num_labels],
            votes: vec![0; num_labels],
        }
    }

    /// Fold one chunk's logits into its document.
    fn fold_row(&self, fold: &mut DocFold, chunk: &Chunk, row: &[f32]) {
        fold.accumulator.push(chunk.token_count as f32, row);
        self.record_evidence(chunk, row, &mut fold.evidence);
        // The same decode `classify_chunks` reports per row, so a vote is
        // exactly a row whose `predicted` names the label.
        let scores = to_probabilities(row, self.cfg().task);
        for label in decode_against_background(&scores, self.cfg().task, self.cfg().background) {
            fold.votes[label] += 1;
        }
    }

    /// Decode a document once every chunk is folded in.
    fn finish_fold(&self, fold: DocFold) -> Prediction {
        let DocFold {
            accumulator,
            evidence,
            votes,
        } = fold;
        let logits = accumulator.finish();
        let scores = to_probabilities(&logits, self.cfg().task);
        let predicted = match self.cfg().chunk_votes {
            Some(min_chunks) => decide_by_votes(&votes, &scores, self.cfg().task, min_chunks),
            None => decode_against_background(&scores, self.cfg().task, self.cfg().background),
        };

        // Keep only the evidence for labels the document actually got.
        let mut evidence: Vec<Evidence> = predicted
            .iter()
            .flat_map(|&l| evidence.get(l).cloned().unwrap_or_default())
            .collect();
        evidence.sort_by(|a, b| b.score.total_cmp(&a.score));
        evidence.truncate(self.evidence_limit);

        Prediction {
            scores,
            predicted,
            chunks_seen: accumulator.count(),
            evidence,
        }
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
    /// # use burn_setfit::Classifier;
    /// # fn main() -> burn_setfit::Result<()> {
    /// # let classifier = Classifier::from_bundle(&[], Default::default())?;
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

        let (input_ids, attention_mask) = pad_batch(
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

/// What one document's chunks fold into, until it is decoded.
struct DocFold {
    accumulator: DocAccumulator,
    /// Tracked per label, because which label matters is not known until the
    /// document is fully reduced. Bounded at limit x labels either way.
    evidence: Vec<Vec<Evidence>>,
    /// Chunks that predict each label on their own, for `chunk_votes`.
    votes: Vec<usize>,
}

/// [`LENGTH_BUCKET`]-rounded length of a pass whose longest sequence is
/// `longest` tokens.
fn bucket_len(longest: usize) -> usize {
    longest.max(1).div_ceil(LENGTH_BUCKET) * LENGTH_BUCKET
}

/// Group sequence indices into forward passes, shortest first, each holding at
/// most `max_rows` sequences and at most `max_tokens` padded tokens (rows × the
/// pass's [`bucket_len`]). A sequence too long to share a pass gets its own.
///
/// Sorting is what keeps padding down: one long window padded together with
/// many short ones makes every row cost what the long one does.
fn plan_passes(lengths: &[usize], max_rows: usize, max_tokens: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    // Stable, so equal lengths keep their order and a run is reproducible.
    order.sort_by_key(|&i| lengths[i]);
    let mut passes: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for i in order {
        // Sorted, so this sequence is the longest the pass would hold.
        let padded = (current.len() + 1) * bucket_len(lengths[i]);
        if !current.is_empty() && (current.len() == max_rows.max(1) || padded > max_tokens) {
            passes.push(core::mem::take(&mut current));
        }
        current.push(i);
    }
    if !current.is_empty() {
        passes.push(current);
    }
    passes
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
///   background. Note the comparison is made on reduced scores; on a long
///   document the background's best chunk is high whatever else the document
///   says, so this rarely helps there. Deciding each chunk on its own does
///   ([`ClassifierConfig::with_chunk_votes`], [`decide_by_votes`]; see the
///   README's long-document section).
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

/// Labels predicted by enough chunks on their own.
///
/// `votes[i]` is how many chunks decoded to label `i` by themselves (the
/// background class never does, so it never wins here either). Multi-label
/// predicts every label with at least `min_chunks` votes. Single-label predicts
/// the most-voted label if it has enough, breaking ties on the reduced `scores`.
///
/// ```
/// use burn_setfit::TaskMode;
/// use burn_setfit::infer::decide_by_votes;
/// use std::num::NonZeroUsize;
///
/// let votes = [2, 0, 5];
/// let scores = [0.3, 0.1, 0.2];
/// let two = NonZeroUsize::new(2).unwrap();
///
/// // Two passages naming a label is enough, however low its reduced score.
/// assert_eq!(decide_by_votes(&votes, &scores, TaskMode::multi_label(), two), vec![0, 2]);
///
/// // Single-label takes the most votes, not the highest score.
/// assert_eq!(decide_by_votes(&votes, &scores, TaskMode::SingleLabel, two), vec![2]);
///
/// // And abstains when no label has enough.
/// let six = NonZeroUsize::new(6).unwrap();
/// assert!(decide_by_votes(&votes, &scores, TaskMode::SingleLabel, six).is_empty());
/// ```
pub fn decide_by_votes(
    votes: &[usize],
    scores: &[f32],
    task: TaskMode,
    min_chunks: NonZeroUsize,
) -> Vec<usize> {
    let enough = |i: &usize| votes[*i] >= min_chunks.get();
    match task {
        TaskMode::SingleLabel => (0..votes.len())
            .filter(enough)
            .max_by(|&a, &b| {
                votes[a].cmp(&votes[b]).then_with(|| {
                    let score = |i: usize| scores.get(i).copied().unwrap_or(f32::NEG_INFINITY);
                    score(a).total_cmp(&score(b))
                })
            })
            .map(|i| vec![i])
            .unwrap_or_default(),
        TaskMode::MultiLabel { .. } => (0..votes.len()).filter(enough).collect(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_len_rounds_up_to_the_bucket() {
        assert_eq!(bucket_len(0), LENGTH_BUCKET);
        assert_eq!(bucket_len(1), 8);
        assert_eq!(bucket_len(16), 16);
        assert_eq!(bucket_len(17), 24);
    }

    #[test]
    fn passes_are_planned_shortest_first_and_cover_everything_once() {
        let lengths = [30, 5, 200, 5, 12, 90, 7];
        let passes = plan_passes(&lengths, 3, usize::MAX);
        let flat: Vec<usize> = passes.iter().flatten().copied().collect();
        let mut sorted = flat.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..lengths.len()).collect::<Vec<_>>());
        assert!(flat.windows(2).all(|w| lengths[w[0]] <= lengths[w[1]]));
        assert!(passes.iter().all(|p| p.len() <= 3));
    }

    #[test]
    fn passes_are_bounded_by_padded_tokens() {
        let lengths = [256; 40];
        for pass in plan_passes(&lengths, 64, 4096) {
            assert!(pass.len() * 256 <= 4096);
        }
    }

    #[test]
    fn a_sequence_too_long_for_the_budget_gets_a_pass_of_its_own() {
        assert_eq!(plan_passes(&[256], 64, 100), vec![vec![0]]);
        assert_eq!(plan_passes(&[8, 256], 64, 100), vec![vec![0], vec![1]]);
    }

    #[test]
    fn a_zero_row_cap_still_makes_progress() {
        assert_eq!(plan_passes(&[5, 5], 0, usize::MAX), vec![vec![0], vec![1]]);
    }
}
