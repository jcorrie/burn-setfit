//! Reducing per-chunk scores into a document verdict.
//!
//! One thing is worth knowing before choosing a reducer: **with a linear head,
//! averaging chunk embeddings and averaging chunk logits are the same operation.**
//! A linear map commutes with a weighted mean, so "pool the embeddings, classify
//! once" and "classify every chunk, average the scores" are not two strategies —
//! they are one, and it is [`Reducer::MeanLogits`]. Everything genuinely new here
//! is nonlinear.
//!
//! That matters most for multi-label work. A document carries a label when *some*
//! passage carries it, and a mean over hundreds of chunks washes a single relevant
//! passage away to nothing. [`Reducer::NoisyOr`] and [`Reducer::MaxLogits`] encode
//! "any chunk suffices" and are the right default there.
//!
//! Every reducer is an online fold in logit space, so a document of unbounded
//! length reduces in constant memory, and probabilities are formed exactly once
//! at the end.

use crate::head::TaskMode;

/// How per-chunk logits collapse into one document-level score vector.
///
/// Every variant is an online fold, so a document of any length reduces in
/// bounded memory. The choice matters most when one passage in many carries the
/// label:
///
/// ```
/// use burn_setfit::Reducer;
///
/// // Three chunks, two labels. Only the middle chunk is about label 1.
/// let chunks = [[2.0, -3.0], [-1.0, 4.0], [1.5, -2.5]];
///
/// let mut mean = Reducer::MeanLogits.accumulator(2);
/// let mut max = Reducer::MaxLogits.accumulator(2);
/// for chunk in &chunks {
///     // The weight is the chunk's token count; only the mean consults it.
///     mean.push(100.0, chunk);
///     max.push(100.0, chunk);
/// }
///
/// // The mean washes the one relevant passage away; the max keeps it.
/// assert!(mean.finish()[1] < 0.0);
/// assert_eq!(max.finish()[1], 4.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Reducer {
    /// Token-count-weighted mean of logits. Equivalent to mean-pooling the chunk
    /// embeddings and classifying once. The right default when the label is a
    /// property of the document as a whole.
    MeanLogits,
    /// Per-class maximum. "Any chunk suffices", in its bluntest form.
    ///
    /// Monotone under sigmoid, so this is identical to taking the maximum
    /// probability — no need for a separate prob-space variant.
    MaxLogits,
    /// Per-class noisy-OR: `1 - Π(1 - pᵢ)`.
    ///
    /// Several moderately confident chunks combine into a confident document.
    ///
    /// **Saturates on long documents.** The combination is multiplicative, so it
    /// only behaves when irrelevant chunks score near zero. At a realistic `p ≈ 0.6`
    /// for background text, eight chunks already reach `1 - 0.4⁸ ≈ 0.999` and every
    /// label fires — measured, not hypothetical, which is why this is no longer the
    /// multi-label default. Reserve it for few chunks, or for a head calibrated with
    /// a background class so filler genuinely scores low. Reducing hierarchically
    /// does not rescue it: the saturation happens within the first block.
    NoisyOr,
    /// Mean of the `k` highest logits per class.
    ///
    /// The middle ground: robust to a single spurious chunk, without letting
    /// hundreds of irrelevant ones dilute a real signal.
    TopKMeanLogits {
        /// How many chunks vote per class.
        k: usize,
    },
    /// Smooth interpolation between mean (high temperature) and max (low).
    LogSumExp {
        /// Softness. Approaches [`Reducer::MeanLogits`] as it grows and
        /// [`Reducer::MaxLogits`] as it approaches zero.
        temperature: f32,
    },
}

impl Reducer {
    /// The reducer that suits a task mode when the caller has no opinion.
    ///
    /// ```
    /// use burn_setfit::{Reducer, TaskMode};
    ///
    /// // "The label describes the whole document" -- every chunk votes.
    /// assert_eq!(Reducer::default_for(TaskMode::SingleLabel), Reducer::MeanLogits);
    /// // "The label describes something the document contains" -- one chunk can
    /// // carry it alone. Not NoisyOr: that saturates once a document is long.
    /// assert_eq!(Reducer::default_for(TaskMode::multi_label()), Reducer::MaxLogits);
    /// ```
    pub fn default_for(task: TaskMode) -> Self {
        match task {
            // The label describes the whole document, so every chunk votes.
            TaskMode::SingleLabel => Reducer::MeanLogits,
            // The label describes something the document contains, so one chunk
            // can carry it alone — but combining evidence multiplicatively
            // saturates once a document has many chunks, so take the strongest
            // chunk rather than accumulating across all of them.
            TaskMode::MultiLabel { .. } => Reducer::MaxLogits,
        }
    }

    /// Reject parameters the fold cannot use.
    ///
    /// Both cases were previously clamped at the point of use, which turned a
    /// configuration mistake into a model that quietly did something else.
    ///
    /// ```
    /// use burn_setfit::Reducer;
    ///
    /// assert!(Reducer::TopKMeanLogits { k: 3 }.validate().is_ok());
    /// // k = 0 would average no chunks at all.
    /// assert!(Reducer::TopKMeanLogits { k: 0 }.validate().is_err());
    /// assert!(Reducer::LogSumExp { temperature: 0.0 }.validate().is_err());
    /// ```
    pub fn validate(&self) -> crate::Result<()> {
        match self {
            Reducer::TopKMeanLogits { k } if *k == 0 => Err(crate::SetFitError::Config(
                "TopKMeanLogits needs k >= 1; k = 0 would average no chunks at all".into(),
            )),
            // Written as a positive test so a NaN temperature is rejected too.
            Reducer::LogSumExp { temperature }
                if !matches!(
                    temperature.partial_cmp(&0.0),
                    Some(core::cmp::Ordering::Greater)
                ) =>
            {
                Err(crate::SetFitError::Config(format!(
                    "LogSumExp temperature must be positive, got {temperature}"
                )))
            }
            _ => Ok(()),
        }
    }

    /// Start folding chunk scores for a document.
    pub fn accumulator(&self, num_labels: usize) -> Accumulator {
        Accumulator::new(*self, num_labels)
    }
}

/// `ln(1 + eˣ)`, computed without overflowing for large `x`.
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Online fold of chunk logits into document logits.
///
/// Memory is `O(num_labels)`, or `O(k · num_labels)` for
/// [`Reducer::TopKMeanLogits`] — bounded either way, regardless of document size.
///
/// ```
/// use burn_setfit::Reducer;
///
/// let mut acc = Reducer::MeanLogits.accumulator(2);
/// assert_eq!(acc.count(), 0);
/// // An empty document reduces to zeros -- check `count`, do not read meaning
/// // into the scores.
/// assert_eq!(acc.finish(), vec![0.0, 0.0]);
///
/// // The mean is weighted by token count, so a half-full trailing chunk does
/// // not vote as loudly as a full one.
/// acc.push(300.0, &[1.0, 0.0]);
/// acc.push(100.0, &[-1.0, 0.0]);
/// assert_eq!(acc.count(), 2);
/// assert_eq!(acc.finish()[0], 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct Accumulator {
    reducer: Reducer,
    num_labels: usize,
    count: usize,
    total_weight: f32,
    state: State,
}

#[derive(Debug, Clone)]
enum State {
    /// Running weighted sum of logits.
    WeightedSum(Vec<f32>),
    /// Running per-class maximum.
    Max(Vec<f32>),
    /// Running `Σ ln(1 - pᵢ)` per class, which stays well-conditioned where a
    /// running product of `(1 - pᵢ)` would underflow to zero.
    LogComplement(Vec<f32>),
    /// Up to `k` largest logits per class, ascending.
    TopK(Vec<Vec<f32>>),
    /// Online log-sum-exp: running max, and the sum rescaled against it.
    LogSumExp { max: Vec<f32>, sum: Vec<f32> },
}

impl Accumulator {
    fn new(reducer: Reducer, num_labels: usize) -> Self {
        let state = match reducer {
            Reducer::MeanLogits => State::WeightedSum(vec![0.0; num_labels]),
            Reducer::MaxLogits => State::Max(vec![f32::NEG_INFINITY; num_labels]),
            Reducer::NoisyOr => State::LogComplement(vec![0.0; num_labels]),
            Reducer::TopKMeanLogits { .. } => State::TopK(vec![Vec::new(); num_labels]),
            Reducer::LogSumExp { .. } => State::LogSumExp {
                max: vec![f32::NEG_INFINITY; num_labels],
                sum: vec![0.0; num_labels],
            },
        };
        Self {
            reducer,
            num_labels,
            count: 0,
            total_weight: 0.0,
            state,
        }
    }

    /// Fold in one chunk's logits.
    ///
    /// `weight` is the chunk's token count; only [`Reducer::MeanLogits`] consults
    /// it, so that a half-full trailing chunk does not vote as loudly as a full one.
    pub fn push(&mut self, weight: f32, logits: &[f32]) {
        debug_assert_eq!(logits.len(), self.num_labels);
        let weight = weight.max(f32::MIN_POSITIVE);
        self.count += 1;
        self.total_weight += weight;

        match (&mut self.state, self.reducer) {
            (State::WeightedSum(sums), _) => {
                for (s, &l) in sums.iter_mut().zip(logits) {
                    *s += weight * l;
                }
            }
            (State::Max(best), _) => {
                for (b, &l) in best.iter_mut().zip(logits) {
                    if l > *b {
                        *b = l;
                    }
                }
            }
            (State::LogComplement(sums), _) => {
                for (s, &l) in sums.iter_mut().zip(logits) {
                    // ln(1 - sigmoid(l)) == -softplus(l), exactly and stably.
                    *s -= softplus(l);
                }
            }
            (State::TopK(per_label), Reducer::TopKMeanLogits { k }) => {
                let k = k.max(1);
                for (heap, &l) in per_label.iter_mut().zip(logits) {
                    if heap.len() < k {
                        let pos = heap.partition_point(|&x| x < l);
                        heap.insert(pos, l);
                    } else if l > heap[0] {
                        heap.remove(0);
                        let pos = heap.partition_point(|&x| x < l);
                        heap.insert(pos, l);
                    }
                }
            }
            (State::LogSumExp { max, sum }, Reducer::LogSumExp { temperature }) => {
                let t = temperature.max(1e-6);
                for ((m, s), &l) in max.iter_mut().zip(sum.iter_mut()).zip(logits) {
                    let x = l / t;
                    if x > *m {
                        // Rescale the running sum against the new maximum.
                        *s = if m.is_finite() {
                            *s * (*m - x).exp()
                        } else {
                            0.0
                        };
                        *m = x;
                    }
                    *s += (x - *m).exp();
                }
            }
            _ => unreachable!("accumulator state always matches its reducer"),
        }
    }

    /// Chunks folded in so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Summed weight of the chunks folded in so far.
    pub fn total_weight(&self) -> f32 {
        self.total_weight
    }

    /// Collapse to document-level logits.
    ///
    /// Returns all-zero logits for an empty document, which decodes to a uniform
    /// distribution or to no labels at all — callers should check [`Self::count`]
    /// rather than read meaning into that.
    pub fn finish(&self) -> Vec<f32> {
        if self.count == 0 {
            return vec![0.0; self.num_labels];
        }

        match (&self.state, self.reducer) {
            (State::WeightedSum(sums), _) => sums.iter().map(|s| s / self.total_weight).collect(),
            (State::Max(best), _) => best.clone(),
            (State::LogComplement(sums), _) => sums
                .iter()
                .map(|&s| {
                    // q = 1 - eˢ, and we want ln(q / (1 - q)) = ln(1 - eˢ) - s.
                    // `expm1` keeps the left term accurate as s approaches 0,
                    // where `1 - exp(s)` would cancel catastrophically.
                    let log_q = (-s.exp_m1()).max(f32::MIN_POSITIVE).ln();
                    log_q - s
                })
                .collect(),
            (State::TopK(per_label), _) => per_label
                .iter()
                .map(|heap| {
                    if heap.is_empty() {
                        0.0
                    } else {
                        heap.iter().sum::<f32>() / heap.len() as f32
                    }
                })
                .collect(),
            (State::LogSumExp { max, sum }, Reducer::LogSumExp { temperature }) => {
                let t = temperature.max(1e-6);
                let n = self.count as f32;
                // Mean form rather than plain log-sum-exp, so the result stays on
                // the same scale as the logits instead of growing with chunk count.
                max.iter()
                    .zip(sum)
                    .map(|(m, s)| t * (m + (s / n).ln()))
                    .collect()
            }
            _ => unreachable!("accumulator state always matches its reducer"),
        }
    }
}

/// A recursive fold: chunks into blocks, blocks into super-blocks, and so on.
///
/// Chunks are reduced in groups of `fanout`; each group's result is promoted and
/// folded at the next level up, with levels created on demand. A document of any
/// size collapses through `log_fanout(n)` levels, all of them bounded.
///
/// Because a promoted block carries its summed weight upward, [`Reducer::MeanLogits`]
/// gives exactly the same answer hierarchically as it does flat. The hierarchy
/// changes the outcome only for the nonlinear reducers — where it is the point:
/// noisy-OR applied within a section and then across sections says "some section
/// is about this", which is a different and usually better claim than "some chunk
/// mentioned it".
#[derive(Debug, Clone)]
pub struct HierarchicalAccumulator {
    fanout: usize,
    within: Reducer,
    across: Reducer,
    num_labels: usize,
    levels: Vec<Accumulator>,
    count: usize,
}

impl HierarchicalAccumulator {
    /// `within` reduces chunks inside a block; `across` reduces block results.
    pub fn new(fanout: usize, within: Reducer, across: Reducer, num_labels: usize) -> Self {
        Self {
            fanout: fanout.max(2),
            within,
            across,
            num_labels,
            levels: vec![within.accumulator(num_labels)],
            count: 0,
        }
    }

    /// Fold in one chunk, cascading any levels that just filled up.
    pub fn push(&mut self, weight: f32, logits: &[f32]) {
        self.count += 1;
        self.levels[0].push(weight, logits);
        self.cascade(0);
    }

    /// Promote every full level into the one above it.
    fn cascade(&mut self, from: usize) {
        let mut level = from;
        while self.levels[level].count() >= self.fanout {
            let logits = self.levels[level].finish();
            let weight = self.levels[level].total_weight();

            // Levels above the leaves aggregate blocks, not chunks.
            let reducer = if level == 0 { self.within } else { self.across };
            self.levels[level] = reducer.accumulator(self.num_labels);

            if level + 1 == self.levels.len() {
                self.levels.push(self.across.accumulator(self.num_labels));
            }
            self.levels[level + 1].push(weight, &logits);
            level += 1;
        }
    }

    /// Chunks folded in so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Collapse every level, bottom up, into document-level logits.
    pub fn finish(&self) -> Vec<f32> {
        if self.count == 0 {
            return vec![0.0; self.num_labels];
        }

        let mut carried: Option<(f32, Vec<f32>)> = None;
        for (level, acc) in self.levels.iter().enumerate() {
            let mut acc = acc.clone();
            if let Some((weight, logits)) = carried.take() {
                acc.push(weight, &logits);
            }
            if acc.count() == 0 {
                continue;
            }
            // The topmost non-empty level is the answer.
            if level + 1 == self.levels.len() {
                return acc.finish();
            }
            carried = Some((acc.total_weight(), acc.finish()));
        }

        carried
            .map(|(_, l)| l)
            .unwrap_or_else(|| vec![0.0; self.num_labels])
    }
}

/// Either reduction strategy, behind one interface.
///
/// Lets a caller switch between a flat fold and a recursive one from configuration
/// without the call site caring which it got.
#[derive(Debug, Clone)]
pub enum DocAccumulator {
    /// Every chunk votes directly into the document score.
    Flat(Accumulator),
    /// Chunks fold into blocks, blocks into the document.
    Hierarchical(HierarchicalAccumulator),
}

impl DocAccumulator {
    /// Build from a reducer, recursing in blocks of `fanout` when one is given.
    pub fn new(reducer: Reducer, fanout: Option<usize>, num_labels: usize) -> Self {
        match fanout {
            // Blocks reduce with the chosen reducer; block results are averaged,
            // so no single block can dominate the document on its own.
            Some(f) => DocAccumulator::Hierarchical(HierarchicalAccumulator::new(
                f,
                reducer,
                Reducer::MeanLogits,
                num_labels,
            )),
            None => DocAccumulator::Flat(reducer.accumulator(num_labels)),
        }
    }

    /// Fold in one chunk's logits.
    pub fn push(&mut self, weight: f32, logits: &[f32]) {
        match self {
            DocAccumulator::Flat(a) => a.push(weight, logits),
            DocAccumulator::Hierarchical(a) => a.push(weight, logits),
        }
    }

    /// Chunks folded in so far.
    pub fn count(&self) -> usize {
        match self {
            DocAccumulator::Flat(a) => a.count(),
            DocAccumulator::Hierarchical(a) => a.count(),
        }
    }

    /// Collapse to document-level logits.
    pub fn finish(&self) -> Vec<f32> {
        match self {
            DocAccumulator::Flat(a) => a.finish(),
            DocAccumulator::Hierarchical(a) => a.finish(),
        }
    }
}
