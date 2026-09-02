//! What a trained model does with a document, as one validated value.
//!
//! Labels, task mode, windowing and reduction are not independent knobs: a reducer
//! suits a task, a threshold only means something for one of them, and an overlap
//! wider than the window is not a preference but a mistake. Gathering them into one
//! type gives a single place to state those relationships and a single place to
//! check them.
//!
//! Setters chain and never fail. Validation happens once, at every entry point that
//! consumes a config — [`crate::bundle::Bundle::pack`], [`crate::infer::Classifier`]
//! and the trainer all call [`ClassifierConfig::validate`] — so an invalid
//! configuration is reported where it is used rather than propagating into a model
//! that merely behaves oddly.

use crate::chunk::ChunkConfig;
use crate::error::{Result, SetFitError};
use crate::head::TaskMode;
use crate::reduce::Reducer;

/// How a trained model classifies.
///
/// ```
/// use burn_setfit::{ChunkConfig, ClassifierConfig, Reducer};
///
/// let config = ClassifierConfig::new(["billing", "outage", "feature"])
///     .multi_label_at(0.4)                              // threshold lives in the task
///     .with_reducer(Reducer::MaxLogits)                 // else the task's default
///     .with_chunking(ChunkConfig::new(256).with_overlap(32))
///     .with_hierarchy(8);
///
/// assert_eq!(config.num_labels(), 3);
/// assert_eq!(config.threshold(), Some(0.4));
/// assert_eq!(config.label_index("outage"), Some(1));
/// config.validate()?;
/// # Ok::<(), burn_setfit::SetFitError>(())
/// ```
///
/// Mistakes are refused rather than clamped, and the message names the culprit:
///
/// ```
/// use burn_setfit::ClassifierConfig;
///
/// let err = ClassifierConfig::new(["spam", "ham", "spam"])
///     .validate()
///     .expect_err("two labels of the same name are ambiguous");
/// assert!(format!("{err}").contains("spam"));
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClassifierConfig {
    /// Class names, in the order of the head's output columns.
    pub labels: Vec<String>,
    /// Single- or multi-label, and the latter's decision threshold.
    pub task: TaskMode,
    /// How documents too long for the encoder are windowed.
    pub chunk: ChunkConfig,
    /// How per-chunk scores collapse into a document verdict.
    pub reducer: Reducer,
    /// Reduce chunks in blocks of this size, recursively, rather than all at once.
    pub hierarchy_fanout: Option<usize>,
    /// The label meaning "none of the above", if there is one.
    ///
    /// An index into [`Self::labels`], because the background class is trained
    /// exactly like every other class — what changes is how it is *decoded*.
    /// See [`Self::with_background_class`].
    #[serde(default)]
    pub background: Option<usize>,
}

impl ClassifierConfig {
    /// A single-label classifier over these labels, with defaults for the rest.
    ///
    /// The reducer follows the task mode, so switching to multi-label with
    /// [`Self::multi_label`] also switches the reducer — unless
    /// [`Self::with_reducer`] has pinned one.
    pub fn new<S: Into<String>>(labels: impl IntoIterator<Item = S>) -> Self {
        let task = TaskMode::SingleLabel;
        Self {
            labels: labels.into_iter().map(Into::into).collect(),
            task,
            chunk: ChunkConfig::default(),
            reducer: Reducer::default_for(task),
            hierarchy_fanout: None,
            background: None,
        }
    }

    /// Switch to multi-label at the conventional 0.5 threshold.
    pub fn multi_label(self) -> Self {
        self.with_task(TaskMode::multi_label())
    }

    /// Switch to multi-label at a specific threshold.
    pub fn multi_label_at(self, threshold: f32) -> Self {
        self.with_task(TaskMode::MultiLabel { threshold })
    }

    /// Set the task mode, moving the reducer to that mode's default.
    ///
    /// Changing the task without changing the reducer is nearly always a mistake:
    /// a mean over chunks says "the document as a whole is about this", which is
    /// not what a multi-label question asks. Call [`Self::with_reducer`] afterwards
    /// to override.
    ///
    /// ```
    /// use burn_setfit::{ClassifierConfig, Reducer};
    ///
    /// let single = ClassifierConfig::new(["a", "b"]);
    /// assert_eq!(single.reducer, Reducer::MeanLogits);
    ///
    /// // Switching the task moves the reducer with it...
    /// let multi = single.multi_label();
    /// assert_eq!(multi.reducer, Reducer::MaxLogits);
    ///
    /// // ...unless you pin one afterwards.
    /// let pinned = multi.with_reducer(Reducer::TopKMeanLogits { k: 3 });
    /// assert_eq!(pinned.reducer, Reducer::TopKMeanLogits { k: 3 });
    /// ```
    pub fn with_task(mut self, task: TaskMode) -> Self {
        self.task = task;
        self.reducer = Reducer::default_for(task);
        self
    }

    /// Pin the reducer, overriding the task's default.
    pub fn with_reducer(mut self, reducer: Reducer) -> Self {
        self.reducer = reducer;
        self
    }

    /// Set how documents are windowed.
    pub fn with_chunking(mut self, chunk: ChunkConfig) -> Self {
        self.chunk = chunk;
        self
    }

    /// Reduce chunks in blocks of `fanout`, recursively, rather than all at once.
    pub fn with_hierarchy(mut self, fanout: usize) -> Self {
        self.hierarchy_fanout = Some(fanout);
        self
    }

    /// Name one of the labels as "none of the above".
    ///
    /// The class is trained like any other — it has to be, since something must
    /// teach the model what filler looks like — but it stops being a class the
    /// model can *predict*. Instead it becomes the bar every other label has to
    /// clear.
    ///
    /// This is aimed at the failure measured in
    /// [#4](https://github.com/jcorrie/burn-setfit/issues/4): over a long
    /// document, a configuration can separate signal from filler well and still
    /// decide wrongly, because the absolute scores sit on the wrong side of a
    /// fixed threshold. Comparing against a background class asks the question
    /// the separation actually answers — "is this label stronger than nothing in
    /// particular?" — instead of "is this label above 0.5?".
    ///
    /// It also gives a single-label head somewhere to abstain *to*. A softmax
    /// over real classes cannot say "none"; a softmax whose argmax lands on the
    /// background class can.
    ///
    /// ```
    /// use burn_setfit::ClassifierConfig;
    ///
    /// let config = ClassifierConfig::new(["billing", "outage", "other"])
    ///     .multi_label()
    ///     .with_background_class("other")
    ///     .expect("`other` is one of the labels");
    ///
    /// assert_eq!(config.background, Some(2));
    /// ```
    ///
    /// Naming a label that does not exist is an error rather than a silent
    /// no-op, because the mistake is invisible at every later point:
    ///
    /// ```
    /// use burn_setfit::ClassifierConfig;
    ///
    /// let err = ClassifierConfig::new(["billing", "outage"])
    ///     .with_background_class("other")
    ///     .expect_err("there is no `other` to be the background");
    /// assert!(format!("{err}").contains("other"));
    /// ```
    pub fn with_background_class(mut self, name: &str) -> Result<Self> {
        match self.label_index(name) {
            Some(i) => {
                self.background = Some(i);
                Ok(self)
            }
            None => Err(SetFitError::Config(format!(
                "no label named {name:?} to use as the background class; labels are {:?}",
                self.labels
            ))),
        }
    }

    /// The background label's name, if one is set.
    pub fn background_label(&self) -> Option<&str> {
        self.background.map(|i| self.labels[i].as_str())
    }

    /// Number of classes.
    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    /// The decision threshold, or `None` for single-label.
    ///
    /// ```
    /// use burn_setfit::ClassifierConfig;
    ///
    /// // An argmax has nothing to threshold, so there is nothing to report.
    /// assert_eq!(ClassifierConfig::new(["a", "b"]).threshold(), None);
    /// assert_eq!(ClassifierConfig::new(["a", "b"]).multi_label().threshold(), Some(0.5));
    /// ```
    pub fn threshold(&self) -> Option<f32> {
        match self.task {
            TaskMode::SingleLabel => None,
            TaskMode::MultiLabel { threshold } => Some(threshold),
        }
    }

    /// Index of a label by name.
    pub fn label_index(&self, name: &str) -> Option<usize> {
        self.labels.iter().position(|l| l == name)
    }

    /// Check every internal relationship, and report the first that fails.
    pub fn validate(&self) -> Result<()> {
        if self.labels.len() < 2 {
            return Err(SetFitError::Config(format!(
                "a classifier needs at least two labels, got {}",
                self.labels.len()
            )));
        }
        if let Some(blank) = self.labels.iter().position(|l| l.trim().is_empty()) {
            return Err(SetFitError::Config(format!(
                "label {blank} is empty; labels name classes and appear in output"
            )));
        }
        // Duplicates would make `label_index` and any name-keyed result ambiguous,
        // and two columns of the head would be trained to mean the same thing.
        for (i, label) in self.labels.iter().enumerate() {
            if let Some(j) = self.labels[..i].iter().position(|l| l == label) {
                return Err(SetFitError::Config(format!(
                    "labels {j} and {i} are both {label:?}"
                )));
            }
        }

        self.task.validate()?;
        self.chunk.validate()?;
        self.reducer.validate()?;

        if let Some(background) = self.background {
            if background >= self.labels.len() {
                return Err(SetFitError::Config(format!(
                    "background class is label {background}, but there are only {} labels",
                    self.labels.len()
                )));
            }
            // With one real class left, "stronger than background" is the whole
            // decision and the classifier has nothing to choose between.
            if self.labels.len() < 3 {
                return Err(SetFitError::Config(format!(
                    "a background class needs at least two other labels to sit against,                      but {:?} leaves only {}",
                    self.labels,
                    self.labels.len() - 1
                )));
            }
        }

        if let Some(fanout) = self.hierarchy_fanout
            && fanout < 2
        {
            return Err(SetFitError::Config(format!(
                "hierarchy fanout must be at least 2, got {fanout}; \
                 a block of one reduces to itself and the hierarchy never terminates"
            )));
        }

        Ok(())
    }
}
