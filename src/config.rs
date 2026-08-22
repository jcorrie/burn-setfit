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

    /// Number of classes.
    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    /// The decision threshold, or `None` for single-label.
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
