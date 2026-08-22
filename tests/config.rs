//! Configuration: builder semantics, and every way a config can be wrong.
//!
//! Validation is only worth having if it is reached, so these check both that a
//! bad value is rejected and that the message says which one.

mod common;

use burn_setfit::chunk::ChunkConfig;
use burn_setfit::config::ClassifierConfig;
use burn_setfit::head::TaskMode;
use burn_setfit::reduce::Reducer;
use burn_setfit::train::TrainConfig;

/// Assert a validation failure, and that its message names the culprit.
#[track_caller]
fn rejects(result: burn_setfit::Result<()>, mentioning: &str) {
    let err = result.expect_err("expected this configuration to be rejected");
    let text = format!("{err}");
    assert!(
        text.to_lowercase().contains(&mentioning.to_lowercase()),
        "error should mention {mentioning:?}, said: {text}"
    );
}

// ── labels ──────────────────────────────────────────────────────────────────

#[test]
fn a_valid_config_passes() {
    ClassifierConfig::new(["a", "b", "c"])
        .validate()
        .expect("a plain three-label config is valid");
}

#[test]
fn fewer_than_two_labels_is_not_a_classification_task() {
    rejects(ClassifierConfig::new(["only"]).validate(), "two labels");
    rejects(
        ClassifierConfig::new(Vec::<String>::new()).validate(),
        "two labels",
    );
}

#[test]
fn empty_labels_are_rejected() {
    rejects(ClassifierConfig::new(["a", ""]).validate(), "empty");
    rejects(ClassifierConfig::new(["a", "   "]).validate(), "empty");
}

#[test]
fn duplicate_labels_are_rejected() {
    // Two head columns trained to mean the same thing, and an ambiguous lookup.
    rejects(
        ClassifierConfig::new(["spam", "ham", "spam"]).validate(),
        "spam",
    );
}

#[test]
fn labels_resolve_by_name_and_position() {
    let config = ClassifierConfig::new(["billing", "outage"]);
    assert_eq!(config.num_labels(), 2);
    assert_eq!(config.label_index("outage"), Some(1));
    assert_eq!(config.label_index("nonexistent"), None);
}

#[test]
fn accepts_labels_from_any_string_like_source() {
    let from_strs = ClassifierConfig::new(["a", "b"]);
    let from_strings = ClassifierConfig::new(vec!["a".to_string(), "b".to_string()]);
    assert_eq!(from_strs.labels, from_strings.labels);
}

// ── task mode ───────────────────────────────────────────────────────────────

#[test]
fn single_label_has_no_threshold_to_get_wrong() {
    let config = ClassifierConfig::new(["a", "b"]);
    assert_eq!(config.task, TaskMode::SingleLabel);
    assert_eq!(config.threshold(), None);
}

#[test]
fn multi_label_defaults_to_a_half() {
    let config = ClassifierConfig::new(["a", "b"]).multi_label();
    assert!(config.task.is_multi_label());
    assert_eq!(config.threshold(), Some(0.5));
}

#[test]
fn thresholds_outside_the_open_unit_interval_are_rejected() {
    for bad in [0.0, 1.0, -0.5, 1.5, f32::NAN, f32::INFINITY] {
        rejects(
            ClassifierConfig::new(["a", "b"])
                .multi_label_at(bad)
                .validate(),
            "threshold",
        );
    }
}

#[test]
fn thresholds_inside_it_are_accepted() {
    for good in [0.01, 0.5, 0.99] {
        ClassifierConfig::new(["a", "b"])
            .multi_label_at(good)
            .validate()
            .unwrap_or_else(|e| panic!("threshold {good} should be valid: {e}"));
    }
}

// ── reducer / task coupling ─────────────────────────────────────────────────

#[test]
fn switching_task_moves_the_reducer_with_it() {
    // A mean over chunks answers "is the document about this", which is not the
    // multi-label question — so the reducer must not be left behind.
    let single = ClassifierConfig::new(["a", "b"]);
    assert_eq!(single.reducer, Reducer::MeanLogits);

    let multi = single.multi_label();
    assert_eq!(multi.reducer, Reducer::MaxLogits);
}

#[test]
fn an_explicit_reducer_survives() {
    let config = ClassifierConfig::new(["a", "b"])
        .multi_label()
        .with_reducer(Reducer::NoisyOr);
    assert_eq!(config.reducer, Reducer::NoisyOr);
}

#[test]
fn but_changing_task_afterwards_resets_it() {
    // Documented behaviour: `with_task` is the coupling point, so ordering matters.
    let config = ClassifierConfig::new(["a", "b"])
        .with_reducer(Reducer::NoisyOr)
        .multi_label();
    assert_eq!(config.reducer, Reducer::MaxLogits);
}

#[test]
fn reducers_that_could_not_fold_anything_are_rejected() {
    rejects(
        ClassifierConfig::new(["a", "b"])
            .with_reducer(Reducer::TopKMeanLogits { k: 0 })
            .validate(),
        "k",
    );
    for bad in [0.0, -1.0, f32::NAN] {
        rejects(
            ClassifierConfig::new(["a", "b"])
                .with_reducer(Reducer::LogSumExp { temperature: bad })
                .validate(),
            "temperature",
        );
    }
}

#[test]
fn usable_reducer_parameters_pass() {
    for reducer in [
        Reducer::MeanLogits,
        Reducer::MaxLogits,
        Reducer::NoisyOr,
        Reducer::TopKMeanLogits { k: 1 },
        Reducer::LogSumExp { temperature: 1e-6 },
    ] {
        ClassifierConfig::new(["a", "b"])
            .with_reducer(reducer)
            .validate()
            .unwrap_or_else(|e| panic!("{reducer:?} should be valid: {e}"));
    }
}

// ── hierarchy ───────────────────────────────────────────────────────────────

#[test]
fn a_hierarchy_that_never_narrows_is_rejected() {
    // Blocks of one reduce to themselves, so the recursion makes no progress.
    for bad in [0, 1] {
        rejects(
            ClassifierConfig::new(["a", "b"])
                .with_hierarchy(bad)
                .validate(),
            "fanout",
        );
    }
    ClassifierConfig::new(["a", "b"])
        .with_hierarchy(2)
        .validate()
        .expect("blocks of two narrow");
}

// ── chunking ────────────────────────────────────────────────────────────────

#[test]
fn the_default_window_is_valid() {
    ChunkConfig::default()
        .validate()
        .expect("defaults are usable");
}

#[test]
fn a_window_with_no_room_for_content_is_rejected() {
    for bad in [0, 1, 2] {
        rejects(ChunkConfig::new(bad).validate(), "max_tokens");
    }
}

#[test]
fn overlap_at_or_beyond_the_window_would_never_advance() {
    let capacity = ChunkConfig::new(32).capacity();
    assert_eq!(capacity, 30);

    // Exactly a full window of carry-over means the next window re-reads it all.
    rejects(
        ChunkConfig::new(32).with_overlap(capacity).validate(),
        "overlap",
    );
    rejects(
        ChunkConfig::new(32).with_overlap(capacity + 5).validate(),
        "overlap",
    );
    ChunkConfig::new(32)
        .with_overlap(capacity - 1)
        .validate()
        .expect("one token short of a full window still advances");
}

#[test]
fn a_minimum_larger_than_the_window_would_discard_everything() {
    rejects(
        ChunkConfig::new(32).with_min_final_tokens(100).validate(),
        "min_final_tokens",
    );
}

#[test]
fn capacity_reserves_exactly_the_two_special_tokens() {
    assert_eq!(ChunkConfig::new(256).capacity(), 254);
    assert_eq!(ChunkConfig::new(3).capacity(), 1);
}

// ── training hyperparameters ────────────────────────────────────────────────

#[test]
fn default_training_hyperparameters_are_valid() {
    TrainConfig::default()
        .validate()
        .expect("defaults are usable");
}

#[test]
fn counts_of_zero_would_train_nothing() {
    let cases: [(&str, TrainConfig); 6] = [
        (
            "num_iterations",
            TrainConfig {
                num_iterations: 0,
                ..Default::default()
            },
        ),
        (
            "body_epochs",
            TrainConfig {
                body_epochs: 0,
                ..Default::default()
            },
        ),
        (
            "head_epochs",
            TrainConfig {
                head_epochs: 0,
                ..Default::default()
            },
        ),
        (
            "body_batch_size",
            TrainConfig {
                body_batch_size: 0,
                ..Default::default()
            },
        ),
        (
            "head_batch_size",
            TrainConfig {
                head_batch_size: 0,
                ..Default::default()
            },
        ),
        (
            "max_tokens",
            TrainConfig {
                max_tokens: 0,
                ..Default::default()
            },
        ),
    ];
    for (name, config) in cases {
        rejects(config.validate(), name);
    }
}

#[test]
fn learning_rates_must_be_finite_and_positive() {
    for bad in [0.0, -1e-5, f64::NAN, f64::INFINITY] {
        rejects(
            TrainConfig {
                body_lr: bad,
                ..Default::default()
            }
            .validate(),
            "body_lr",
        );
        rejects(
            TrainConfig {
                head_lr: bad,
                ..Default::default()
            }
            .validate(),
            "head_lr",
        );
    }
}

#[test]
fn weight_decay_may_be_zero_but_not_negative() {
    TrainConfig {
        head_weight_decay: 0.0,
        ..Default::default()
    }
    .validate()
    .expect("no regularisation is a legitimate choice");
    rejects(
        TrainConfig {
            head_weight_decay: -0.1,
            ..Default::default()
        }
        .validate(),
        "weight_decay",
    );
}

// ── serialisation ───────────────────────────────────────────────────────────

#[test]
fn a_config_survives_a_json_round_trip() {
    // The bundle stores this as JSON, so anything that does not round-trip is
    // silently lost between training and inference.
    let original = ClassifierConfig::new(["billing", "outage", "feature"])
        .multi_label_at(0.42)
        .with_reducer(Reducer::TopKMeanLogits { k: 7 })
        .with_chunking(
            ChunkConfig::new(128)
                .with_overlap(16)
                .with_min_final_tokens(8),
        )
        .with_hierarchy(4);

    let json = serde_json::to_string(&original).expect("serialises");
    let restored: ClassifierConfig = serde_json::from_str(&json).expect("deserialises");

    assert_eq!(original, restored);
    assert_eq!(restored.threshold(), Some(0.42));
    assert_eq!(restored.reducer, Reducer::TopKMeanLogits { k: 7 });
    assert_eq!(restored.chunk.overlap_tokens, 16);
    assert_eq!(restored.hierarchy_fanout, Some(4));
}
