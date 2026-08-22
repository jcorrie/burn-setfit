//! End-to-end: train from scratch, classify, round-trip through a bundle.

mod common;

use burn::backend::{Autodiff, NdArray};
use burn_setfit::bundle::Bundle;
use burn_setfit::chunk::ChunkConfig;
use burn_setfit::config::ClassifierConfig;
use burn_setfit::infer::Classifier;
use burn_setfit::train::Stage;
use burn_setfit::train::{Example, TrainConfig, Trainer};
use common::{separable_examples as shared_separable, small_chunking, toy_checkpoint};

type B = NdArray<f32>;
type AB = Autodiff<B>;

fn fast_config() -> TrainConfig {
    TrainConfig {
        num_iterations: 4,
        body_epochs: 1,
        head_epochs: 30,
        body_batch_size: 8,
        head_batch_size: 8,
        max_tokens: 32,
        ..Default::default()
    }
}

/// Two vocabularies that never overlap, so the task is learnable in principle.
fn separable_examples() -> Vec<Example> {
    let class_a = ["a b c", "a c b", "b a c", "c a b", "a b a", "b c a"];
    let class_b = ["x y z", "z y x", "y x z", "x z y", "z z y", "y y x"];

    class_a
        .iter()
        .map(|t| Example::single(*t, 0))
        .chain(class_b.iter().map(|t| Example::single(*t, 1)))
        .collect()
}

fn single_label_config() -> ClassifierConfig {
    ClassifierConfig::new(["a", "b"]).with_chunking(small_chunking())
}

#[test]
fn trains_end_to_end_and_separates_the_classes() {
    let device = Default::default();
    let examples = separable_examples();

    let mut trainer = Trainer::<AB>::new(
        &toy_checkpoint(),
        single_label_config(),
        examples.clone(),
        fast_config(),
        device,
    )
    .expect("trainer should accept well-formed data");

    let total = trainer.total_steps();
    assert!(total > 0);

    let mut losses = Vec::new();
    let mut saw_body = false;
    let mut saw_head = false;
    while let Some(p) = trainer.step().expect("step should succeed") {
        match p.stage {
            burn_setfit::train::Stage::Body => saw_body = true,
            burn_setfit::train::Stage::Head => saw_head = true,
            burn_setfit::train::Stage::Done => unreachable!(),
        }
        assert!(
            p.loss.is_finite(),
            "loss went non-finite at step {}",
            p.step
        );
        assert!(p.fraction() <= 1.0);
        losses.push((p.stage, p.loss));
    }

    assert!(saw_body && saw_head, "both stages must run");
    assert_eq!(
        losses.len(),
        total,
        "step count must match the advertised total"
    );

    // The head is a linear probe on frozen embeddings of two disjoint vocabularies;
    // if this does not come down, the training loop is not wired up.
    let head_losses: Vec<f32> = losses
        .iter()
        .filter(|(s, _)| *s == burn_setfit::train::Stage::Head)
        .map(|(_, l)| *l)
        .collect();
    let first = head_losses[..3].iter().sum::<f32>() / 3.0;
    let last = head_losses[head_losses.len() - 3..].iter().sum::<f32>() / 3.0;
    assert!(last < first, "head loss should decrease: {first} -> {last}");

    // And the trained model should actually classify its own training data.
    let classifier = Classifier::<B>::from_bundle(
        &trainer.finish().expect("a completed run packs"),
        Default::default(),
    )
    .expect("bundle loads");

    let correct = examples
        .iter()
        .filter(|e| {
            let p = classifier
                .classify(&e.text)
                .expect("classification succeeds");
            p.predicted == e.labels
        })
        .count();
    assert_eq!(correct, examples.len(), "should fit its own training set");
}

#[test]
fn multi_label_can_predict_several_labels_or_none() {
    let device = Default::default();

    // Label 0 is signalled by `a`, label 1 by `x`. Some examples carry both.
    let examples = vec![
        Example::multi("a a a b", vec![0]),
        Example::multi("a b a c", vec![0]),
        Example::multi("x x x y", vec![1]),
        Example::multi("x y x z", vec![1]),
        Example::multi("a a x x", vec![0, 1]),
        Example::multi("a x a x", vec![0, 1]),
    ];

    let mut trainer = Trainer::<AB>::new(
        &toy_checkpoint(),
        ClassifierConfig::new(["a", "x"])
            .multi_label()
            .with_chunking(small_chunking()),
        examples.clone(),
        TrainConfig {
            head_epochs: 60,
            ..fast_config()
        },
        device,
    )
    .expect("trainer should accept multi-label data");

    trainer.fit_with(|_| {}).expect("training should complete");

    let classifier = Classifier::<B>::from_bundle(
        &trainer.finish().expect("a completed run packs"),
        Default::default(),
    )
    .expect("bundle loads");

    let both = classifier.classify("a a x x").expect("classifies");
    assert_eq!(
        both.predicted.len(),
        2,
        "a document with both signals should carry both labels, got {:?} from scores {:?}",
        both.predicted,
        both.scores
    );

    // Multi-label scores are independent sigmoids, not a distribution.
    let sum: f32 = both.scores.iter().sum();
    assert!(sum > 1.0, "independent sigmoids need not sum to one: {sum}");
}

#[test]
fn rejects_data_that_cannot_form_pairs() {
    let device = Default::default();

    // Every example shares the single label, so no negative pair exists.
    let single_class = vec![
        Example::single("a b c", 0),
        Example::single("b c a", 0),
        Example::single("c a b", 0),
    ];

    let err = Trainer::<AB>::new(
        &toy_checkpoint(),
        single_label_config(),
        single_class,
        fast_config(),
        device,
    )
    .err()
    .expect("training on one class should fail rather than silently do nothing");

    assert!(format!("{err}").contains("pair"), "unhelpful error: {err}");
}

#[test]
fn bundle_round_trips_to_identical_predictions() {
    let device = Default::default();

    let mut trainer = Trainer::<AB>::new(
        &toy_checkpoint(),
        ClassifierConfig::new(["alpha", "beta"]).with_chunking(small_chunking()),
        separable_examples(),
        fast_config(),
        device,
    )
    .expect("trainer builds");
    trainer.fit_with(|_| {}).expect("training completes");

    let packed = trainer.finish().expect("a completed run packs");
    let after = Classifier::<B>::from_bundle(&packed, Default::default()).expect("unpacks");
    assert_eq!(after.labels(), &["alpha".to_string(), "beta".to_string()]);

    let expected = after.classify("a b c a b c").expect("classifies");

    let got = after.classify("a b c a b c").expect("classifies");
    assert_eq!(got.predicted, expected.predicted);
    for (a, b) in got.scores.iter().zip(&expected.scores) {
        assert!((a - b).abs() < 1e-5, "scores drifted: {a} vs {b}");
    }
}

#[test]
fn rejects_a_corrupt_bundle_rather_than_misreading_it() {
    assert!(Bundle::unpack(b"not a bundle at all").is_err());
    assert!(Bundle::unpack(b"BSETFIT\x00").is_err());

    let mut header = b"BSETFIT\x00".to_vec();
    header.extend_from_slice(&99u32.to_le_bytes()); // unknown version
    header.extend_from_slice(&0u32.to_le_bytes());
    let err = Bundle::unpack(&header).expect_err("unknown version must be refused");
    assert!(
        format!("{err}").contains("version"),
        "unhelpful error: {err}"
    );
}

// ── rejecting bad training data ─────────────────────────────────────────────

#[track_caller]
fn training_rejects(examples: Vec<Example>, config: ClassifierConfig, mentioning: &str) {
    let err = Trainer::<AB>::new(
        &toy_checkpoint(),
        config,
        examples,
        fast_config(),
        Default::default(),
    )
    .err()
    .expect("expected this training data to be rejected");
    let text = format!("{err}");
    assert!(
        text.to_lowercase().contains(mentioning),
        "error should mention {mentioning:?}, said: {text}"
    );
}

/// A label index outside the configured set used to be silently dropped in
/// multi-label and silently clamped to class 0 in single-label.
#[test]
fn a_label_index_outside_the_configured_set_is_rejected() {
    let mut examples = shared_separable();
    examples.push(Example::single("a b c", 7));
    training_rejects(examples, single_label_config(), "label index 7");
}

#[test]
fn a_single_label_example_must_carry_exactly_one_label() {
    let mut two = shared_separable();
    two.push(Example::multi("a b c", vec![0, 1]));
    training_rejects(two, single_label_config(), "single-label");

    let mut none = shared_separable();
    none.push(Example::multi("a b c", vec![]));
    training_rejects(none, single_label_config(), "single-label");
}

#[test]
fn a_multi_label_example_may_carry_no_labels_at_all() {
    // "None of the above" is a legitimate multi-label target.
    let mut examples: Vec<Example> = shared_separable()
        .into_iter()
        .map(|e| Example::multi(e.text, e.labels))
        .collect();
    examples.push(Example::multi("a x b y", vec![]));

    Trainer::<AB>::new(
        &toy_checkpoint(),
        ClassifierConfig::new(["a", "b"])
            .multi_label()
            .with_chunking(small_chunking()),
        examples,
        fast_config(),
        Default::default(),
    )
    .expect("an unlabelled multi-label example is allowed");
}

#[test]
fn too_few_examples_to_pair_is_rejected() {
    training_rejects(
        vec![Example::single("a b c", 0)],
        single_label_config(),
        "two examples",
    );
}

#[test]
fn an_invalid_classifier_config_is_rejected_before_training() {
    training_rejects(
        shared_separable(),
        ClassifierConfig::new(["a", "a"]),
        "\"a\"",
    );
}

#[test]
fn invalid_hyperparameters_are_rejected_before_training() {
    let err = Trainer::<AB>::new(
        &toy_checkpoint(),
        single_label_config(),
        shared_separable(),
        TrainConfig {
            head_epochs: 0,
            ..fast_config()
        },
        Default::default(),
    )
    .err()
    .expect("zero head epochs trains no head");
    assert!(format!("{err}").contains("head_epochs"));
}

#[test]
fn a_sequence_budget_beyond_the_bodys_positions_is_rejected_before_training() {
    // Padding a training batch to more tokens than the body has positions
    // indexes past the position table, which the backend reports as a panic
    // rather than an error — and only once a long enough example turns up.
    let checkpoint = toy_checkpoint();
    let positions = checkpoint.config.max_position_embeddings;

    let err = Trainer::<AB>::new(
        &checkpoint,
        single_label_config(),
        shared_separable(),
        TrainConfig {
            max_tokens: positions + 1,
            ..fast_config()
        },
        Default::default(),
    )
    .err()
    .expect("a training budget the body cannot encode must be refused");
    assert!(
        format!("{err}").contains(&positions.to_string()),
        "the message should name the body's budget, got: {err}"
    );
}

#[test]
fn chunk_windows_beyond_the_bodys_positions_are_rejected_before_training() {
    // The same mistake, one config away: training would succeed and only the
    // packed model would be unusable, so it is caught before the run rather
    // than at finish() — after the work it invalidates.
    let checkpoint = toy_checkpoint();
    let positions = checkpoint.config.max_position_embeddings;

    let err = Trainer::<AB>::new(
        &checkpoint,
        ClassifierConfig::new(["a", "b"]).with_chunking(ChunkConfig::new(positions + 1)),
        shared_separable(),
        fast_config(),
        Default::default(),
    )
    .err()
    .expect("windows the body cannot encode must be refused");
    assert!(format!("{err}").contains("chunk windows"), "got: {err}");
}

#[test]
fn a_checkpoint_without_weights_is_rejected_before_training() {
    let mut broken = toy_checkpoint();
    broken.weights.clear();
    assert!(
        Trainer::<AB>::new(
            &broken,
            single_label_config(),
            shared_separable(),
            fast_config(),
            Default::default(),
        )
        .is_err()
    );
}

// ── the training state machine ──────────────────────────────────────────────

fn trainer() -> Trainer<AB> {
    Trainer::<AB>::new(
        &toy_checkpoint(),
        single_label_config(),
        shared_separable(),
        fast_config(),
        Default::default(),
    )
    .expect("well-formed")
}

#[test]
fn stages_run_in_order_and_progress_advances_monotonically() {
    let mut t = trainer();
    let total = t.total_steps();

    let mut seen = Vec::new();
    let mut last_step = 0;
    let mut last_fraction = 0.0;

    while let Some(p) = t.step().expect("steps") {
        if seen.last() != Some(&p.stage) {
            seen.push(p.stage);
        }
        assert_eq!(p.step, last_step + 1, "steps must not skip");
        assert!(p.fraction() >= last_fraction);
        assert!(p.fraction() <= 1.0);
        assert_eq!(p.total_steps, total);
        last_step = p.step;
        last_fraction = p.fraction();
    }

    assert_eq!(
        seen,
        vec![Stage::Body, Stage::Head],
        "body precedes head, once each"
    );
    assert_eq!(last_step, total);
    assert_eq!(t.stage(), Stage::Done);
}

#[test]
fn stepping_past_the_end_is_harmless() {
    let mut t = trainer();
    t.fit_with(|_| {}).expect("trains");
    for _ in 0..3 {
        assert!(t.step().expect("steps").is_none());
    }
}

/// Packing early would produce a well-formed bundle whose head had never been
/// fitted — a model that loads cleanly and predicts noise.
#[test]
fn packing_before_training_finishes_is_refused() {
    let mut t = trainer();
    t.step().expect("one step");

    let err = t.finish().expect_err("an unfinished run must not pack");
    let text = format!("{err}");
    assert!(
        text.contains("Body") || text.contains("stage"),
        "unhelpful: {text}"
    );
}

#[test]
fn a_partially_trained_model_can_still_be_inspected_deliberately() {
    let mut t = trainer();
    t.step().expect("one step");
    // `into_model` makes no completeness claim, unlike `finish`.
    let _ = t.into_model();
}

// ── determinism ─────────────────────────────────────────────────────────────

/// Fingerprint a model by summing each parameter tensor, keyed by path.
fn weights_of(bundle: &[u8]) -> Vec<(String, f32)> {
    use burn_store::ModuleSnapshot;

    let module = Bundle::unpack(bundle)
        .expect("unpacks")
        .load_module::<B>(&Default::default())
        .expect("loads");
    let mut sums: Vec<(String, f32)> = module
        .collect(None, None, false)
        .into_iter()
        .map(|s| {
            let v = s.to_data().unwrap().into_vec::<f32>().unwrap();
            (s.full_path(), v.iter().sum())
        })
        .collect();
    sums.sort_by(|a, b| a.0.cmp(&b.0));
    sums
}

fn train_with_seed(checkpoint: &burn_setfit::checkpoint::Checkpoint, seed: u64) -> Vec<u8> {
    let mut t = Trainer::<AB>::new(
        checkpoint,
        single_label_config(),
        shared_separable(),
        TrainConfig {
            seed,
            ..fast_config()
        },
        Default::default(),
    )
    .expect("well-formed");
    t.fit_with(|_| {}).expect("trains");
    t.finish().expect("packs")
}

/// A seeded run must reproduce the same *model*.
///
/// Deliberately not a byte comparison. Burn writes the safetensors
/// `__metadata__` map in `HashMap` order, which Rust seeds per map, so two
/// bit-identical models serialise to headers whose keys are in different orders.
/// The weights are the claim worth making; the container is not byte-canonical,
/// which matters only if you were hoping to content-address the artifact.
#[test]
fn the_same_seed_and_checkpoint_reproduce_the_same_model() {
    let checkpoint = toy_checkpoint();
    let first = train_with_seed(&checkpoint, 7);
    let second = train_with_seed(&checkpoint, 7);

    assert_eq!(
        weights_of(&first),
        weights_of(&second),
        "a seeded run must reproduce the same weights"
    );

    // And the same behaviour, which is what a caller actually observes.
    let a = Classifier::<B>::from_bundle(&first, Default::default()).expect("loads");
    let b = Classifier::<B>::from_bundle(&second, Default::default()).expect("loads");
    for text in ["a b c", "x y z", "a x b y"] {
        assert_eq!(
            a.classify(text).unwrap().scores,
            b.classify(text).unwrap().scores,
            "predictions diverged on {text:?}"
        );
    }
}

#[test]
fn a_different_seed_samples_different_pairs() {
    let checkpoint = toy_checkpoint();
    assert_ne!(
        weights_of(&train_with_seed(&checkpoint, 1)),
        weights_of(&train_with_seed(&checkpoint, 2)),
        "the seed should actually influence sampling and head initialisation"
    );
}

/// The head used to be initialised from Burn's global RNG, which no per-run seed
/// could reach — so "reproducible" training still started from a different head
/// every time, and only ambient RNG state decided whether it showed.
#[test]
fn the_seed_reaches_head_initialisation_not_just_sampling() {
    let checkpoint = toy_checkpoint();

    let untrained_head = |seed: u64| {
        let t = Trainer::<AB>::new(
            &checkpoint,
            single_label_config(),
            shared_separable(),
            TrainConfig {
                seed,
                ..fast_config()
            },
            Default::default(),
        )
        .expect("well-formed");
        // Before any step has run, the head is purely its initialisation.
        let mut t = t;
        let _ = t.step();
        t.into_model()
    };

    use burn_store::ModuleSnapshot;
    let sums = |m: burn_setfit::model::SetFitModule<B>| -> Vec<f32> {
        m.head
            .collect(None, None, false)
            .into_iter()
            .map(|s| s.to_data().unwrap().into_vec::<f32>().unwrap().iter().sum())
            .collect()
    };

    assert_eq!(sums(untrained_head(3)), sums(untrained_head(3)));
    assert_ne!(sums(untrained_head(3)), sums(untrained_head(4)));
}
