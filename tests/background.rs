//! The background class and per-chunk output — the decode-time half of #4.
//!
//! What these can and cannot check: the toy model's weights are near-random, so
//! nothing here measures whether a background class *classifies* long documents
//! better. That question needs the real checkpoint and
//! `examples/long_document.rs`. What is checked is the decision rule itself,
//! which is exact arithmetic and holds regardless of the model behind it.

mod common;

use burn::backend::NdArray;
use burn_setfit::config::ClassifierConfig;
use burn_setfit::head::TaskMode;
use burn_setfit::infer::{Classifier, decode, decode_against_background};
use common::{small_chunking, toy_bundle, words};

type B = NdArray<f32>;

// ── the decision rule ───────────────────────────────────────────────────────

#[test]
fn a_label_must_outscore_the_background_not_merely_the_threshold() {
    let task = TaskMode::MultiLabel { threshold: 0.4 };
    // `a` 0.61, `b` 0.45, background 0.50. Both clear 0.4; only `a` clears the
    // background. This is the case a fixed threshold gets wrong.
    let scores = [0.61, 0.45, 0.50];

    assert_eq!(decode_against_background(&scores, task, Some(2)), vec![0]);
    assert_eq!(
        decode(&scores, task),
        vec![0, 1, 2],
        "without a background class every label above the threshold is predicted"
    );
}

#[test]
fn the_background_class_is_never_itself_predicted() {
    let task = TaskMode::MultiLabel { threshold: 0.1 };
    // The background wins outright and still must not appear.
    let scores = [0.2, 0.2, 0.9];
    let got = decode_against_background(&scores, task, Some(2));
    assert!(
        !got.contains(&2),
        "background leaked into the prediction: {got:?}"
    );
    assert!(
        got.is_empty(),
        "nothing outscores the background here: {got:?}"
    );
}

/// The finding this exists for: a softmax cannot abstain, unless it has a class
/// to abstain *to*.
#[test]
fn a_single_label_head_abstains_when_the_background_wins() {
    let task = TaskMode::SingleLabel;
    let filler = [0.20, 0.15, 0.65];

    assert!(
        decode_against_background(&filler, task, Some(2)).is_empty(),
        "filler must produce no label at all"
    );
    assert_eq!(
        decode(&filler, task),
        vec![2],
        "without a background class the argmax is forced to name something"
    );
}

#[test]
fn a_real_winner_still_wins_in_single_label() {
    let task = TaskMode::SingleLabel;
    let signal = [0.70, 0.10, 0.20];
    assert_eq!(decode_against_background(&signal, task, Some(2)), vec![0]);
}

// ── configuration ───────────────────────────────────────────────────────────

#[test]
fn naming_a_label_that_does_not_exist_is_refused() {
    let err = ClassifierConfig::new(["billing", "outage", "other"])
        .with_background_class("nope")
        .expect_err("there is no such label");
    assert!(format!("{err}").contains("nope"), "unhelpful: {err}");
}

#[test]
fn a_background_class_needs_something_to_sit_against() {
    // Two labels minus the background leaves one, and a classifier with one
    // class has no decision to make.
    let err = ClassifierConfig::new(["billing", "other"])
        .with_background_class("other")
        .expect("the label exists")
        .validate()
        .expect_err("one real class is not a classifier");
    assert!(format!("{err}").contains("background"), "unhelpful: {err}");
}

#[test]
fn a_background_class_survives_the_bundle() {
    let config = ClassifierConfig::new(["alpha", "beta", "other"])
        .multi_label()
        .with_background_class("other")
        .expect("label exists")
        .with_chunking(small_chunking());

    let classifier =
        Classifier::<B>::from_bundle(&toy_bundle(config), Default::default()).expect("loads");

    assert_eq!(classifier.cfg().background, Some(2));
    assert_eq!(classifier.cfg().background_label(), Some("other"));
}

// ── per-chunk output ────────────────────────────────────────────────────────

fn chunked() -> Classifier<B> {
    let config = ClassifierConfig::new(["alpha", "beta"]).with_chunking(small_chunking());
    Classifier::<B>::from_bundle(&toy_bundle(config), Default::default()).expect("loads")
}

#[test]
fn every_chunk_gets_a_row_covering_the_document_in_order() {
    let classifier = chunked();
    let text = words(200);

    let rows = classifier.classify_chunks(&text).expect("classifies");
    let whole = classifier.classify(&text).expect("classifies");

    assert_eq!(
        rows.len(),
        whole.chunks_seen,
        "one row per chunk the reducing path saw"
    );
    assert!(rows.len() > 1, "the test text must actually window");

    for row in &rows {
        assert_eq!(row.scores.len(), 2, "one score per label");
        assert!(row.scores.iter().all(|s| s.is_finite()));
        assert!(row.token_count > 0, "an empty chunk should not exist");
        assert!(
            row.byte_range.end <= text.len(),
            "byte range {:?} runs past the document",
            row.byte_range
        );
    }

    // Chunks arrive in document order, which is what makes the ranges useful.
    for pair in rows.windows(2) {
        assert!(
            pair[0].byte_range.start <= pair[1].byte_range.start,
            "chunks out of order: {:?} then {:?}",
            pair[0].byte_range,
            pair[1].byte_range
        );
    }
}

#[test]
fn per_chunk_scores_match_the_evidence_the_reducing_path_reports() {
    // Both paths go through `forward_rows`, so a disagreement here means one of
    // them is post-processing differently — which is exactly the drift worth
    // catching.
    let classifier = chunked().with_evidence_limit(32);
    let text = words(120);

    let rows = classifier.classify_chunks(&text).expect("classifies");
    let whole = classifier.classify(&text).expect("classifies");

    for evidence in &whole.evidence {
        let row = rows
            .iter()
            .find(|r| r.byte_range == evidence.byte_range)
            .expect("every evidence span is one of the chunks");
        assert!(
            (row.scores[evidence.label] - evidence.score).abs() < 1e-6,
            "chunk {:?} scored {} here and {} as evidence",
            evidence.byte_range,
            row.scores[evidence.label],
            evidence.score
        );
    }
}

#[test]
fn classify_chunks_async_agrees_with_the_blocking_one() {
    use burn::tensor::try_read_sync;

    let classifier = chunked();
    let text = words(80);

    let sync = classifier.classify_chunks(&text).expect("classifies");
    let asynchronous = try_read_sync(classifier.classify_chunks_async(&text))
        .expect("a native future completes")
        .expect("classifies");

    assert_eq!(sync.len(), asynchronous.len());
    for (a, b) in sync.iter().zip(&asynchronous) {
        assert_eq!(a.byte_range, b.byte_range);
        assert_eq!(a.scores, b.scores);
        assert_eq!(a.predicted, b.predicted);
    }
}
