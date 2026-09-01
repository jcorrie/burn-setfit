//! The `_async` twins, against the blocking methods they implement.
//!
//! `ndarray` reads back immediately, so nothing here can prove anything about a
//! GPU backend in a browser — that combination is unverified and says so, in
//! `src/readback.rs` and in the README. What these tests do hold is the
//! invariant that makes the async path trustworthy at all: the blocking methods
//! are wrappers around the async ones, so there is one implementation, and any
//! divergence between the two is a bug in the wrapper rather than a second
//! code path that drifted.

mod common;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::try_read_sync;
use burn_setfit::checkpoint::Checkpoint;
use burn_setfit::config::ClassifierConfig;
use burn_setfit::infer::Classifier;
use burn_setfit::train::{Progress, Trainer};
use common::{
    fast_train_config, separable_examples, small_chunking, toy_bundle, toy_checkpoint, words,
};
use core::future::Future;

type B = NdArray<f32>;
type Ad = Autodiff<NdArray<f32>>;

/// Run a future to completion on this thread.
///
/// Burn's own `try_read_sync` is a real blocking executor on native targets and
/// a single poll on wasm; native is the only place these tests run, so this
/// always completes. Using it keeps the suite free of an async runtime.
fn block_on<T>(f: impl Future<Output = T>) -> T {
    try_read_sync(f).expect("a native future runs to completion without yielding")
}

fn single_label() -> Classifier<B> {
    let config = ClassifierConfig::new(["alpha", "beta"]).with_chunking(small_chunking());
    Classifier::<B>::from_bundle(&toy_bundle(config), Default::default()).expect("loads")
}

// ── inference ───────────────────────────────────────────────────────────────

#[test]
fn classify_async_agrees_with_classify() {
    let classifier = single_label();
    // Long enough to span several chunks, so the batching loop is exercised
    // rather than a single forward pass.
    let text = words(200);

    let sync = classifier.classify(&text).expect("classifies");
    let asynchronous = block_on(classifier.classify_async(&text)).expect("classifies");

    assert_eq!(sync.scores, asynchronous.scores);
    assert_eq!(sync.predicted, asynchronous.predicted);
    assert_eq!(sync.chunks_seen, asynchronous.chunks_seen);
    assert_eq!(sync.evidence.len(), asynchronous.evidence.len());
}

#[test]
fn classify_stream_async_agrees_with_classify_stream() {
    let classifier = single_label();
    let pieces = || (0..8).map(|_| words(30) + "\n");

    let sync = classifier.classify_stream(pieces()).expect("classifies");
    let asynchronous = block_on(classifier.classify_stream_async(pieces())).expect("classifies");

    assert_eq!(sync.scores, asynchronous.scores);
    assert_eq!(sync.chunks_seen, asynchronous.chunks_seen);
}

#[test]
fn embed_async_agrees_with_embed() {
    let classifier = single_label();
    let texts = ["a b c", "x y z"];

    let sync = classifier.embed(&texts).expect("embeds");
    let asynchronous = block_on(classifier.embed_async(&texts)).expect("embeds");

    assert_eq!(sync, asynchronous);
}

#[test]
fn embed_async_returns_nothing_for_an_empty_batch() {
    // The zero-row guard lives before the first await, so it is worth checking
    // that it survived being moved into the async body.
    let classifier = single_label();
    assert!(
        block_on(classifier.embed_async(&[]))
            .expect("embeds")
            .is_empty()
    );
}

// ── training ────────────────────────────────────────────────────────────────

fn trainer(checkpoint: &Checkpoint) -> Trainer<Ad> {
    Trainer::<Ad>::new(
        checkpoint,
        ClassifierConfig::new(["alpha", "beta"]),
        separable_examples(),
        fast_train_config(),
        Default::default(),
    )
    .expect("toy training data is well formed")
}

#[test]
fn stepping_asynchronously_trains_the_same_model() {
    // One checkpoint, cloned. `toy_checkpoint()` initialises randomly, so
    // calling it twice would compare two different models learning two
    // different things — see the note on `toy_bundle`.
    let checkpoint = toy_checkpoint();

    let sync: Vec<Progress> = {
        let mut seen = Vec::new();
        trainer(&checkpoint)
            .fit_with(|p| seen.push(p))
            .expect("trains");
        seen
    };
    let asynchronous: Vec<Progress> = {
        let mut seen = Vec::new();
        block_on(trainer(&checkpoint).fit_with_async(|p| seen.push(p))).expect("trains");
        seen
    };

    assert_eq!(sync.len(), asynchronous.len(), "same number of steps");
    assert!(!sync.is_empty(), "the toy config trains for some steps");
    for (a, b) in sync.iter().zip(&asynchronous) {
        assert_eq!(a.stage, b.stage);
        assert_eq!(a.step, b.step);
        assert_eq!(a.total_steps, b.total_steps);
        // Bit-identical: same seed, same data, same order, one implementation.
        assert_eq!(a.loss, b.loss, "loss diverged at step {}", a.step);
    }
}

#[test]
fn step_async_crosses_the_body_to_head_boundary() {
    // The stage transition used to recurse through `step()`; the async version
    // falls through instead, because a self-awaiting `async fn` has an
    // infinitely sized future. This checks the fall-through still hands the
    // caller a head step rather than losing one or stalling.
    let mut t = trainer(&toy_checkpoint());

    let mut stages = Vec::new();
    while let Some(p) = block_on(t.step_async()).expect("steps") {
        stages.push(p.stage);
    }

    use burn_setfit::train::Stage;
    assert!(stages.contains(&Stage::Body), "body stage ran");
    assert!(stages.contains(&Stage::Head), "head stage ran");
    let first_head = stages.iter().position(|s| *s == Stage::Head).expect("head");
    let last_body = stages
        .iter()
        .rposition(|s| *s == Stage::Body)
        .expect("body");
    assert!(last_body < first_head, "stages do not interleave");
}
