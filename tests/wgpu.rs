//! The `wgpu` backend, checked against `ndarray`.
//!
//! Issue #5 exists because this backend compiled and had never been executed.
//! These tests execute it. They need the feature, so they are absent from the
//! default `cargo test`; the command is in `CLAUDE.md`.
//!
//! **What a pass here does and does not mean.** The adapter in CI and in a
//! container is usually `llvmpipe` — Mesa's software Vulkan — so this exercises
//! the whole wgpu path (kernel codegen, buffer management, dispatch, readback)
//! on a CPU. That catches a backend that does not run, or that disagrees with
//! `ndarray`, which is what these tests are for. It says nothing about real
//! hardware, and nothing at all about WebGPU in a browser, which is a separate
//! claim with its own unknowns.
//!
//! Tolerances are loose on purpose. Reductions run in a different order than on
//! `ndarray`, so agreement is approximate by nature; a mismatch worth reporting
//! is a wrong answer, not a last-place bit.
#![cfg(feature = "wgpu")]

mod common;

use burn::tensor::{Device, DeviceKind};
use burn_setfit::config::ClassifierConfig;
use burn_setfit::infer::Classifier;
use burn_setfit::train::Trainer;
use common::{
    cpu, fast_train_config, separable_examples, small_chunking, toy_bundle, toy_checkpoint, words,
};

/// The wgpu device. Named, like [`cpu`], because with both features compiled in
/// a defaulted device is whichever backend Burn prefers, not the one under test.
fn gpu() -> Device {
    Device::wgpu(DeviceKind::DefaultDevice)
}

/// Where two backends may legitimately differ: summation order, mostly.
const TOL: f32 = 1e-3;

fn config() -> ClassifierConfig {
    ClassifierConfig::new(["alpha", "beta"]).with_chunking(small_chunking())
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: different lengths");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= TOL,
            "{what}: element {i} differs by {} ({x} vs {y})",
            (x - y).abs()
        );
    }
}

#[test]
fn a_tensor_survives_a_round_trip_through_the_device() {
    // The cheapest possible failure to diagnose: if this fails, no adapter was
    // available and every other failure in this file is a consequence.
    use burn::tensor::Tensor;
    let t = Tensor::<1>::from_data([1.0f32, 2.0, 3.0], &gpu());
    let out = (t.clone() * t)
        .into_data()
        .try_into_vec::<f32>()
        .expect("floats");
    assert_close(&out, &[1.0, 4.0, 9.0], "elementwise square");
}

#[test]
fn classifying_on_wgpu_agrees_with_ndarray() {
    // One bundle, both backends. A `.setfit` bundle is bytes, so the model is
    // identical by construction and any difference is the backend's.
    let bundle = toy_bundle(config());
    let text = words(200);

    let cpu = Classifier::from_bundle(&bundle, cpu()).expect("loads on cpu");
    let gpu = Classifier::from_bundle(&bundle, gpu()).expect("loads on wgpu");

    let want = cpu.classify(&text).expect("classifies on cpu");
    let got = gpu.classify(&text).expect("classifies on wgpu");

    assert_close(&want.scores, &got.scores, "scores");
    assert_eq!(want.predicted, got.predicted, "predicted labels");
    assert_eq!(want.chunks_seen, got.chunks_seen, "chunks");
}

#[test]
fn classifying_many_on_wgpu_matches_one_at_a_time() {
    // Padded passes of mixed lengths are where a backend's masking would show:
    // a short document padded beside a long one must score as it does alone.
    let gpu = Classifier::from_bundle(&toy_bundle(config()), gpu()).expect("loads on wgpu");
    let docs = ["a b c".to_string(), words(200), words(9), String::new()];
    let refs: Vec<&str> = docs.iter().map(String::as_str).collect();

    let many = gpu.classify_many(&refs).expect("classifies on wgpu");
    for (doc, got) in docs.iter().zip(&many) {
        let alone = gpu.classify(doc).expect("classifies on wgpu");
        assert_close(&alone.scores, &got.scores, "scores");
        assert_eq!(alone.predicted, got.predicted, "predicted labels");
        assert_eq!(alone.chunks_seen, got.chunks_seen, "chunks");
    }
}

#[test]
fn embedding_on_wgpu_agrees_with_ndarray() {
    let bundle = toy_bundle(config());
    let texts = ["a b c", "x y z"];

    let cpu = Classifier::from_bundle(&bundle, cpu()).expect("loads on cpu");
    let gpu = Classifier::from_bundle(&bundle, gpu()).expect("loads on wgpu");

    let want = cpu.embed(&texts).expect("embeds on cpu");
    let got = gpu.embed(&texts).expect("embeds on wgpu");

    assert_eq!(want.len(), got.len());
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        assert_close(w, g, &format!("embedding {i}"));
    }
}

#[test]
fn training_on_wgpu_learns_the_task() {
    // Not compared against ndarray step for step: the two backends diverge
    // through the optimiser once their gradients differ in the last bits, and
    // demanding they not is a test that fails for the wrong reason. What must
    // hold is that training on wgpu produces a model that works.
    let mut trainer = Trainer::new(
        &toy_checkpoint(),
        config(),
        separable_examples(),
        fast_train_config(),
        gpu(),
    )
    .expect("toy training data is well formed");

    let mut losses = Vec::new();
    trainer
        .fit_with(|p| losses.push(p.loss))
        .expect("trains on wgpu");
    let bundle = trainer.finish().expect("a completed run packs");

    assert!(!losses.is_empty(), "training reported steps");
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "a non-finite loss means the backend produced garbage, not that the \
         model learned badly: {losses:?}"
    );

    // The two classes have disjoint vocabularies, so a model that trained at
    // all separates them.
    let classifier = Classifier::from_bundle(&bundle, gpu()).expect("loads on wgpu");
    let alpha = classifier.classify("a b c").expect("classifies");
    let beta = classifier.classify("x y z").expect("classifies");
    assert_ne!(
        alpha.predicted, beta.predicted,
        "disjoint vocabularies landed in the same class"
    );
}
