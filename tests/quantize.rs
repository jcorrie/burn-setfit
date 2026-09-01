//! Quantized bundles: smaller, still correct, and honest about what they cost.
//!
//! The size claims here are on the toy model, whose 32-wide body is mostly
//! LayerNorm and bias — the parameters quantization deliberately skips — so the
//! ratios are worse than a real checkpoint's. `examples/quantize.rs` measures
//! MiniLM-L6, where the embedding table dominates and the ratios approach the
//! theoretical half and quarter.

mod common;

use burn::backend::NdArray;
use burn_setfit::bundle::{Bundle, Manifest};
use burn_setfit::config::ClassifierConfig;
use burn_setfit::infer::Classifier;
use burn_setfit::quantize::Quantization;
use common::{small_chunking, toy_bundle_quantized, words};

type B = NdArray<f32>;

fn config() -> ClassifierConfig {
    ClassifierConfig::new(["alpha", "beta"]).with_chunking(small_chunking())
}

fn classifier(q: Quantization) -> Classifier<B> {
    Classifier::<B>::from_bundle(&toy_bundle_quantized(config(), q), Default::default())
        .expect("a quantized bundle loads")
}

#[test]
fn every_mode_round_trips_into_a_working_model() {
    for mode in [Quantization::None, Quantization::F16, Quantization::Int8] {
        let bytes = toy_bundle_quantized(config(), mode);
        let bundle = Bundle::unpack(&bytes).expect("unpacks");
        assert_eq!(
            bundle.manifest.quantization, mode,
            "the manifest must record what was done to the weights"
        );
        let c = Classifier::<B>::from_bundle(&bytes, Default::default()).expect("loads");
        let p = c.classify(&words(40)).expect("classifies");
        assert_eq!(p.scores.len(), 2);
        assert!(
            p.scores.iter().all(|s| s.is_finite()),
            "{mode:?} produced non-finite scores"
        );
    }
}

#[test]
fn quantizing_shrinks_the_weights() {
    let plain = toy_bundle_quantized(config(), Quantization::None).len();
    let f16 = toy_bundle_quantized(config(), Quantization::F16).len();
    let int8 = toy_bundle_quantized(config(), Quantization::Int8).len();

    assert!(f16 < plain, "f16 {f16} should be under plain {plain}");
    assert!(int8 < f16, "int8 {int8} should be under f16 {f16}");
}

/// The reason to prefer f16 when in doubt.
#[test]
fn f16_barely_moves_the_scores_and_int8_moves_them_more() {
    let text = words(120);

    let exact = classifier(Quantization::None).classify(&text).expect("cpu");
    let half = classifier(Quantization::F16).classify(&text).expect("f16");
    let byte = classifier(Quantization::Int8)
        .classify(&text)
        .expect("int8");

    let drift = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    let f16_drift = drift(&exact.scores, &half.scores);
    let int8_drift = drift(&exact.scores, &byte.scores);

    // These bounds are roughly an order of magnitude above what the toy model
    // actually produces (~1e-5 and ~1e-4), which leaves room for a different
    // random initialisation without leaving room for a bug. A failure here
    // means the weights are being corrupted rather than rounded — when this
    // was first written it caught exactly that, though the cause turned out to
    // be the harness comparing two separately trained models rather than the
    // quantizer. See `toy_bundle_quantized`.
    assert!(
        f16_drift < 1e-3,
        "f16 drifted {f16_drift}, which is far more than half precision loses"
    );
    assert!(
        int8_drift < 1e-2,
        "int8 drifted {int8_drift}, far enough to suggest a bug rather than precision loss"
    );
    assert!(
        f16_drift <= int8_drift,
        "f16 ({f16_drift}) should not drift further than int8 ({int8_drift})"
    );
}

/// A quantized bundle must not be readable as though it were an ordinary one.
#[test]
fn a_quantized_bundle_declares_a_newer_format_version() {
    let plain = toy_bundle_quantized(config(), Quantization::None);
    let quantized = toy_bundle_quantized(config(), Quantization::F16);

    // Bytes 8..12 are the version, little-endian.
    let version = |b: &[u8]| u32::from_le_bytes(b[8..12].try_into().unwrap());

    assert_eq!(
        version(&plain),
        1,
        "an unquantized bundle must stay a version 1 file that older builds read"
    );
    assert_eq!(
        version(&quantized),
        2,
        "a quantized bundle must announce itself, so an older build refuses it \
         rather than reading f16 bytes as f32"
    );
}

/// Weights written without the field at all — the pre-quantization layout.
#[test]
fn a_manifest_without_a_quantization_field_reads_as_unquantized() {
    let json = serde_json::to_value(Manifest::new(
        burn_setfit::minilm::MiniLmVariant::L6,
        common::toy_body_config(),
        config(),
    ))
    .expect("serialises");
    let mut map = json.as_object().expect("an object").clone();
    map.remove("quantization");

    let manifest: Manifest =
        serde_json::from_value(serde_json::Value::Object(map)).expect("an older manifest parses");
    assert_eq!(manifest.quantization, Quantization::None);
}
