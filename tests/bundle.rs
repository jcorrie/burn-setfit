//! The `.setfit` container: what survives it, and what it refuses.

mod common;

use burn::backend::NdArray;
use burn_setfit::bundle::{Bundle, FORMAT_VERSION, Manifest};
use burn_setfit::chunk::ChunkConfig;
use burn_setfit::config::ClassifierConfig;
use burn_setfit::infer::Classifier;
use burn_setfit::minilm::{MiniLmConfig, MiniLmVariant};
use burn_setfit::model::SetFitModule;
use burn_setfit::reduce::Reducer;
use common::{small_chunking, toy_body_config, toy_bundle, toy_checkpoint, toy_tokenizer_json};

type B = NdArray<f32>;

fn config() -> ClassifierConfig {
    ClassifierConfig::new(["alpha", "beta"]).with_chunking(small_chunking())
}

// ── round-tripping ──────────────────────────────────────────────────────────

#[test]
fn every_configuration_field_survives_the_container() {
    // Anything that fails to round-trip is silently lost between training and
    // inference, where it looks like a model that behaves oddly.
    let original = ClassifierConfig::new(["alpha", "beta"])
        .multi_label_at(0.31)
        .with_reducer(Reducer::LogSumExp { temperature: 2.5 })
        .with_chunking(
            ChunkConfig::new(48)
                .with_overlap(6)
                .with_min_final_tokens(3),
        )
        .with_hierarchy(3);

    let bundle = Bundle::unpack(&toy_bundle(original.clone())).expect("unpacks");
    assert_eq!(bundle.manifest.classifier, original);
    assert_eq!(bundle.manifest.variant, MiniLmVariant::L6);
    assert_eq!(
        bundle.manifest.body.hidden_size,
        toy_body_config().hidden_size
    );
}

#[test]
fn the_tokenizer_travels_with_the_weights() {
    let bundle = Bundle::unpack(&toy_bundle(config())).expect("unpacks");
    assert_eq!(bundle.tokenizer, toy_tokenizer_json());
    bundle
        .load_tokenizer()
        .expect("the carried tokenizer is usable");
}

#[test]
fn a_bundle_is_self_sufficient() {
    // One fetch, one call, no side files: the property the format exists for.
    let classifier =
        Classifier::<B>::from_bundle(&toy_bundle(config()), Default::default()).expect("loads");
    assert_eq!(
        classifier.labels(),
        &["alpha".to_string(), "beta".to_string()]
    );
    classifier.classify("a b c").expect("classifies");
}

// ── rejecting malformed input ───────────────────────────────────────────────

#[test]
fn rejects_input_that_is_not_a_bundle() {
    assert!(Bundle::unpack(b"").is_err());
    assert!(Bundle::unpack(b"not a bundle at all").is_err());
    assert!(Bundle::unpack(&[0u8; 4096]).is_err());
}

#[test]
fn rejects_a_format_version_it_cannot_read() {
    let mut header = b"BSETFIT\x00".to_vec();
    header.extend_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes());

    let err = Bundle::unpack(&header).expect_err("a future version must be refused");
    assert!(format!("{err}").contains("version"), "unhelpful: {err}");
}

/// Truncation is the realistic corruption: an interrupted download, a partial
/// write. Every prefix must produce an error rather than a panic or, worse, a
/// bundle assembled from whatever bytes happened to be there.
#[test]
fn no_prefix_of_a_valid_bundle_parses_or_panics() {
    let full = toy_bundle(config());

    // Dense at the head where the length fields live, sparse through the payload.
    let offsets = (0..256.min(full.len()))
        .chain((256..full.len()).step_by(997))
        .collect::<Vec<_>>();

    for cut in offsets {
        let result = Bundle::unpack(&full[..cut]);
        assert!(
            result.is_err(),
            "a {cut}-byte prefix of a {}-byte bundle must not parse",
            full.len()
        );
    }

    // The whole thing, however, must.
    Bundle::unpack(&full).expect("the untruncated bundle parses");
}

#[test]
fn a_corrupted_length_field_does_not_read_out_of_bounds() {
    let mut bundle = toy_bundle(config());
    // The manifest length lives at byte 12; claim it is enormous.
    bundle[12..16].copy_from_slice(&u32::MAX.to_le_bytes());

    let err = Bundle::unpack(&bundle).expect_err("an impossible length must be refused");
    assert!(format!("{err}").contains("truncated") || format!("{err}").contains("manifest"));
}

#[test]
fn a_bundle_carrying_an_invalid_config_is_refused_on_load() {
    // Reach into the serialised manifest and break it, as a hand-edited or
    // version-skewed bundle might be.
    let mut tampered = toy_bundle(config());
    let at = tampered
        .windows(7)
        .position(|w| w == b"\"alpha\"")
        .expect("the label appears in the serialised manifest");
    // Same length, so no offset shifts: "alpha" -> "beta" duplicates a label.
    tampered[at..at + 7].copy_from_slice(b"\"beta\" ");

    // Either the JSON no longer parses or validation catches the duplicate;
    // both are correct, and neither is a silently mislabelled model.
    assert!(Bundle::unpack(&tampered).is_err());
}

#[test]
fn a_manifest_whose_windows_outrun_the_position_table_is_refused() {
    // Position ids run 0..seq_len, so a window longer than the position table
    // indexes past the end of it — which surfaces as an out-of-bounds panic
    // inside the backend, and only once a document long enough to fill a window
    // arrives. Both published MiniLM checkpoints have 512 positions against a
    // 256-token window, so this is only reachable on a smaller body.
    let small_body = MiniLmConfig {
        max_position_embeddings: 64,
        ..toy_body_config()
    };
    let manifest = Manifest::new(
        MiniLmVariant::L6,
        small_body,
        ClassifierConfig::new(["alpha", "beta"]).with_chunking(ChunkConfig::new(128)),
    );

    let err = manifest
        .validate()
        .expect_err("windows the body cannot encode must be refused");
    let message = format!("{err}");
    assert!(
        message.contains("128") && message.contains("64"),
        "the message should name both budgets, got: {message}"
    );
}

#[test]
fn a_window_the_body_can_just_encode_is_accepted() {
    // The boundary is inclusive: a window exactly as long as the position table
    // uses positions 0..max, which is precisely what the table holds.
    let manifest = Manifest::new(
        MiniLmVariant::L6,
        MiniLmConfig {
            max_position_embeddings: 64,
            ..toy_body_config()
        },
        ClassifierConfig::new(["alpha", "beta"]).with_chunking(ChunkConfig::new(64)),
    );
    manifest.validate().expect("a window that fits is fine");
}

// ── manifest and weights must agree ─────────────────────────────────────────

#[test]
fn packing_refuses_a_manifest_that_misdescribes_the_head() {
    let device = Default::default();
    let module = SetFitModule::<B>::init(&toy_body_config(), 2, &device);

    // Three names for a two-output head: every score would be attributed to the
    // wrong class, silently, forever.
    let manifest = Manifest::new(
        MiniLmVariant::L6,
        toy_body_config(),
        ClassifierConfig::new(["a", "b", "c"]),
    );

    let err = Bundle::pack(&module, &manifest, &toy_tokenizer_json())
        .expect_err("a label-count mismatch must be refused");
    let text = format!("{err}");
    assert!(
        text.contains('3') && text.contains('2'),
        "unhelpful: {text}"
    );
}

#[test]
fn packing_refuses_a_manifest_that_misdescribes_the_body() {
    let device = Default::default();
    let module = SetFitModule::<B>::init(&toy_body_config(), 2, &device);

    let mut wrong_body = toy_body_config();
    wrong_body.hidden_size *= 2;
    let manifest = Manifest::new(MiniLmVariant::L6, wrong_body, config());

    let err = Bundle::pack(&module, &manifest, &toy_tokenizer_json())
        .expect_err("a body-width mismatch must be refused");
    assert!(format!("{err}").contains("wide"), "unhelpful: {err}");
}

#[test]
fn packing_refuses_a_bundle_with_no_tokenizer() {
    let device = Default::default();
    let module = SetFitModule::<B>::init(&toy_body_config(), 2, &device);
    let manifest = Manifest::new(MiniLmVariant::L6, toy_body_config(), config());

    let err = Bundle::pack(&module, &manifest, &[])
        .expect_err("a bundle that cannot tokenize cannot classify");
    assert!(format!("{err}").contains("tokenizer"), "unhelpful: {err}");
}

#[test]
fn packing_refuses_an_invalid_configuration() {
    let device = Default::default();
    let module = SetFitModule::<B>::init(&toy_body_config(), 2, &device);
    let manifest = Manifest::new(
        MiniLmVariant::L6,
        toy_body_config(),
        ClassifierConfig::new(["a", "b"]).with_reducer(Reducer::TopKMeanLogits { k: 0 }),
    );

    assert!(Bundle::pack(&module, &manifest, &toy_tokenizer_json()).is_err());
}

// ── checkpoint validation ───────────────────────────────────────────────────

#[test]
fn a_checkpoint_without_weights_or_tokenizer_is_refused() {
    let mut no_weights = toy_checkpoint();
    no_weights.weights.clear();
    assert!(format!("{}", no_weights.validate().unwrap_err()).contains("weights"));

    let mut no_tokenizer = toy_checkpoint();
    no_tokenizer.tokenizer_json.clear();
    assert!(format!("{}", no_tokenizer.validate().unwrap_err()).contains("tokenizer"));
}

#[test]
fn a_body_whose_heads_do_not_divide_its_width_is_refused() {
    let mut checkpoint = toy_checkpoint();
    checkpoint.config.num_attention_heads = 5; // 32 is not divisible by 5
    let err = checkpoint.validate().expect_err("indivisible head count");
    assert!(format!("{err}").contains("divisible"), "unhelpful: {err}");
}
