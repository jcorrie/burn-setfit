// Each test binary uses a different subset of these helpers.
#![allow(dead_code)]

//! A tiny hermetic WordPiece tokenizer, so chunker tests need no network.
//!
//! The vocabulary is the single letters `a`–`z`, which makes token counts
//! trivially predictable: the text `"a b c"` is exactly three tokens.

use burn_setfit::checkpoint::Checkpoint;
use burn_setfit::minilm::{MiniLmConfig, MiniLmVariant};
use burn_setfit::tokenize::Tokenizer;

pub fn toy_tokenizer() -> Tokenizer {
    Tokenizer::from_bytes(toy_tokenizer_json_string().as_bytes())
        .expect("toy tokenizer should build")
}

/// The toy tokenizer's `tokenizer.json`.
pub fn toy_tokenizer_json_string() -> String {
    let mut vocab = String::from("\"[PAD]\":0,\"[UNK]\":1,\"[CLS]\":2,\"[SEP]\":3");
    for (i, c) in ('a'..='z').enumerate() {
        vocab.push_str(&format!(",\"{c}\":{}", i + 4));
    }
    // Punctuation is split off by BertPreTokenizer, so it needs vocab entries of
    // its own or it would become [UNK] and distort counts.
    for (i, c) in ['.', '!', '?', ','].iter().enumerate() {
        vocab.push_str(&format!(",\"{c}\":{}", i + 30));
    }

    // `##` cannot appear literally inside an r#".."# literal under Rust 2024.
    let subword = "#".repeat(2);
    let json = format!(
        r#"{{
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [],
          "normalizer": {{"type":"BertNormalizer","clean_text":true,
                          "handle_chinese_chars":true,"strip_accents":null,"lowercase":true}},
          "pre_tokenizer": {{"type":"BertPreTokenizer"}},
          "post_processor": null,
          "decoder": null,
          "model": {{"type":"WordPiece","unk_token":"[UNK]",
                     "continuing_subword_prefix":"{subword}","max_input_chars_per_word":100,
                     "vocab":{{{vocab}}}}}
        }}"#
    );

    json
}

/// `n` space-separated single-letter tokens.
pub fn words(n: usize) -> String {
    (0..n)
        .map(|i| ((b'a' + (i % 26) as u8) as char).to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// A deliberately tiny body config, so end-to-end tests train in seconds.
///
/// Same architecture as MiniLM, three orders of magnitude smaller.
pub fn toy_body_config() -> burn_setfit::minilm::MiniLmConfig {
    burn_setfit::minilm::MiniLmConfig {
        hidden_size: 32,
        num_attention_heads: 2,
        num_hidden_layers: 2,
        intermediate_size: 64,
        vocab_size: 64,
        // Real MiniLM carries 512 against a 256-token window. Keeping that
        // relationship matters: a toy body with fewer positions than the
        // default chunk window is a configuration no real checkpoint has, and
        // testing against it tests the wrong thing.
        max_position_embeddings: 512,
        type_vocab_size: 2,
        hidden_dropout_prob: 0.0,
        layer_norm_eps: 1e-12,
    }
}

/// The tokenizer JSON matching [`toy_tokenizer`], for bundle round-trips.
pub fn toy_tokenizer_json() -> Vec<u8> {
    toy_tokenizer_json_string().into_bytes()
}

/// A complete toy checkpoint: tiny random body, toy tokenizer, Burn naming.
///
/// Lets tests drive the same `Checkpoint`-based API real callers use, rather than
/// a separate construction path that could drift away from it.
pub fn toy_checkpoint() -> Checkpoint {
    use burn::backend::NdArray;
    use burn_store::{ModuleSnapshot, SafetensorsStore};

    let device = Default::default();
    let config = toy_body_config();
    let body = config.init::<NdArray<f32>>(&device);

    let mut store = SafetensorsStore::from_bytes(None);
    body.save_into(&mut store).expect("toy body should save");
    let weights = store.get_bytes().expect("toy weights should serialise");

    Checkpoint::new(MiniLmVariant::L6, config, weights, toy_tokenizer_json()).with_burn_naming()
}

/// The toy body config, for tests that need it directly.
pub fn toy_body_config_of(hidden: usize) -> MiniLmConfig {
    MiniLmConfig {
        hidden_size: hidden,
        ..toy_body_config()
    }
}

/// Hyperparameters that train a toy model in well under a second.
pub fn fast_train_config() -> burn_setfit::train::TrainConfig {
    burn_setfit::train::TrainConfig {
        num_iterations: 4,
        body_epochs: 1,
        head_epochs: 20,
        body_batch_size: 8,
        head_batch_size: 8,
        max_tokens: 32,
        ..Default::default()
    }
}

/// Windows small enough that toy inputs still chunk.
pub fn small_chunking() -> burn_setfit::chunk::ChunkConfig {
    burn_setfit::chunk::ChunkConfig::new(32).with_overlap(4)
}

/// Two classes with disjoint vocabularies, so the task is learnable.
pub fn separable_examples() -> Vec<burn_setfit::train::Example> {
    use burn_setfit::train::Example;
    let class_a = ["a b c", "a c b", "b a c", "c a b", "a b a", "b c a"];
    let class_b = ["x y z", "z y x", "y x z", "x z y", "z z y", "y y x"];
    class_a
        .iter()
        .map(|t| Example::single(*t, 0))
        .chain(class_b.iter().map(|t| Example::single(*t, 1)))
        .collect()
}

/// Train a toy model on [`separable_examples`] and return a packed bundle.
///
/// Memoised by configuration. Toy bodies are randomly initialised from a global
/// RNG, so building "the same" bundle twice yields two different models — and any
/// test that compares results across them is comparing noise. Callers get the same
/// bytes back for the same configuration, within one test binary.
pub fn toy_bundle(classifier: burn_setfit::config::ClassifierConfig) -> Vec<u8> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<HashMap<String, Vec<u8>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    let key = serde_json::to_string(&classifier).expect("config serialises");

    // The lock is held across training deliberately. Releasing it to build and
    // re-taking it to insert lets two threads both miss, both train models with
    // different random initialisations, and one overwrite the other — after which
    // the two threads hold different "identical" bundles. Toy training is fast
    // enough that serialising it costs less than the flakiness did.
    let mut cache = cache.lock().expect("bundle cache is not poisoned");
    if let Some(hit) = cache.get(&key) {
        return hit.clone();
    }
    let built = train_toy_bundle(classifier);
    cache.insert(key, built.clone());
    built
}

/// [`toy_bundle`]'s model, repacked at the given precision.
///
/// Repacked rather than retrained, and that distinction is the whole point.
/// `toy_checkpoint` initialises randomly, so training once per precision would
/// produce a different model each time and any comparison across precisions
/// would be measuring initialisation noise rather than quantization. One model
/// goes in; only the storage differs.
pub fn toy_bundle_quantized(
    classifier: burn_setfit::config::ClassifierConfig,
    quantization: burn_setfit::quantize::Quantization,
) -> Vec<u8> {
    use burn::backend::NdArray;
    use burn_setfit::bundle::Bundle;
    use burn_setfit::quantize::Quantization;

    let plain = toy_bundle(classifier);
    if quantization == Quantization::None {
        return plain;
    }

    let bundle = Bundle::unpack(&plain).expect("the toy bundle unpacks");
    let module = bundle
        .load_module::<NdArray<f32>>(&Default::default())
        .expect("the toy model loads");
    let manifest = bundle.manifest.clone().with_quantization(quantization);
    Bundle::pack(&module, &manifest, &bundle.tokenizer).expect("repacking succeeds")
}

fn train_toy_bundle(classifier: burn_setfit::config::ClassifierConfig) -> Vec<u8> {
    use burn::backend::{Autodiff, NdArray};
    use burn_setfit::train::Trainer;

    let mut trainer = Trainer::<Autodiff<NdArray<f32>>>::new(
        &toy_checkpoint(),
        classifier,
        separable_examples(),
        fast_train_config(),
        Default::default(),
    )
    .expect("toy training data is well formed");
    trainer.fit_with(|_| {}).expect("toy training completes");
    trainer.finish().expect("a completed run packs")
}
