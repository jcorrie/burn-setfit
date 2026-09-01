//! What quantization actually costs, measured on the real checkpoint.
//!
//! Issue #2 asks for size and accuracy deviation to be *documented*, not
//! guessed, and the toy model in `tests/quantize.rs` cannot answer it: its
//! 32-wide body is mostly LayerNorm and bias, which quantization deliberately
//! skips. MiniLM-L6 is the opposite — roughly half its 22.7M parameters are one
//! `30522 x 384` embedding table — so the ratios only mean something here.
//!
//! Trains one model, packs it three ways, and reports what changed. One model
//! is the point: repacking rather than retraining is what makes the columns
//! comparable.
//!
//! Two modes:
//!
//! ```bash
//! cargo run --release --example quantize              # the real thing; downloads the checkpoint
//! cargo run --release --example quantize -- --sizes   # sizes only, offline
//! ```
//!
//! `--sizes` exists because bundle size is decided entirely by tensor *shapes*,
//! so a randomly-initialised body with MiniLM-L6's exact architecture gives
//! byte-for-byte the same answer as the trained one. That makes the size column
//! measurable without a 90 MB download, or anywhere the HuggingFace CDN is not
//! reachable. It says nothing whatever about accuracy — random weights cannot —
//! so the deviation columns are omitted rather than filled with a number that
//! would look like a measurement.

use burn::backend::{Autodiff, NdArray};
use burn_setfit::bundle::Bundle;
use burn_setfit::checkpoint::Checkpoint;
use burn_setfit::config::ClassifierConfig;
use burn_setfit::infer::Classifier;
use burn_setfit::minilm::MiniLmVariant;
use burn_setfit::quantize::Quantization;
use burn_setfit::train::{Example, TrainConfig, Trainer};

type B = NdArray<f32>;
type AB = Autodiff<B>;

/// Held-out probes: not training data, so a wrong answer here is a real
/// regression rather than a memorised one.
const PROBES: &[(&str, usize)] = &[
    ("My credit card was billed twice this cycle.", 0),
    ("Everything is returning 500s right now.", 1),
    ("Please add a way to export to CSV.", 2),
    ("The invoice total is wrong again.", 0),
    ("The whole platform seems to be offline.", 1),
    ("Could you support keyboard shortcuts?", 2),
];

/// MiniLM-L6-v2's architecture, as published.
///
/// Only the shapes matter here, and they are what decide a bundle's size: the
/// `30522 x 384` word-embedding table alone is half the parameters.
fn minilm_l6_config() -> burn_setfit::minilm::MiniLmConfig {
    burn_setfit::minilm::MiniLmConfig {
        hidden_size: 384,
        num_attention_heads: 12,
        num_hidden_layers: 6,
        intermediate_size: 1536,
        vocab_size: 30522,
        max_position_embeddings: 512,
        type_vocab_size: 2,
        hidden_dropout_prob: 0.1,
        layer_norm_eps: 1e-12,
    }
}

/// A tokenizer just valid enough to pack beside the weights.
///
/// Its size is identical in every bundle, so it cannot affect the ratios; a
/// bundle simply refuses to exist without one.
fn placeholder_tokenizer() -> Vec<u8> {
    let subword = "#".repeat(2);
    format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
           "normalizer":{{"type":"BertNormalizer","clean_text":true,
                         "handle_chinese_chars":true,"strip_accents":null,"lowercase":true}},
           "pre_tokenizer":{{"type":"BertPreTokenizer"}},"post_processor":null,"decoder":null,
           "model":{{"type":"WordPiece","unk_token":"[UNK]",
                    "continuing_subword_prefix":"{subword}","max_input_chars_per_word":100,
                    "vocab":{{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3}}}}}}"#
    )
    .into_bytes()
}

/// Size only, from real shapes and random values.
fn report_sizes() -> Result<(), Box<dyn std::error::Error>> {
    use burn_setfit::bundle::Manifest;
    use burn_setfit::model::SetFitModule;

    let config = minilm_l6_config();
    let labels = ["billing", "outage", "feature"];
    let module = SetFitModule::<B>::init(&config, labels.len(), &Default::default());
    let manifest = Manifest::new(
        MiniLmVariant::L6,
        config,
        ClassifierConfig::new(labels.iter().map(|s| s.to_string()).collect::<Vec<_>>()),
    );
    let tokenizer = placeholder_tokenizer();

    eprintln!(
        "MiniLM-L6 shapes, random weights — sizes are exact, accuracy is not measured.
"
    );
    println!("| Mode | Bundle | vs fp32 |");
    println!("| ---- | ------ | ------- |");

    let baseline = Bundle::pack(&module, &manifest, &tokenizer)?.len() as f64;
    for mode in [Quantization::None, Quantization::F16, Quantization::Int8] {
        let bytes = Bundle::pack(
            &module,
            &manifest.clone().with_quantization(mode),
            &tokenizer,
        )?;
        println!(
            "| `{}` | {:.1} MB | {:.0}% |",
            mode.as_str(),
            bytes.len() as f64 / 1_048_576.0,
            100.0 * bytes.len() as f64 / baseline
        );
    }
    println!();
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|a| a == "--sizes") {
        return report_sizes();
    }

    let device = Default::default();
    let labels: Vec<String> = ["billing", "outage", "feature"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let examples = vec![
        Example::single("I was charged twice for last month's subscription.", 0),
        Example::single("My invoice shows an amount I don't recognise.", 0),
        Example::single("Can I get a refund for the duplicate payment?", 0),
        Example::single("The card on file was declined but you still billed me.", 0),
        Example::single("Why did my monthly price go up without warning?", 0),
        Example::single("Please cancel my plan and refund the last charge.", 0),
        Example::single("The site has been returning 503 errors all morning.", 1),
        Example::single("Nothing loads, the dashboard just spins forever.", 1),
        Example::single("API requests are timing out across all our regions.", 1),
        Example::single("Is there an incident? Everything is down for our team.", 1),
        Example::single("We're seeing connection refused on every endpoint.", 1),
        Example::single("The service went offline about twenty minutes ago.", 1),
        Example::single("It would be great if you supported dark mode.", 2),
        Example::single("Could you add an option to export as CSV?", 2),
        Example::single("Please consider adding keyboard shortcuts.", 2),
        Example::single("I'd love to see a bulk edit tool in a future release.", 2),
        Example::single("Any chance of an API for scheduled reports?", 2),
        Example::single("A mobile app would make this much more useful.", 2),
    ];

    eprintln!("loading {} ...", MiniLmVariant::L6.model_id());
    let checkpoint = Checkpoint::download(MiniLmVariant::L6, None)?;

    let mut trainer = Trainer::<AB>::new(
        &checkpoint,
        ClassifierConfig::new(labels.clone()),
        examples,
        TrainConfig {
            num_iterations: 10,
            head_epochs: 40,
            ..Default::default()
        },
        device,
    )?;
    eprintln!("training: {} steps", trainer.total_steps());
    trainer.fit_with(|_| {})?;

    // One trained model, packed three ways. Retraining per mode would compare
    // three different models and call the difference quantization error.
    let plain = trainer.finish()?;
    let base = Bundle::unpack(&plain)?;
    let module = base.load_module::<B>(&Default::default())?;

    let pack = |q: Quantization| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let manifest = base.manifest.clone().with_quantization(q);
        Ok(Bundle::pack(&module, &manifest, &base.tokenizer)?)
    };

    let modes = [
        (Quantization::None, plain.clone()),
        (Quantization::F16, pack(Quantization::F16)?),
        (Quantization::Int8, pack(Quantization::Int8)?),
    ];

    // The fp32 model is the baseline every deviation is measured against.
    let reference = Classifier::<B>::from_bundle(&plain, Default::default())?;
    let probe_texts: Vec<&str> = PROBES.iter().map(|(t, _)| *t).collect();
    let reference_vectors = reference.embed(&probe_texts)?;

    println!();
    println!("| Mode | Bundle | vs fp32 | Max score drift | Max cosine drift | Probes correct |");
    println!("| ---- | ------ | ------- | --------------- | ---------------- | -------------- |");

    let baseline_len = plain.len() as f64;
    for (mode, bytes) in &modes {
        let classifier = Classifier::<B>::from_bundle(bytes, Default::default())?;

        let mut worst_score = 0.0f32;
        let mut correct = 0usize;
        for (text, want) in PROBES {
            let got = classifier.classify(text)?;
            let reference_scores = reference.classify(text)?.scores;
            for (a, b) in reference_scores.iter().zip(&got.scores) {
                worst_score = worst_score.max((a - b).abs());
            }
            if got.predicted.first() == Some(want) {
                correct += 1;
            }
        }

        // Cosine against the fp32 embedding is the backend-agnostic measure of
        // how far the body moved, independent of what the head then does.
        let vectors = classifier.embed(&probe_texts)?;
        let mut worst_cosine = 0.0f32;
        for (a, b) in reference_vectors.iter().zip(&vectors) {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            worst_cosine = worst_cosine.max(1.0 - dot);
        }

        let mb = bytes.len() as f64 / 1_048_576.0;
        let pct = 100.0 * bytes.len() as f64 / baseline_len;
        println!(
            "| `{}` | {:.1} MB | {:.0}% | {:.2e} | {:.2e} | {}/{} |",
            mode.as_str(),
            mb,
            pct,
            worst_score,
            worst_cosine,
            correct,
            PROBES.len()
        );
    }
    println!();

    Ok(())
}
