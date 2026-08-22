//! Training from a checkpoint on disk, without touching the network.
//!
//! [`Checkpoint::download`] is a convenience, not the interface. Underneath it,
//! a checkpoint is three files handed over as bytes — which is what makes the
//! browser path and the native path the same code. This example uses that entry
//! point directly, so it works behind a proxy, from an internal mirror, or from
//! a HuggingFace snapshot already on disk.
//!
//! Point it at a directory holding `config.json`, `model.safetensors` and
//! `tokenizer.json`:
//!
//! ```text
//! cargo run --release --example local_checkpoint -- ~/models/all-MiniLM-L6-v2
//! ```
//!
//! The toy checkpoint the browser harness generates works too, and is the
//! quickest way to see the whole pipeline run:
//!
//! ```text
//! python3 browser-test/make_toy_checkpoint.py /tmp/toy
//! cargo run --release --example local_checkpoint -- /tmp/toy
//! ```
//!
//! Note what that does *not* show you. The toy body is 32 wide with a
//! vocabulary of single letters, so every word below tokenizes to `[UNK]` and
//! the scores mean nothing. It exercises the pipeline, not the model.

use burn::backend::{Autodiff, NdArray};
use burn_setfit::{
    Checkpoint, Classifier, ClassifierConfig, Example, MiniLmVariant, Reducer, TrainConfig, Trainer,
};
use std::path::{Path, PathBuf};

type B = NdArray<f32>;
type AB = Autodiff<B>;

/// Read the three files a checkpoint is made of.
fn load(dir: &Path) -> Result<Checkpoint, Box<dyn std::error::Error>> {
    let config_json = std::fs::read_to_string(dir.join("config.json"))?;
    let weights = std::fs::read(dir.join("model.safetensors"))?;
    let tokenizer_json = std::fs::read(dir.join("tokenizer.json"))?;

    // `from_files` assumes HuggingFace naming and PyTorch's transposed Linear
    // layout, which is what a published checkpoint carries. Weights this crate
    // wrote itself need `Checkpoint::with_burn_naming` instead — running the
    // adapter over them would turn correct weights into wrong ones.
    Ok(Checkpoint::from_files(
        MiniLmVariant::L6,
        &config_json,
        weights,
        tokenizer_json,
    )?)
}

fn training_data() -> Vec<Example> {
    let rows: Vec<(&str, usize)> = vec![
        ("I was charged twice for last month's subscription.", 0),
        ("My invoice shows an amount I don't recognise.", 0),
        ("Can I get a refund for the duplicate payment?", 0),
        ("The card on file was declined but you still billed me.", 0),
        ("Why did my monthly price go up without warning?", 0),
        ("The receipt total doesn't match what I agreed to pay.", 0),
        ("The site has been returning 503 errors all morning.", 1),
        ("Nothing loads, the dashboard just spins forever.", 1),
        ("API requests are timing out across all our regions.", 1),
        ("Is there an incident? Everything is down for our team.", 1),
        ("We're seeing connection refused on every endpoint.", 1),
        ("Login is broken, users can't authenticate at all.", 1),
    ];
    rows.into_iter()
        .map(|(text, label)| Example::single(text, label))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(dir) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!(
            "usage: cargo run --release --example local_checkpoint -- <checkpoint-dir>\n\n\
             The directory needs config.json, model.safetensors and tokenizer.json.\n\
             Generate a toy one with: python3 browser-test/make_toy_checkpoint.py /tmp/toy"
        );
        std::process::exit(2);
    };

    let checkpoint = load(&dir)?;
    println!(
        "loaded {}: {} layers, {} wide, {} positions",
        dir.display(),
        checkpoint.config.num_hidden_layers,
        checkpoint.config.hidden_size,
        checkpoint.config.max_position_embeddings,
    );

    let labels = ["billing", "outage"];
    let mut trainer = Trainer::<AB>::new(
        &checkpoint,
        ClassifierConfig::new(labels),
        training_data(),
        TrainConfig {
            num_iterations: 10,
            head_epochs: 40,
            ..Default::default()
        },
        Default::default(),
    )?;

    let started = std::time::Instant::now();
    let total = trainer.total_steps();
    trainer.fit_with(|p| {
        if p.step % 10 == 0 || p.step == total {
            eprint!("\r{:?} {}/{} loss {:.4}   ", p.stage, p.step, total, p.loss);
        }
    })?;
    eprintln!(
        "\rtrained in {:.1}s{:20}",
        started.elapsed().as_secs_f32(),
        ""
    );

    let bundle = trainer.finish()?;
    println!("bundle: {:.1} MB", bundle.len() as f64 / 1_048_576.0);

    let classifier = Classifier::<B>::from_bundle(&bundle, Default::default())?;
    println!("\nheld out from training:");
    for text in [
        "You billed my card three times this week.",
        "Everything is 500ing, our whole team is blocked.",
    ] {
        let p = classifier.classify(text)?;
        let (i, score) = p.top().expect("a trained model always scores something");
        println!("  {:<8} {score:.3}   {text}", labels[i]);
    }

    // A document too long for one window. `MaxLogits` rather than the
    // single-label default because one paragraph in many carries the signal --
    // see the long_document example for the measurements behind that choice.
    let filler = "The quarterly review covered headcount, office moves, and the \
                  updated travel policy. Attendance was steady. ";
    let document = format!(
        "{}Separately, we were billed twice for the same subscription period.  {}",
        filler.repeat(20),
        filler.repeat(20)
    );

    let long = classifier
        .with_reducer(Reducer::MaxLogits)
        .classify(&document)?;
    let (i, score) = long.top().expect("a trained model always scores something");
    println!(
        "\nlong document ({} bytes, {} chunks): {} {score:.3}",
        document.len(),
        long.chunks_seen,
        labels[i]
    );

    // Evidence carries byte ranges into the document classified, so a verdict
    // can be shown its source rather than asserted.
    if let Some(e) = long.evidence.first() {
        let excerpt: String = document[e.byte_range.clone()].chars().take(72).collect();
        println!("  strongest chunk ({:.3}): {excerpt}...", e.score);
    }

    Ok(())
}
