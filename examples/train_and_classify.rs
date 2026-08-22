//! Few-shot training on the real checkpoint, then classifying a long document.
//!
//! Eight examples per class, no labelled data beyond that. Downloads
//! `all-MiniLM-L6-v2` on first run.
//!
//! Run with: `cargo run --release --example train_and_classify`

use burn::backend::{Autodiff, NdArray};
use burn_setfit::checkpoint::Checkpoint;
use burn_setfit::config::ClassifierConfig;
use burn_setfit::infer::Classifier;
use burn_setfit::minilm::MiniLmVariant;
use burn_setfit::reduce::Reducer;
use burn_setfit::train::{Example, TrainConfig, Trainer};

type B = NdArray<f32>;
type AB = Autodiff<B>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    let labels = vec![
        "billing".to_string(),
        "outage".to_string(),
        "feature".to_string(),
    ];

    // Eight examples per class — the regime SetFit is designed for.
    let examples = vec![
        Example::single("I was charged twice for last month's subscription.", 0),
        Example::single("My invoice shows an amount I don't recognise.", 0),
        Example::single("Can I get a refund for the duplicate payment?", 0),
        Example::single("The card on file was declined but you still billed me.", 0),
        Example::single("Why did my monthly price go up without warning?", 0),
        Example::single("Please cancel my plan and refund the last charge.", 0),
        Example::single("The receipt total doesn't match what I agreed to pay.", 0),
        Example::single("I need a VAT invoice for accounting.", 0),
        Example::single("The site has been returning 503 errors all morning.", 1),
        Example::single("Nothing loads, the dashboard just spins forever.", 1),
        Example::single("API requests are timing out across all our regions.", 1),
        Example::single("Is there an incident? Everything is down for our team.", 1),
        Example::single("We're seeing connection refused on every endpoint.", 1),
        Example::single("The service went offline about twenty minutes ago.", 1),
        Example::single("Login is broken, users can't authenticate at all.", 1),
        Example::single("Latency spiked and now requests fail outright.", 1),
        Example::single("It would be great if you supported dark mode.", 2),
        Example::single("Could you add an option to export as CSV?", 2),
        Example::single("Please consider adding keyboard shortcuts.", 2),
        Example::single("I'd love to see a bulk edit tool in a future release.", 2),
        Example::single("Any chance of an API for scheduled reports?", 2),
        Example::single("A mobile app would make this much more useful.", 2),
        Example::single("Suggestion: let us tag items with custom labels.", 2),
        Example::single("Can we get webhooks for status changes?", 2),
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
    let started = std::time::Instant::now();
    let mut last_stage = None;
    trainer.fit_with(|p| {
        if last_stage != Some(p.stage) {
            eprintln!("  {:?} stage", p.stage);
            last_stage = Some(p.stage);
        }
        if p.step % 20 == 0 {
            eprintln!("  step {:>4}/{}  loss {:.4}", p.step, p.total_steps, p.loss);
        }
    })?;
    eprintln!("trained in {:.1}s\n", started.elapsed().as_secs_f32());

    // The trainer already knows the checkpoint, the label set and the task, so a
    // finished bundle needs no further arguments — and cannot be handed metadata
    // that disagrees with what was actually trained.
    let packed = trainer.finish()?;
    eprintln!("bundle: {:.1} MB\n", packed.len() as f64 / 1_048_576.0);

    let classifier = Classifier::<B>::from_bundle(&packed, Default::default())?;

    println!("short inputs held out from training:");
    for text in [
        "You billed my card three times this week.",
        "Everything is 500ing, our whole team is blocked.",
        "Would you ever add a Slack integration?",
    ] {
        let p = classifier.classify(text)?;
        let (i, score) = p.top().unwrap();
        println!("  {:<10} {:.3}   {text}", labels[i], score);
    }

    // A long document: one relevant paragraph buried in unrelated filler.
    let filler = "The quarterly review covered headcount, office moves, and the \
                  updated travel policy. Attendance was steady and the agenda ran \
                  to time. Notes were circulated afterwards. ";
    let mut long_doc = String::new();
    for _ in 0..40 {
        long_doc.push_str(filler);
    }
    long_doc.push_str(
        "Separately, we were billed twice for the same subscription period and \
         need a refund for the duplicate charge. ",
    );
    for _ in 0..40 {
        long_doc.push_str(filler);
    }

    // Control: the same document with the relevant paragraph removed. If this
    // scores the same, the verdict was never driven by the billing paragraph.
    let control: String = filler.repeat(80);

    println!(
        "\nlong document ({} bytes, one relevant paragraph):",
        long_doc.len()
    );
    for (name, reducer) in [
        ("MeanLogits", Reducer::MeanLogits),
        ("MaxLogits", Reducer::MaxLogits),
        ("TopKMean(3)", Reducer::TopKMeanLogits { k: 3 }),
    ] {
        // Reduction is a decode-time choice, so it can be varied without
        // retraining or repacking anything.
        let c = Classifier::<B>::from_bundle(&packed, Default::default())?.with_reducer(reducer);
        let p = c.classify(&long_doc)?;
        let (i, score) = p.top().unwrap();
        println!(
            "  {name:<12} {} chunks -> {:<10} {:.3}",
            p.chunks_seen, labels[i], score
        );
        if let Some(e) = p.evidence.first() {
            let excerpt: String = long_doc[e.byte_range.clone()].chars().take(70).collect();
            println!(
                "               strongest chunk ({:.2}): {excerpt}...",
                e.score
            );
        }

        let cp = c.classify(&control)?;
        let (ci, cscore) = cp.top().unwrap();
        println!(
            "               control (filler only)  -> {:<10} {:.3}",
            labels[ci], cscore
        );
    }

    Ok(())
}
