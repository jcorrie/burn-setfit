//! Why long-document classification needs a head that can abstain.
//!
//! Trains the same few-shot data two ways and runs both over a long document
//! containing exactly one relevant paragraph, plus a filler-only control. The
//! control is the whole point: if a document without the signal scores the same
//! as one with it, the classifier is reporting the filler, not the signal.
//!
//! Run with: `cargo run --release --example long_document`
use burn_setfit::bundle::Bundle;
use burn_setfit::checkpoint::Checkpoint;
use burn_setfit::chunk::ChunkConfig;
use burn_setfit::config::ClassifierConfig;
use burn_setfit::head::TaskMode;
use burn_setfit::infer::{Classifier, decode_against_background};
use burn_setfit::minilm::MiniLmVariant;
use burn_setfit::reduce::Reducer;
use burn_setfit::train::{Example, TrainConfig, Trainer};

const FILLER: &str = "The quarterly review covered headcount, office moves, and the \
                      updated travel policy. Attendance was steady and the agenda ran \
                      to time. Notes were circulated afterwards. ";

fn training_texts() -> Vec<(&'static str, usize)> {
    vec![
        ("I was charged twice for last month's subscription.", 0),
        ("My invoice shows an amount I don't recognise.", 0),
        ("Can I get a refund for the duplicate payment?", 0),
        ("The card on file was declined but you still billed me.", 0),
        ("Why did my monthly price go up without warning?", 0),
        ("Please cancel my plan and refund the last charge.", 0),
        ("The receipt total doesn't match what I agreed to pay.", 0),
        ("I need a VAT invoice for accounting.", 0),
        ("The site has been returning 503 errors all morning.", 1),
        ("Nothing loads, the dashboard just spins forever.", 1),
        ("API requests are timing out across all our regions.", 1),
        ("Is there an incident? Everything is down for our team.", 1),
        ("We're seeing connection refused on every endpoint.", 1),
        ("The service went offline about twenty minutes ago.", 1),
        ("Login is broken, users can't authenticate at all.", 1),
        ("Latency spiked and now requests fail outright.", 1),
        ("It would be great if you supported dark mode.", 2),
        ("Could you add an option to export as CSV?", 2),
        ("Please consider adding keyboard shortcuts.", 2),
        ("I'd love to see a bulk edit tool in a future release.", 2),
        ("Any chance of an API for scheduled reports?", 2),
        ("A mobile app would make this much more useful.", 2),
        ("Suggestion: let us tag items with custom labels.", 2),
        ("Can we get webhooks for status changes?", 2),
    ]
}

/// Background examples, so the model has somewhere to put irrelevant text.
fn background() -> Vec<&'static str> {
    vec![
        "The quarterly review covered headcount and office moves.",
        "Attendance was steady and the agenda ran to time.",
        "Notes from the meeting were circulated afterwards.",
        "The travel policy was updated earlier this year.",
        "Room bookings should be made through the usual system.",
        "Please remember to submit your timesheet by Friday.",
        "The offsite is scheduled for the second week of June.",
        "Parking permits are available from reception.",
    ]
}

/// Train on the given examples and return a packed bundle.
fn train(
    checkpoint: &Checkpoint,
    classifier: ClassifierConfig,
    examples: Vec<Example>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut trainer = Trainer::new(
        checkpoint,
        classifier,
        examples,
        TrainConfig {
            num_iterations: 10,
            head_epochs: 40,
            ..Default::default()
        },
        Default::default(),
    )?;
    trainer.fit_with(|_| {})?;
    Ok(trainer.finish()?)
}

/// The same trained model, repacked to read documents in `max_tokens` windows.
///
/// Windowing happens only at inference, so this changes nothing the model
/// learned, only how much filler shares a window with the signal.
fn with_windows(bundle: &[u8], max_tokens: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let unpacked = Bundle::unpack(bundle)?;
    let module = unpacked.load_module(&Default::default())?;
    let mut manifest = unpacked.manifest.clone();
    manifest.classifier.chunk = ChunkConfig::new(max_tokens).with_overlap(max_tokens / 8);
    Ok(Bundle::pack(&module, &manifest, &unpacked.tokenizer)?)
}

/// Score a document and its control, and report whether they can be told apart.
fn evaluate(
    name: &str,
    bundle: &[u8],
    reducer: Reducer,
    fanout: Option<usize>,
    signal_doc: &str,
    control_doc: &str,
    background: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut classifier = Classifier::from_bundle(bundle, Default::default())?.with_reducer(reducer);
    if let Some(f) = fanout {
        classifier = classifier.with_hierarchy(f);
    }

    let with_signal = classifier.classify(signal_doc)?;
    let control = classifier.classify(control_doc)?;

    // "billing" is label 0 in every configuration here.
    let signal_score = with_signal.scores[0];
    let control_score = control.scores[0];

    println!("\n{name}");
    println!("  billing score, document containing a billing complaint: {signal_score:.3}");
    println!("  billing score, filler only (control):                   {control_score:.3}");
    println!(
        "  separation:                                             {:+.3}",
        signal_score - control_score
    );
    let task = classifier.manifest().task();
    let names = classifier.labels();
    let decode = |scores: &[f32]| -> Vec<&str> {
        decode_against_background(scores, task, background)
            .into_iter()
            .map(|i| names[i].as_str())
            .collect()
    };
    println!(
        "  predicted: {:?}   control predicted: {:?}",
        decode(&with_signal.scores),
        decode(&control.scores)
    );

    // Calibration (#12): which multi-label thresholds would get both documents
    // right — billing for the signal document, not for the control? With a
    // background class, billing must also beat it.
    if let TaskMode::MultiLabel { threshold } = task {
        let (s, c) = (signal_score, control_score);
        let floors = background.map(|b| (with_signal.scores[b], control.scores[b]));
        if let Some((sf, cf)) = floors {
            println!("  background score, signal / control:                     {sf:.3} / {cf:.3}");
        }
        let window = match floors {
            Some((sf, _)) if s <= sf => None,
            Some((_, cf)) if c <= cf => Some((0.0, s)),
            _ if s > c => Some((c, s)),
            _ => None,
        };
        match window {
            Some((lo, hi)) => println!(
                "  thresholds that get both right: above {lo:.3}, up to {hi:.3} (current {threshold})"
            ),
            None => println!("  no threshold gets both right (current {threshold})"),
        }
    }

    // With a background class, also decode each chunk on its own. A
    // document-level comparison pits billing's best chunk against the
    // background's best chunk, and a long document is mostly filler, so the
    // background's best is high in every document. Per chunk, billing only
    // has to beat the background where the billing sentence actually is.
    if background.is_some() {
        let votes = |doc: &str| -> Result<(usize, usize), Box<dyn std::error::Error>> {
            let chunks = classifier.classify_chunks(doc)?;
            let billing = chunks
                .iter()
                .filter(|c| decode_against_background(&c.scores, task, background).contains(&0))
                .count();
            Ok((billing, chunks.len()))
        };
        let (sb, sn) = votes(signal_doc)?;
        let (cb, cn) = votes(control_doc)?;
        println!("  chunks that decode to billing, signal / control:        {sb}/{sn} / {cb}/{cn}");
        // The direct test of per-chunk competition: in each document's most
        // billing-like chunk, does billing beat the background?
        let b = background.unwrap_or(0);
        let strongest = |doc: &str| -> Result<(f32, f32), Box<dyn std::error::Error>> {
            let chunks = classifier.classify_chunks(doc)?;
            let best = chunks
                .iter()
                .max_by(|x, y| x.scores[0].total_cmp(&y.scores[0]))
                .ok_or("no chunks")?;
            Ok((best.scores[0], best.scores[b]))
        };
        let (s_bill, s_bg) = strongest(signal_doc)?;
        let (c_bill, c_bg) = strongest(control_doc)?;
        println!(
            "  most billing-like chunk, billing vs background:         signal {s_bill:.3} vs {s_bg:.3}, control {c_bill:.3} vs {c_bg:.3}"
        );
    }

    // A chunk is ~256 tokens, so roughly a thousand characters: report whether
    // the winning chunk actually contains the planted signal rather than
    // printing its first line and guessing.
    if let Some(e) = with_signal.evidence.first() {
        let text = &signal_doc[e.byte_range.clone()];
        println!(
            "  strongest supporting chunk ({:.2}, {} bytes) contains the billing sentence: {}",
            e.score,
            text.len(),
            text.contains("billed twice")
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("loading {} ...", MiniLmVariant::L6.model_id());
    let checkpoint = Checkpoint::download(MiniLmVariant::L6, None)?;

    let signal_doc = format!(
        "{}Separately, we were billed twice for the same subscription period and \
         need a refund for the duplicate charge. {}",
        FILLER.repeat(40),
        FILLER.repeat(40)
    );
    let control_doc = FILLER.repeat(80);

    let three = ["billing", "outage", "feature"];

    eprintln!("training single-label ...");
    let single = train(
        &checkpoint,
        ClassifierConfig::new(three),
        training_texts()
            .into_iter()
            .map(|(t, l)| Example::single(t, l))
            .collect(),
    )?;
    evaluate(
        "SINGLE-LABEL, MeanLogits",
        &single,
        Reducer::MeanLogits,
        None,
        &signal_doc,
        &control_doc,
        None,
    )?;

    eprintln!("training multi-label ...");
    let multi = train(
        &checkpoint,
        ClassifierConfig::new(three).multi_label(),
        training_texts()
            .into_iter()
            .map(|(t, l)| Example::multi(t, vec![l]))
            .collect(),
    )?;
    for (name, reducer, fanout) in [
        ("MULTI-LABEL, NoisyOr", Reducer::NoisyOr, None),
        ("MULTI-LABEL, MaxLogits", Reducer::MaxLogits, None),
        (
            "MULTI-LABEL, NoisyOr within blocks of 8, averaged across",
            Reducer::NoisyOr,
            Some(8),
        ),
    ] {
        evaluate(
            name,
            &multi,
            reducer,
            fanout,
            &signal_doc,
            &control_doc,
            None,
        )?;
    }

    eprintln!("training multi-label with a background class ...");
    let mut bg_examples: Vec<Example> = training_texts()
        .into_iter()
        .map(|(t, l)| Example::multi(t, vec![l]))
        .collect();
    bg_examples.extend(background().into_iter().map(|t| Example::multi(t, vec![3])));
    let bg = train(
        &checkpoint,
        ClassifierConfig::new(["billing", "outage", "feature", "other"]).multi_label(),
        bg_examples,
    )?;
    evaluate(
        "MULTI-LABEL + background class, MaxLogits",
        &bg,
        Reducer::MaxLogits,
        None,
        &signal_doc,
        &control_doc,
        None,
    )?;

    // The same model and the same scores, decoded differently. Above, "other"
    // is a label like any other and the verdict is still "is billing above the
    // threshold?". Decoding against it as the background class changes the
    // question to "is billing above the threshold *and* above other?". One
    // model, so the two rows differ only in decoding (training it twice would
    // compare two models).
    evaluate(
        "MULTI-LABEL + background class DECODED AS ONE, MaxLogits",
        &bg,
        Reducer::MaxLogits,
        None,
        &signal_doc,
        &control_doc,
        Some(3),
    )?;

    // Is the background class failing because the signal is diluted within
    // its window? Same model, smaller windows: less filler per window.
    for tokens in [128, 64] {
        let narrow = with_windows(&bg, tokens)?;
        evaluate(
            &format!(
                "MULTI-LABEL + background class DECODED AS ONE, MaxLogits, {tokens}-token windows"
            ),
            &narrow,
            Reducer::MaxLogits,
            None,
            &signal_doc,
            &control_doc,
            Some(3),
        )?;
    }
    let narrow = with_windows(&multi, 64)?;
    evaluate(
        "MULTI-LABEL, MaxLogits, 64-token windows",
        &narrow,
        Reducer::MaxLogits,
        None,
        &signal_doc,
        &control_doc,
        None,
    )?;

    Ok(())
}
