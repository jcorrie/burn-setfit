//! Where the time goes, stage by stage, on MiniLM-L6's real shapes.
//!
//! Every cost measured here depends on tensor *shapes* and token counts, never
//! on the weights' values, so a randomly-initialised body with MiniLM-L6's exact
//! architecture times the same as the published one. That makes this runnable
//! offline, where `huggingface.co` is unreachable, and the numbers transfer.
//! (It says nothing about accuracy, and does not try to.)
//!
//! The tokenizer is synthetic too: a WordPiece vocabulary built from the
//! generated corpus, so documents tokenize to roughly one token per word, as
//! English does with the real vocabulary. Chunk and token counts are printed
//! beside each timing so a reader can tell what was actually encoded.
//!
//! ```bash
//! cargo run --release --example bench                  # ndarray, every stage
//! cargo run --release --example bench -- infer         # only stages whose name contains "infer"
//! cargo run --release --example bench --features wgpu -- --wgpu
//! cargo run --release --example bench --features flex -- --flex
//! ```
//!
//! On a Mac, `--features wgpu -- --wgpu` runs on Metal, the backend a browser
//! there reaches through WebGPU. Timings are a median of several runs after a
//! warm-up, and on a GPU backend each one ends in a readback, so they include
//! the work rather than just its submission.

use burn::tensor::Device;
use burn_setfit::bundle::{Bundle, Manifest};
use burn_setfit::checkpoint::Checkpoint;
use burn_setfit::chunk::{ChunkConfig, Chunker};
use burn_setfit::config::ClassifierConfig;
use burn_setfit::infer::Classifier;
use burn_setfit::minilm::{MiniLmConfig, MiniLmVariant};
use burn_setfit::model::SetFitModule;
use burn_setfit::quantize::Quantization;
use burn_setfit::tokenize::{Tokenizer, pad_batch};
use burn_setfit::train::{Example, Stage, TrainConfig, Trainer};
use std::time::{Duration, Instant};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// One stage of the benchmark, selectable by name on the command line.
type Bench = fn(&Fixture) -> Res<()>;

/// MiniLM-L6-v2's architecture, as published.
fn minilm_l6_config() -> MiniLmConfig {
    MiniLmConfig {
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

/// A deterministic xorshift, so every run encodes the same text.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A fixed lexicon of pronounceable pseudo-words.
fn lexicon(n: usize) -> Vec<String> {
    let (c, v) = (b"bcdfghklmnprstvz", b"aeiou");
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut words: Vec<String> = Vec::with_capacity(n);
    while words.len() < n {
        let syllables = 1 + rng.below(3);
        let w: String = (0..syllables)
            .flat_map(|_| [c[rng.below(c.len())] as char, v[rng.below(v.len())] as char])
            .collect();
        if !words.contains(&w) {
            words.push(w);
        }
    }
    words
}

/// Prose-shaped text: sentences of 6–24 words, paragraphs of 3–8 sentences.
fn document(words: &[String], approx_words: usize, seed: u64) -> String {
    let mut rng = Rng(seed | 1);
    let mut out = String::new();
    let mut written = 0;
    while written < approx_words {
        for _ in 0..3 + rng.below(6) {
            let len = 6 + rng.below(19);
            for i in 0..len {
                // Squaring skews towards the front of the lexicon, roughly Zipfian.
                let r = rng.below(words.len());
                let w = &words[r * r / words.len()];
                if i > 0 {
                    out.push(' ');
                }
                out.push_str(w);
            }
            out.push_str(". ");
            written += len;
        }
        out.push_str("\n\n");
    }
    out
}

/// One sentence of 8–16 words: the shape of a few-shot training example.
fn sentence(words: &[String], seed: u64) -> String {
    let mut rng = Rng(seed | 1);
    let len = 8 + rng.below(9);
    let mut out: Vec<&str> = (0..len)
        .map(|_| {
            let r = rng.below(words.len());
            words[r * r / words.len()].as_str()
        })
        .collect();
    out.push(".");
    out.join(" ")
}

fn tokenizer_json(words: &[String]) -> Vec<u8> {
    let mut vocab = vec![
        "[PAD]".to_string(),
        "[UNK]".into(),
        "[CLS]".into(),
        "[SEP]".into(),
        ".".into(),
    ];
    for ch in 'a'..='z' {
        vocab.push(ch.to_string());
        vocab.push(format!("##{ch}"));
    }
    vocab.extend(words.iter().cloned());
    let entries: Vec<String> = vocab
        .iter()
        .enumerate()
        .map(|(i, t)| format!("\"{t}\":{i}"))
        .collect();
    format!(
        r###"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
           "normalizer":{{"type":"BertNormalizer","clean_text":true,
                         "handle_chinese_chars":true,"strip_accents":null,"lowercase":true}},
           "pre_tokenizer":{{"type":"BertPreTokenizer"}},"post_processor":null,"decoder":null,
           "model":{{"type":"WordPiece","unk_token":"[UNK]",
                    "continuing_subword_prefix":"##","max_input_chars_per_word":100,
                    "vocab":{{{}}}}}}}"###,
        entries.join(",")
    )
    .into_bytes()
}

/// Run `f` once to warm up, then `runs` times; report the median.
fn time<T>(runs: usize, mut f: impl FnMut() -> T) -> Duration {
    std::hint::black_box(f());
    let mut samples: Vec<Duration> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(f());
            t.elapsed()
        })
        .collect();
    samples.sort();
    samples[samples.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

struct Fixture {
    device: Device,
    words: Vec<String>,
    tokenizer_json: Vec<u8>,
    tokenizer: Tokenizer,
    labels: Vec<String>,
}

impl Fixture {
    fn manifest(&self, chunk: ChunkConfig) -> Manifest {
        Manifest::new(
            MiniLmVariant::L6,
            minilm_l6_config(),
            ClassifierConfig::new(self.labels.clone()).with_chunking(chunk),
        )
    }

    fn classifier(&self, module: SetFitModule, chunk: ChunkConfig) -> Res<Classifier> {
        Ok(Classifier::new(
            module,
            self.tokenizer.clone(),
            self.manifest(chunk),
            self.device.clone(),
        )?)
    }
}

fn bench_chunking(fx: &Fixture) -> Res<()> {
    println!("## chunking (tokenize + segment + pack, no model)\n");
    println!("| document | bytes | chunks | tokens | padded at batch 8 | ms | MB/s |");
    println!("| -------- | ----- | ------ | ------ | ----------------- | -- | ---- |");
    for words in [2_000, 20_000, 200_000] {
        let doc = document(&fx.words, words, 7);
        let mut stats = (0, 0);
        let t = time(5, || {
            let chunker = Chunker::new(
                core::iter::once(doc.clone()),
                &fx.tokenizer,
                ChunkConfig::default(),
            );
            stats = chunker.fold((0, 0), |(n, tok), c| {
                (n + 1, tok + c.expect("chunks").token_count)
            });
        });
        // What batching consecutive chunks costs: every row is padded to the
        // longest in its batch, and the encoder computes over the padding.
        let lengths: Vec<usize> = Chunker::new(
            core::iter::once(doc.clone()),
            &fx.tokenizer,
            ChunkConfig::default(),
        )
        .map(|c| c.expect("chunks").ids.len())
        .collect();
        let real: usize = lengths.iter().sum();
        let padded: usize = lengths
            .chunks(8)
            .map(|b| b.len() * b.iter().max().copied().unwrap_or(0))
            .sum();
        println!(
            "| {words} words | {} | {} | {} | +{:.0}% | {:.1} | {:.1} |",
            doc.len(),
            stats.0,
            stats.1,
            100.0 * (padded as f64 / real as f64 - 1.0),
            ms(t),
            doc.len() as f64 / t.as_secs_f64() / 1e6
        );
    }
    println!();
    Ok(())
}

fn bench_forward(fx: &Fixture) -> Res<()> {
    println!("## forward pass (body + head, one batch, includes readback)\n");
    println!("| batch | seq | ms | ms/seq | tokens/s |");
    println!("| ----- | --- | -- | ------ | -------- |");
    let module = SetFitModule::init(&minilm_l6_config(), fx.labels.len(), &fx.device);
    for (batch, seq) in [
        (1, 16),
        (16, 16),
        (32, 16),
        (1, 128),
        (8, 128),
        (1, 256),
        (8, 256),
        (16, 256),
        (32, 256),
    ] {
        let seqs: Vec<Vec<u32>> = (0..batch)
            .map(|b| {
                (0..seq)
                    .map(|i| 5 + ((b * 31 + i * 7) % 500) as u32)
                    .collect()
            })
            .collect();
        let runs = if batch * seq > 2048 { 3 } else { 7 };
        let t = time(runs, || {
            let (ids, mask) = pad_batch(&seqs, 0, &fx.device);
            module.forward(ids, mask).into_data()
        });
        println!(
            "| {batch} | {seq} | {:.1} | {:.2} | {:.0} |",
            ms(t),
            ms(t) / batch as f64,
            (batch * seq) as f64 / t.as_secs_f64()
        );
    }
    println!();
    Ok(())
}

fn bench_infer(fx: &Fixture) -> Res<()> {
    println!("## classify (end to end: chunk, batch, forward, reduce)\n");
    println!("| document | batch size | chunks | ms | ms/chunk |");
    println!("| -------- | ---------- | ------ | -- | -------- |");
    let module = SetFitModule::init(&minilm_l6_config(), fx.labels.len(), &fx.device);
    let base = fx.classifier(module, ChunkConfig::default())?;
    for words in [200, 5_000] {
        let doc = document(&fx.words, words, 11);
        for batch in [8, 32] {
            let classifier = fx.classifier(
                SetFitModule::init(&minilm_l6_config(), fx.labels.len(), &fx.device),
                ChunkConfig::default(),
            )?;
            let classifier = classifier.with_batch_size(batch);
            let mut chunks = 0;
            let t = time(3, || {
                chunks = classifier.classify(&doc).expect("classifies").chunks_seen;
            });
            println!(
                "| {words} words | {batch} | {chunks} | {:.0} | {:.1} |",
                ms(t),
                ms(t) / chunks.max(1) as f64
            );
        }
    }

    let sentences: Vec<String> = (0..32).map(|i| sentence(&fx.words, 100 + i)).collect();
    let refs: Vec<&str> = sentences.iter().map(String::as_str).collect();
    let t = time(5, || base.embed(&refs).expect("embeds"));
    println!(
        "\n`embed` of 32 short sentences: {:.1} ms ({:.2} ms/sentence)\n",
        ms(t),
        ms(t) / 32.0
    );
    Ok(())
}

fn bench_train(fx: &Fixture) -> Res<()> {
    println!("## training (MiniLM-L6 shapes, 24 examples, 3 labels)\n");
    let body = minilm_l6_config().init(&fx.device);
    let weights = {
        use burn_store::{ModuleSnapshot, SafetensorsStore};
        let mut store = SafetensorsStore::from_bytes(None);
        body.save_into(&mut store)?;
        store.get_bytes()?
    };
    let checkpoint = Checkpoint::new(
        MiniLmVariant::L6,
        minilm_l6_config(),
        weights,
        fx.tokenizer_json.clone(),
    )
    .with_burn_naming();

    let t = Instant::now();
    let _ = checkpoint.body(&fx.device)?;
    println!("checkpoint → body: {:.0} ms", ms(t.elapsed()));

    let examples: Vec<Example> = (0..24)
        .map(|i| Example::single(sentence(&fx.words, 500 + i), i as usize % 3))
        .collect();
    let config = TrainConfig {
        num_iterations: 2,
        ..TrainConfig::default()
    };
    let t = Instant::now();
    let mut trainer = Trainer::new(
        &checkpoint,
        ClassifierConfig::new(fx.labels.clone()),
        examples,
        config,
        fx.device.clone(),
    )?;
    println!("Trainer::new: {:.0} ms", ms(t.elapsed()));

    let mut body_steps = Vec::new();
    let mut head_steps = Vec::new();
    let mut transition = Duration::ZERO;
    let mut last_stage = Stage::Body;
    loop {
        let t = Instant::now();
        let Some(p) = trainer.step()? else { break };
        let d = t.elapsed();
        match p.stage {
            Stage::Body => body_steps.push(d),
            Stage::Head if last_stage == Stage::Body => transition = d,
            Stage::Head => head_steps.push(d),
            Stage::Done => {}
        }
        last_stage = p.stage;
    }
    let median = |v: &mut Vec<Duration>| {
        v.sort();
        v.get(v.len() / 2).copied().unwrap_or_default()
    };
    let (nb, nh) = (body_steps.len(), head_steps.len());
    let body_total: Duration = body_steps.iter().sum();
    let head_total: Duration = head_steps.iter().sum();
    println!(
        "body: {nb} steps of 16 pairs, median {:.0} ms/step, total {:.1} s",
        ms(median(&mut body_steps)),
        body_total.as_secs_f64()
    );
    println!(
        "body → head (embed every example once, plus the first head step): {:.0} ms",
        ms(transition)
    );
    println!(
        "head: {nh} steps, median {:.2} ms/step, total {:.0} ms",
        ms(median(&mut head_steps)),
        ms(head_total)
    );

    let t = Instant::now();
    let bundle = trainer.finish()?;
    println!(
        "finish (pack): {:.0} ms, {:.1} MB\n",
        ms(t.elapsed()),
        bundle.len() as f64 / 1_048_576.0
    );
    Ok(())
}

fn bench_bundle(fx: &Fixture) -> Res<()> {
    println!("## bundles\n");
    println!("| mode | pack ms | unpack ms | load_module ms | Classifier::from_bundle ms |");
    println!("| ---- | ------- | --------- | -------------- | -------------------------- |");
    let module = SetFitModule::init(&minilm_l6_config(), fx.labels.len(), &fx.device);
    for mode in [Quantization::None, Quantization::F16, Quantization::Int8] {
        let manifest = fx.manifest(ChunkConfig::default()).with_quantization(mode);
        let mut bytes = Vec::new();
        let pack = time(3, || {
            bytes = Bundle::pack(&module, &manifest, &fx.tokenizer_json).expect("packs");
        });
        let unpack = time(3, || Bundle::unpack(&bytes).expect("unpacks"));
        let bundle = Bundle::unpack(&bytes)?;
        let load = time(3, || bundle.load_module(&fx.device).expect("loads"));
        let full = time(3, || {
            Classifier::from_bundle(&bytes, fx.device.clone()).expect("loads")
        });
        println!(
            "| `{}` | {:.0} | {:.0} | {:.0} | {:.0} |",
            mode.as_str(),
            ms(pack),
            ms(unpack),
            ms(load),
            ms(full)
        );
    }
    println!();
    Ok(())
}

#[allow(deprecated)] // `ndarray` is deprecated upstream in favour of `flex`
fn device() -> Device {
    #[cfg(feature = "wgpu")]
    if std::env::args().any(|a| a == "--wgpu") {
        return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
    }
    #[cfg(feature = "flex")]
    if std::env::args().any(|a| a == "--flex") {
        return Device::flex();
    }
    Device::ndarray()
}

fn main() -> Res<()> {
    let filter: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .collect();
    let wanted = |name: &str| filter.is_empty() || filter.iter().any(|f| name.contains(f.as_str()));

    let words = lexicon(4000);
    let tokenizer_json = tokenizer_json(&words);
    let fx = Fixture {
        device: device(),
        tokenizer: Tokenizer::from_bytes(&tokenizer_json)?,
        tokenizer_json,
        words,
        labels: ["billing", "outage", "feature"].map(String::from).to_vec(),
    };
    println!("# burn-setfit bench on {:?}\n", fx.device);

    let stages: [(&str, Bench); 5] = [
        ("chunking", bench_chunking),
        ("forward", bench_forward),
        ("infer", bench_infer),
        ("train", bench_train),
        ("bundle", bench_bundle),
    ];
    for (name, stage) in stages {
        if wanted(name) {
            stage(&fx)?;
        }
    }
    Ok(())
}
