# burn-setfit

SetFit few-shot text classification in Rust on [Burn](https://github.com/tracel-ai/burn),
running natively and in the browser, over documents of unbounded length.

Both SetFit stages work on `wasm32-unknown-unknown` — a browser can fine-tune from
a handful of labelled examples, not merely run a model trained elsewhere.

## What it does

- **Few-shot training.** Contrastive fine-tuning of a MiniLM body, then a linear head.
- **Single- and multi-label.** One set of weights; the loss and decoding differ.
- **Unbounded input.** Documents stream through a windowing chunker in constant memory.
- **Traceable labels.** Predictions carry the byte ranges of the passages that caused them.
- **One-file models.** A `.setfit` bundle holds weights, tokenizer and config together.

## Status

Working and tested end to end. The embedding path is verified against upstream
(below); the long-document path has a measured limitation worth reading before
you rely on it.

## Quick start

```rust
use burn::backend::{Autodiff, NdArray};
use burn_setfit::{Checkpoint, ClassifierConfig, minilm::MiniLmVariant};
use burn_setfit::train::{Example, TrainConfig, Trainer};

let checkpoint = Checkpoint::download(MiniLmVariant::L6, None)?;

let mut trainer = Trainer::<Autodiff<NdArray<f32>>>::new(
    &checkpoint,
    ClassifierConfig::new(["billing", "outage"]),
    vec![
        Example::single("I was charged twice this month.", 0),
        Example::single("The dashboard is returning 503s.", 1),
        // ...eight or so per class
    ],
    TrainConfig::default(),
    Default::default(),
)?;
trainer.fit_with(|p| eprintln!("{:?} {}/{}", p.stage, p.step, p.total_steps))?;

// The trainer knows the checkpoint, labels and task, so this needs nothing else.
let bundle: Vec<u8> = trainer.finish()?;
```

Then classify, natively or in a browser, from those bytes alone:

```rust
let classifier = Classifier::<NdArray<f32>>::from_bundle(&bundle, Default::default())?;
let prediction = classifier.classify(very_long_document)?;
```

## Configuration

[`ClassifierConfig`] is the one place behaviour is decided, and the one place it
is checked. Setters chain and never fail; every entry point that consumes a
config validates it, so a mistake surfaces where it is used rather than as a
model that merely behaves oddly.

```rust
ClassifierConfig::new(["billing", "outage", "feature"])
    .multi_label_at(0.4)                              // threshold lives in the task
    .with_reducer(Reducer::MaxLogits)                 // else the task's default
    .with_chunking(ChunkConfig::new(256).with_overlap(32))
    .with_hierarchy(8)
```

Two things are deliberate. The multi-label threshold lives *inside* the task
mode, because an argmax has nothing to threshold — a single-label config with a
carefully tuned threshold that silently does nothing is not representable.
And changing the task moves the reducer to that task's default, because a mean
over chunks answers "is the document about this", which is not the multi-label
question; pin one afterwards with `with_reducer` if you disagree.

Reduction, hierarchy and threshold are decode-time choices, so they can be swept
without retraining or repacking:

```rust
Classifier::from_bundle(&bundle, device)?.with_reducer(Reducer::TopKMeanLogits { k: 3 })
```

Run the examples:

```bash
cargo run --release --example train_and_classify
```

```bash
cargo run --release --example long_document
```

## Tests

```bash
cargo test --features ndarray,train,native
```

133 tests. The ones worth knowing about:

| File | Covers |
| ---- | ------ |
| `tests/config.rs` | Every way a configuration can be invalid, and that the message names the culprit |
| `tests/bundle.rs` | Container integrity — including that no truncated prefix of a bundle ever parses |
| `tests/chunk.rs` | Windowing, overlap, streaming equivalence, UTF-8 boundary safety |
| `tests/reduce.rs` | Each online fold against the plain definition it implements |
| `tests/infer.rs` | Decoding, evidence, and invariance to batch size |
| `tests/train.rs` | Data validation, the training state machine, reproducibility |
| `tests/pretrained.rs` | Fidelity against the real checkpoint (network; `--ignored`) |

## Fidelity

`tests/pretrained.rs` checks the embedding path against the similarity matrix
published for `all-MiniLM-L6-v2`, and reproduces it to four decimal places:

|            | `[0]`  | `[1]`  | `[2]`  |
| ---------- | ------ | ------ | ------ |
| **`[0]`**  | 1.0000 | 0.6660 | 0.1046 |
| **`[1]`**  | 0.6660 | 1.0000 | 0.1411 |
| **`[2]`**  | 0.1046 | 0.1411 | 1.0000 |

The Burn forward pass was additionally cross-checked against an independent
BERT implementation written from the raw safetensors
(`examples/reference.rs`): agreement `cos 1.000000`, largest element difference
`5e-6`.

```bash
cargo test --release --features ndarray,train,native -- --ignored
```

## Long documents

MiniLM sees 256 tokens. Longer input is windowed at sentence boundaries with
overlap, each window is scored, and the per-window scores are folded into one
verdict by a [`Reducer`].

One property is worth knowing before choosing one: **with a linear head,
averaging chunk embeddings and averaging chunk logits are the same operation.**
A linear map commutes with a weighted mean, so "pool then classify" and "classify
then pool" are not alternatives — they are both `MeanLogits`. Only the nonlinear
reducers add anything.

| Reducer            | Behaviour                                              |
| ------------------ | ------------------------------------------------------ |
| `MeanLogits`       | Token-weighted mean. Equivalent to mean-pooling.        |
| `MaxLogits`        | Strongest chunk wins. Default for multi-label.          |
| `TopKMeanLogits`   | Mean of the best `k`. Robust to one spurious chunk.     |
| `LogSumExp`        | Smoothly interpolates mean ↔ max.                       |
| `NoisyOr`          | `1 - Π(1 - pᵢ)`. **Saturates** — see below.             |

Reduction can also be **hierarchical**: chunks fold into blocks, blocks into the
document, recursively. `MeanLogits` gives an identical answer either way (proven
in `tests/reduce.rs`); the hierarchy exists for the nonlinear reducers, where
"some section is about this" is a better claim than "some chunk mentioned it".

### A measured limitation

`examples/long_document.rs` plants one billing complaint in ~13 KB of unrelated
filler and classifies both that document and a filler-only control. If a
classifier reports the same score for both, it is reading the filler:

| Configuration                         | With signal | Control | Separation |
| ------------------------------------- | ----------- | ------- | ---------- |
| Single-label, `MeanLogits`            | 0.698       | 0.672   | **+0.026** |
| Multi-label, `NoisyOr`                | 1.000       | 1.000   | **+0.000** |
| Multi-label, `NoisyOr`, blocks of 8   | 0.997       | 0.995   | **+0.002** |
| Multi-label, `MaxLogits`              | 0.764       | 0.559   | **+0.206** |
| Multi-label + background, `MaxLogits` | 0.389       | 0.169   | **+0.220** |

Two conclusions, both load-bearing:

**A single-label softmax head cannot abstain.** Every chunk must be assigned to
some class, so filler votes as confidently as signal. With 12 chunks of which one
is relevant, the verdict is whatever the *filler* most resembles. This is not a
tuning problem; it is what softmax over a fixed label set means.

**Noisy-OR saturates.** Combining evidence multiplicatively only behaves when
irrelevant chunks score near zero. At a realistic `p ≈ 0.6` for background text,
eight chunks reach `1 - 0.4⁸ ≈ 0.999` and every label fires. Reducing
hierarchically does not rescue it — saturation happens inside the first block.
This measurement is why `Reducer::default_for(MultiLabel)` is `MaxLogits`.

**If you classify long documents, use multi-label with `MaxLogits`, and train a
background class** on text representative of your filler. That is the only
configuration above that both separates signal from noise and correctly rejects
the control.

## WebAssembly

```bash
cargo build --release --target wasm32-unknown-unknown -p setfit-wasm
```

Training yields between steps, so a browser stays responsive:

```js
const trainer = new Trainer(configJson, weights, tokenizerJson, request);
let p;
do {
  p = JSON.parse(trainer.step());
  render(p.fraction);
  await new Promise(requestAnimationFrame);
} while (!p.done);
const bundle = trainer.finish(0.5);
```

Payload, and the part that actually matters:

| Component            | Size    |
| -------------------- | ------- |
| `setfit_wasm.wasm`   | 8.5 MB (before `wasm-opt -Oz`) |
| `.setfit` bundle     | 86.5 MB (MiniLM-L6, f32) |

**The model dominates, not the code.** Roughly half of MiniLM-L6's 22.7M
parameters are the 30522×384 embedding table, which quantizes well; f16 roughly
halves the bundle and int8 roughly quarters it. Cache the bundle in the Cache API
or IndexedDB — it is one `fetch`, by design.

## Design notes

**The MiniLM body is vendored**, not depended upon. Upstream `minilm-burn` pins
`tokenizers` with the `onig` feature; Cargo features are additive, so a downstream
crate cannot switch it off, and oniguruma does not build for wasm. The vendored
copy (`src/minilm/`, MIT OR Apache-2.0) additionally loads from bytes rather than
paths, which is what lets the browser and native paths share everything below the
byte level.

**`tokenizers` padding is stripped on load.** Published `tokenizer.json` files
often pad to a fixed length — `all-MiniLM-L6-v2` pads to 128. This crate packs raw
token ids into windows and pads once per batch, deriving the attention mask from
each sequence's true length, so upstream padding is indistinguishable from content
by the time that mask is built: the encoder attends to it and mean pooling averages
it in. Left in place it made every sentence ~94% identical padding and pushed
unrelated-sentence similarity from 0.10 to 0.90. Regression test in
`tests/pretrained.rs`.

**A seeded run reproduces the same weights, not the same bytes.** `TrainConfig::seed`
covers pair sampling, shuffling *and* head initialisation — the last of which
previously drew from Burn's global RNG, so seeded runs were not actually
reproducible. The `.setfit` container is still not byte-canonical: Burn writes the
safetensors `__metadata__` map in `HashMap` order, which Rust seeds per map, so
identical models serialise to headers whose keys are ordered differently. Compare
models by their tensors; content-addressing a bundle will not work.

**The head is `use_differentiable_head`**, SetFit's own alternative to the sklearn
`LogisticRegression` default: a `Linear` layer with AdamW and weight decay. Same
model class, no LBFGS to reimplement, and it trains on every Burn backend.

## Features

| Feature     | Purpose                                    | wasm |
| ----------- | ------------------------------------------ | ---- |
| `ndarray`   | CPU backend                                | yes  |
| `wgpu`      | GPU backend (WebGPU in browsers)           | yes  |
| `train`     | Both training stages                       | yes  |
| `native`    | HuggingFace download, filesystem           | no   |

## Licence

MIT OR Apache-2.0. Vendored MiniLM code from
[tracel-ai/models](https://github.com/tracel-ai/models), same terms.
