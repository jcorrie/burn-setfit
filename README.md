# burn-setfit

SetFit few-shot text classification in Rust on [Burn](https://github.com/tracel-ai/burn),
built for native and browser targets, over documents of unbounded length.

Both SetFit stages build for `wasm32-unknown-unknown`, and the training loop is a
step-wise state machine rather than a blocking `fit()` — so a browser fine-tunes
from a handful of labelled examples rather than merely running a model trained
elsewhere. A browser now does exactly that: `browser-test/` trains, packs and
classifies in headless Chromium on every run. What it uses is a toy checkpoint,
so fidelity against the real `all-MiniLM-L6-v2` in a browser is still open ([#1]).

## What it does

- **Few-shot training.** Contrastive fine-tuning of a MiniLM body, then a linear head.
- **Single- and multi-label.** One set of weights; the loss and decoding differ.
- **Unbounded input.** Documents stream through a windowing chunker in constant memory.
- **Traceable labels.** Predictions carry the byte ranges of the passages that caused them.
- **One-file models.** A `.setfit` bundle holds weights, tokenizer and config together.

## Status

Working and tested end to end, but pre-release and unpublished. What is actually
verified, and what is not, matters more than a version number here:

| | |
| --- | --- |
| **Verified** | Embedding path reproduces `all-MiniLM-L6-v2` exactly (below). 148 offline tests. Native training and inference, on `NdArray`. Training and inference in a browser, on a toy checkpoint ([below](#webassembly)). |
| **Compiles, never run** | The `wgpu` backend, on any target ([#5]). On `wasm32` it also needs an explicit opt-in — see [Features](#features). |
| **Runs, but not against the real model** | The browser path, which has only been driven with a 146 KB stand-in checkpoint ([#1]). |
| **Known limitation** | Long-document classification separates signal from filler only in one configuration ([below](#a-measured-limitation), [#4]). |

Open work is tracked in [issues](https://github.com/jcorrie/burn-setfit/issues);
[#1] and [#2] are the two that matter.

[#1]: https://github.com/jcorrie/burn-setfit/issues/1
[#2]: https://github.com/jcorrie/burn-setfit/issues/2
[#3]: https://github.com/jcorrie/burn-setfit/issues/3
[#4]: https://github.com/jcorrie/burn-setfit/issues/4
[#5]: https://github.com/jcorrie/burn-setfit/issues/5
[#6]: https://github.com/jcorrie/burn-setfit/issues/6

## Quick start

```rust
use burn::backend::{Autodiff, NdArray};
use burn_setfit::{Checkpoint, Classifier, ClassifierConfig, Example, MiniLmVariant,
                  TrainConfig, Trainer};

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

Then classify from those bytes alone — the same code path natively and in the
browser:

```rust
let classifier = Classifier::<NdArray<f32>>::from_bundle(&bundle, Default::default())?;
let prediction = classifier.classify(very_long_document)?;
```

`examples/train_and_classify.rs` is the same thing, complete and compiled.

Everything a caller ordinarily names is re-exported at the crate root, so that
is one `use` line rather than six. The modules stay public: accumulators, the
chunker, the head and the vendored body are all still reachable at their own
paths.

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

## Errors

One type, [`SetFitError`], across the whole crate, and `?` composes with
`std::io` so reading a bundle off disk and parsing it need one error type rather
than a `Box<dyn Error>`:

```rust
fn load(path: &str) -> burn_setfit::Result<Classifier<NdArray<f32>>> {
    let bytes = std::fs::read(path)?;                    // io::Error
    Classifier::from_bundle(&bytes, Default::default())  // SetFitError
}
```

The underlying `io::Error` is kept rather than flattened into a string, so
`ErrorKind` and `source()` both still work. The other variants are built from
foreign errors — `burn_store`, `tokenizers`, `hf_hub` — and fold those messages
into their own, because naming which file or which example failed is worth more
than the chain. The enum is `#[non_exhaustive]`, so gaining a variant is not a
breaking change.

## Examples

| Example | What it shows | Needs |
| ------- | ------------- | ----- |
| `train_and_classify` | The whole path: download, fine-tune, pack, classify short and long input | network |
| `long_document` | Four configurations over a planted signal and a filler-only control — where the measurements in [A measured limitation](#a-measured-limitation) come from | network |
| `local_checkpoint` | The same round trip from three files on disk rather than a download, which is the entry point the browser uses | a checkpoint directory |
| `reference` | An independent BERT forward pass from the raw safetensors, for bisecting a disagreement between Burn and the checkpoint | network |

```bash
cargo run --release --example train_and_classify
cargo run --release --example long_document
```

`local_checkpoint` is the one that runs without network access. Point it at a
HuggingFace snapshot you already have, or generate the toy checkpoint the
browser harness uses:

```bash
python3 browser-test/make_toy_checkpoint.py /tmp/toy
cargo run --release --example local_checkpoint -- /tmp/toy
```

That toy body is 32 wide over a vocabulary of single letters, so it exercises
the pipeline and tells you nothing about the model.

## API documentation

```bash
cargo doc --open --features ndarray,train,native
```

37 examples in the docs, all compiled as tests on every run — the ones that
need no weights actually execute:

```bash
cargo test --doc --features ndarray,train,native
```

## Tests

148 tests, none needing a network. What each file is for:

| File | Covers |
| ---- | ------ |
| `tests/config.rs` | Every way a configuration can be invalid, and that the message names the culprit |
| `tests/bundle.rs` | Container integrity — including that no truncated prefix of a bundle ever parses |
| `tests/chunk.rs` | Windowing, overlap, streaming equivalence, UTF-8 boundary safety |
| `tests/reduce.rs` | Each online fold against the plain definition it implements |
| `tests/infer.rs` | Decoding, evidence, and invariance to batch size |
| `tests/train.rs` | Data validation, the training state machine, reproducibility |
| `tests/error.rs` | That one error type crosses the crate, and that a message names the culprit |
| `tests/async_api.rs` | That the `_async` methods and their blocking wrappers agree exactly |
| `tests/tokenize.rs` | Encoding, batching, and that tokenizer-level padding stays stripped |
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
background class** ([#4] tracks making this less manual) on text representative of
your filler. That is the only configuration above that both separates signal from
noise and correctly rejects the control — though note it is *under-confident*: the
signal document scores 0.389 and so predicts `other` at the default 0.5 threshold.
It separates well and calibrates badly.

## WebAssembly

```bash
cargo build --release --target wasm32-unknown-unknown -p setfit-wasm
```

**A browser runs this.** `browser-test/` drives the built artifact in headless
Chromium: it fetches a checkpoint, fine-tunes 12 examples over 26 steps, packs a
bundle, loads it back and classifies a document that windows into 5 chunks. The
run asserts on what it measures, so it fails rather than reports:

| | |
| --- | --- |
| wasm instantiate | 188 ms |
| `new Trainer(...)` | 55 ms — tokenizes every example and seeds the head |
| 26 training steps | 515 ms |
| median main-thread frame gap during training | 17 ms |
| **worst main-thread frame gap** | **133 ms** |
| classify a 2.7 KB document | 79 ms |

The worst frame gap is the number that matters: the tab keeps painting
throughout, which is the entire reason `Trainer` is a step-wise state machine
rather than a `fit()` loop. The 133 ms outlier is the body-to-head transition,
which embeds the whole training set inside one `step()`.

The `tokenizers` rayon worry that shaped `src/tokenize.rs` looks unfounded on
this evidence — the single-sequence `encode` path runs fine in a browser. That
is not a licence to reach for `encode_batch`, which remains untested there.

Still open ([#1]): every browser number above comes from a 32-wide toy
checkpoint. Fidelity in a browser needs the real 86.5 MB model.

Training yields between steps, so a browser stays responsive:

```js
const trainer = new Trainer(configJson, weights, tokenizerJson, request);
let p;
do {
  p = JSON.parse(trainer.step());
  render(p.fraction);
  await new Promise(requestAnimationFrame);
} while (!p.done);
const bundle = trainer.finish();
```

Payload, and the part that actually matters:

| Component            | Size    |
| -------------------- | ------- |
| `setfit_wasm.wasm`   | 8.0 MB (`cargo build --release`, before `wasm-opt -Oz`, [#6]) |
| `.setfit` bundle     | 86.5 MB (MiniLM-L6, f32) |

**The model dominates, not the code** — by an order of magnitude. Roughly half of
MiniLM-L6's 22.7M parameters are the 30522×384 embedding table, which quantizes
well; f16 should roughly halve the bundle and int8 roughly quarter it. Not yet
implemented ([#2]). Cache the bundle in the Cache API or IndexedDB — it is one
`fetch`, by design.

## Design notes

**The MiniLM body is vendored**, not depended upon ([#3]). Upstream `minilm-burn` pins
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

| Feature   | Purpose                            | wasm | Exercised by tests |
| --------- | ---------------------------------- | ---- | ------------------ |
| `ndarray` | CPU backend                        | yes  | yes                |
| `wgpu`    | GPU backend                        | opt-in below | **no** — compiles only, see [#5] |
| `wgpu-wasm-unverified` | `wgpu` on `wasm32`    | yes  | **no** — never executed |
| `train`   | Both training stages               | yes  | yes                |
| `native`  | HuggingFace download, filesystem   | no   | yes (`--ignored`)  |

`native` is filesystem and network, and is quarantined to one module for that
reason — see the design notes.

### A GPU backend in a browser

WebGPU has no synchronous readback. A browser cannot block a thread on a GPU
buffer map, so Burn's `Tensor::into_data()` polls the map future once, finds it
pending and panics. Lazy `WgpuDevice` acquisition goes through the same
`block_on`, so the trap arrives even earlier — while a tensor is being built,
before anything is read. The combination used to compile and then die in the
browser as a bare `unreachable`, with nothing in the message naming the cause.

Every method that reads a tensor now has an `_async` twin:

```rust
let prediction = classifier.classify_async(text).await?;
let vectors    = classifier.embed_async(&texts).await?;
while let Some(progress) = trainer.step_async().await? { /* ... */ }
```

Those are the implementations. The blocking methods are wrappers that drive them
to completion, so there is one code path rather than two kept in step by hand —
`tests/async_api.rs` holds them to identical results. Natively the wrapper
always works. On `wasm32` it succeeds only for a backend whose reads finish
immediately, which `ndarray` does and a GPU backend does not; where it cannot
wait it returns `SetFitError::Readback` naming the `_async` method, instead of
panicking inside Burn.

Two things stay the caller's job on that target:

- **Bring the device up first**, with `burn::backend::wgpu::init_setup_async`.
  The lazy path cannot work, and without this the trap happens while the first
  tensor is built, before any readback.
- **Ask for it explicitly**, with `wgpu-wasm-unverified`. The combination
  compiles and the API it needs is here, but it has never been run on a GPU in a
  browser. The feature is named so that enabling it cannot be mistaken for
  evidence that it has.

## Repository map

| Path | What lives there |
| ---- | ---------------- |
| `src/config.rs` | `ClassifierConfig` — the one place behaviour is decided and validated |
| `src/checkpoint.rs` | `Checkpoint` — config + weights + tokenizer, however they arrived |
| `src/chunk.rs` | Streaming windower. Constant memory over unbounded input |
| `src/reduce.rs` | Per-chunk scores to a document verdict. Read the module docs before adding one |
| `src/train.rs` | Both SetFit stages, as a step-wise state machine |
| `src/bundle.rs` | The `.setfit` container |
| `src/infer.rs` | `Classifier`, decoding, evidence |
| `src/readback.rs` | Getting tensors off the device. Read it before adding an `into_data` |
| `src/minilm/` | Vendored MiniLM body — see [#3] before touching |
| `crates/setfit-wasm/` | Browser bindings |

## Working on this

```bash
cargo test --features ndarray,train,native      # 133 tests, ~2s, no network
```

That is the loop to stay in. The network tests and both examples download ~90 MB
and train, so they are slow and awkward on a constrained connection:

```bash
cargo test --release --features ndarray,train,native -- --ignored   # fidelity, needs network
cargo run --release --example train_and_classify                    # ~21s after download
cargo run --release --example long_document                         # trains four models
```

Two things in the test harness are deliberate and easy to break:

- `tests/common/mod.rs::toy_bundle` memoises **holding the lock across training**.
  Releasing it to build and re-taking it to insert lets two threads each train a
  differently-initialised model and compare them — the failures move between runs.
- `toy_checkpoint` builds weights by saving a randomly-initialised body, so two
  calls give two different models. Tests that compare must share one bundle.

## Licence

MIT OR Apache-2.0. Vendored MiniLM code from
[tracel-ai/models](https://github.com/tracel-ai/models), same terms.
