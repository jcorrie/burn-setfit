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
| **Verified** | Embedding path reproduces `all-MiniLM-L6-v2` exactly (below). 185 offline tests. Native training and inference, on `NdArray`. Training and inference in a browser, on a toy checkpoint ([below](#webassembly)). Native `wgpu` trains and classifies, and agrees with `NdArray` ([#5]). |
| **Run on software Vulkan only** | `wgpu` — the suite passes on Mesa's lavapipe (`llvmpipe`, `device_type: Cpu`), which exercises the whole backend but is not hardware ([#5]). |
| **Run in a headless browser on a software GPU only** | `wgpu` on `wasm32`, i.e. WebGPU in a browser. Needs an explicit opt-in — see [Features](#features). The excel-embed add-in trained, packed and classified through it in headless Chromium on SwiftShader (a software Vulkan), with losses matching `ndarray`. Never on real GPU hardware, and never in Safari. |
| **Reported, not reproduced here** | The real model in a real browser: the maintainer runs this client-side in an Excel add-in on `NdArray` ([#1]). The repository's own harness still only drives a 146 KB stand-in. |
| **Known limitation** | Long-document classification separates signal from filler only with multi-label `MaxLogits`, and only once the threshold is calibrated; a background class does not help ([below](#a-measured-limitation), #12). |

Open work is tracked in [issues](https://github.com/jcorrie/burn-setfit/issues);
#12 holds what still needs the real checkpoint.

[#1]: https://github.com/jcorrie/burn-setfit/issues/1
[#2]: https://github.com/jcorrie/burn-setfit/issues/2
[#3]: https://github.com/jcorrie/burn-setfit/issues/3
[#4]: https://github.com/jcorrie/burn-setfit/issues/4
[#5]: https://github.com/jcorrie/burn-setfit/issues/5
[#6]: https://github.com/jcorrie/burn-setfit/issues/6

## Quick start

```rust
use burn::tensor::Device;
use burn_setfit::{Checkpoint, Classifier, ClassifierConfig, Example, MiniLmVariant,
                  TrainConfig, Trainer};

let checkpoint = Checkpoint::download(MiniLmVariant::L6, None)?;

let mut trainer = Trainer::new(
    &checkpoint,
    ClassifierConfig::new(["billing", "outage"]),
    vec![
        Example::single("I was charged twice this month.", 0),
        Example::single("The dashboard is returning 503s.", 1),
        // ...eight or so per class
    ],
    TrainConfig::default(),
    Device::default(), // the backend is a runtime choice; autodiff is added for you
)?;
trainer.fit_with(|p| eprintln!("{:?} {}/{}", p.stage, p.step, p.total_steps))?;

// The trainer knows the checkpoint, labels and task, so this needs nothing else.
let bundle: Vec<u8> = trainer.finish()?;
```

Then classify from those bytes alone — the same code path natively and in the
browser:

```rust
let classifier = Classifier::from_bundle(&bundle, Device::default())?;
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
fn load(path: &str) -> burn_setfit::Result<Classifier> {
    let bytes = std::fs::read(path)?;                   // io::Error
    Classifier::from_bundle(&bytes, Device::default())  // SetFitError
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

152 tests, none needing a network. What each file is for:

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
| `tests/quantize.rs` | That every precision round-trips, shrinks, and stays close to fp32 |
| `tests/background.rs` | The background-class decision rule, and per-chunk output |
| `tests/wgpu.rs` | The `wgpu` backend against `ndarray` (needs the feature) |
| `tests/wgpu.rs` | That the `wgpu` backend runs at all, and agrees with `ndarray` (needs `--features wgpu`) |
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
classifier reports the same score for both, it is reading the filler. Measured
on `all-MiniLM-L6-v2` by the Measure workflow (`.github/workflows/measure.yml`),
one training run per row group:

| Configuration                         | With signal | Control | Separation | Default verdict (signal / control) | Thresholds that get both right |
| ------------------------------------- | ----------- | ------- | ---------- | ---------------------------------- | ------------------------------ |
| Single-label, `MeanLogits`            | 0.737       | 0.714   | **+0.023** | billing / billing                  | — (a softmax cannot abstain)   |
| Multi-label, `NoisyOr`                | 1.000       | 1.000   | **+0.000** | everything / everything            | none                           |
| Multi-label, `NoisyOr`, blocks of 8   | 0.999       | 0.998   | **+0.001** | billing, feature / same            | none usable                    |
| Multi-label, `MaxLogits`              | 0.808       | 0.617   | **+0.191** | billing / billing                  | **0.62 – 0.81**                |
| Multi-label + background, `MaxLogits` | 0.392       | 0.164   | **+0.228** | other / other                      | **0.17 – 0.39**                |
| … decoded against the background      | 0.392       | 0.164   | +0.228     | nothing / nothing                  | none                           |

Three conclusions, all load-bearing:

**A single-label softmax head cannot abstain.** Every chunk must be assigned to
some class, so filler votes as confidently as signal. With 12 chunks of which one
is relevant, the verdict is whatever the *filler* most resembles. This is not a
tuning problem; it is what softmax over a fixed label set means.

**Noisy-OR saturates.** Combining evidence multiplicatively only behaves when
irrelevant chunks score near zero. At a realistic `p ≈ 0.6` for background text,
eight chunks reach `1 - 0.4⁸ ≈ 0.999` and every label fires. Reducing
hierarchically does not rescue it — saturation happens inside the first block.
This measurement is why `Reducer::default_for(MultiLabel)` is `MaxLogits`.

**`MaxLogits` separates, and the threshold has to be calibrated.** It is the
only reducer whose scores tell the documents apart, but at the default 0.5 both
are labelled billing. Any threshold between the control's score and the
signal's gets both right, so pick it on held-out documents of your own:
`Classifier::clone_with_threshold` makes a sweep cheap. Training a background
class lowers every score (the window moves to 0.17 – 0.39) without widening it.

### Why the background class does not help here

Decoding against a background class asks "is billing above *other*?" rather than
"is billing above a fixed threshold?":

```rust
let config = ClassifierConfig::new(["billing", "outage", "feature", "other"])
    .multi_label()
    .with_background_class("other")?;
```

On long documents it predicts nothing for either one, and the example's
diagnostics show why. Under `MaxLogits` the background's document score is its
best chunk's, and a long document is mostly filler, so *other* scores 0.79 in
both. Nor does comparing chunk by chunk rescue it at the default window: in the
signal document's most billing-like chunk, billing scores 0.39 against
*other*'s 0.61. The billing sentence is one of roughly ten in its 256-token
window, so that window really is mostly *other*.

Narrower windows remove the dilution. The same trained model, repacked to read
in smaller windows (windowing happens only at inference):

| Windows | Billing chunk: billing vs *other* | Control's most billing-like chunk | Chunks that decode to billing, signal / control |
| ------- | --------------------------------- | --------------------------------- | ----------------------------------------------- |
| 256     | 0.40 vs 0.59                      | 0.17 vs 0.76                      | 0 / 12 — 0 / 12                                 |
| 128     | 0.39 vs 0.57                      | 0.11 vs 0.80                      | 0 / 27 — 0 / 27                                 |
| **64**  | **0.66 vs 0.39**                  | 0.11 vs 0.88                      | **2 / 61 — 0 / 60**                             |

At 64 tokens, deciding *per chunk* against the background gets both documents
right at the default threshold, and without any calibration. The
document-level decode this crate implements still predicts nothing there
(*other*'s best chunk scores 0.91), so the working rule — "a label is predicted
if some chunk prefers it to the background" — is not yet a decode mode; until
it is, `classify_chunks` plus `decode_against_background` per chunk does it by
hand. Plain multi-label `MaxLogits` also separates better at 64 tokens (+0.33
against +0.19 at 256), with a wider threshold window (0.57 – 0.90).

The decision rule itself is exact and pinned by `tests/background.rs`. It gives a
single-label head somewhere to abstain *to*, and it suits inputs whose chunks
are mostly on-topic. One sentence in a long document needs small windows and a
per-chunk decision, or a calibrated threshold.

### Per-chunk output

The reducer is the lossy step, and sometimes the right move is not to reduce:

```rust
for chunk in classifier.classify_chunks(document)? {
    println!("{:?} -> {:?}", chunk.byte_range, chunk.predicted);
}
```

One row per passage — byte range, token count, per-label scores and its own
verdict — leaving the aggregating to the caller. Both paths share one forward
pass, so a chunk scores the same either way. Note this holds every row in
memory, unlike `classify_stream`, which folds as it goes; a genuinely unbounded
input wants that one.

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

**Those numbers are the toy checkpoint's.** The Measure workflow runs the same
harness on the real `all-MiniLM-L6-v2` (single-threaded wasm, `ndarray`), in
Chromium and in WebKit, the engine behind Safari and Excel for Mac:

| | Toy (Chromium) | Real MiniLM-L6, Chromium | Real MiniLM-L6, WebKit |
| --- | --- | --- | --- |
| 26 training steps | 515 ms | 9.2 s | 13.5 s |
| median main-thread frame gap during training | 17 ms | 17 ms | 19 ms |
| **worst main-thread frame gap** | 133 ms | **3.0 s** | **2.3 s** |
| classify a ~2 KB document (5 chunks) | 79 ms | 3.3 s | 5.7 s |
| held-in examples correct | 12 / 12 | 12 / 12 | 12 / 12 |
| bundle | 146 KB | 90.7 MB | 90.7 MB |

It trains and classifies correctly at full size, but stepping once per frame
does not keep a page responsive there: one contrastive step through the full
body stalls the main thread for seconds. **Train in a Web Worker**, as
excel-embed does, and drive `step()` from there; the step-wise API still matters,
since it is what lets a worker report progress and be cancelled between steps.

The `tokenizers` rayon worry that shaped `src/tokenize.rs` looks unfounded on
this evidence — the single-sequence `encode` path runs fine in a browser. That
is not a licence to reach for `encode_batch`, which remains untested there.

The numbers above come from a 32-wide toy checkpoint, so they measure the
machinery rather than the model. The remaining question in [#1] — whether the
real thing runs in a real browser — is answered separately and not by this
harness: the maintainer runs this crate client-side in a production Excel
add-in on `NdArray`. That is a report rather than something CI reproduces, and
it is recorded as one.

### Payload

The two things a browser downloads, and what can be done about them:

```bash
cargo build --release --target wasm32-unknown-unknown -p setfit-wasm
wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
  target/wasm32-unknown-unknown/release/setfit_wasm.wasm -o setfit_wasm.min.wasm
```

| | Before | After |
| --- | --- | --- |
| wasm (`wasm-opt -Oz`) | 8.45 MB | **5.78 MB** (−32%) |
| bundle (`Quantization::F16`) | 86.5 MB | **43.6 MB** (−50%) |
| bundle (`Quantization::Int8`) | 86.5 MB | **22.3 MB** (−74%) |

`wasm-opt` takes about five minutes on this artifact, which is why it is a
release step rather than part of the build. Quantization is a decode-time
non-event: weights are expanded back to `f32` before Burn sees them, so the
model computes identically on every backend — see [Quantization](#quantization)
for what it costs.

Training yields between steps. With the real model a step takes seconds, so run this loop in a Web Worker (see above); the main thread then only renders progress:

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
well: f16 halves the bundle and int8 quarters it, with no wrong answers on the
held-out probes ([Quantization](#quantization)). Cache the bundle in the Cache
API or IndexedDB — it is one `fetch`, by design.

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

**A seeded run reproduces the same bundle, byte for byte.** `TrainConfig::seed`
covers pair sampling, shuffling *and* head initialisation — the last of which
previously drew from Burn's global RNG, so seeded runs were not actually
reproducible. The bytes match because `Bundle::pack` is canonical: burn-store
writes the safetensors `__metadata__` map in `HashMap` order, which differs from
one map to the next, so `pack` rewrites the header with every key sorted. A
bundle can therefore be content-addressed — hashed for a cache key, an ETag, or a
reproducibility check.

**The head is `use_differentiable_head`**, SetFit's own alternative to the sklearn
`LogisticRegression` default: a `Linear` layer with AdamW and weight decay. Same
model class, no LBFGS to reimplement, and it trains on every Burn backend.

## Quantization

The weights are the payload: 86.1 MB against 8.5 MB of wasm. Roughly half of
MiniLM-L6's 22.7M parameters are one `30522 x 384` embedding table, which is
exactly the kind of tensor that survives losing precision.

```rust
let bundle = trainer.finish_quantized(Quantization::F16)?;
```

| Mode | Bundle | vs fp32 | Max score drift | Max cosine drift | Held-out probes |
| ---- | ------ | ------- | --------------- | ---------------- | --------------- |
| `none` | 86.5 MB | 100% | 0 | 4.8e-7 | 6/6 |
| `f16`  | 43.6 MB | **50%** | 2.3e-5 | 5.4e-7 | 6/6 |
| `int8` | 22.3 MB | **26%** | 2.4e-3 | 7.3e-4 | 6/6 |

Measured on the real `all-MiniLM-L6-v2` with `cargo run --release --example
quantize`: one model trained once (103 steps) and packed three ways, so the rows
differ only in storage. Drift is the largest change in any probe's class score,
or in any embedding's cosine to its fp32 counterpart. GitHub's runners produce
it on demand (`.github/workflows/measure.yml`); `--sizes` reproduces the size
column offline from random weights, since size depends only on tensor shapes.

**This is storage, not compute.** Burn's `Tensor::quantize` produces a `QFloat`
that arithmetic runs *on*, which needs quantized-op support from every backend
and changes what the model computes. What is wanted here is a smaller download,
so weights are compressed when a bundle is packed and expanded back to `f32`
before Burn sees them. Inference is the same code on every backend and the only
cost is the precision lost in the round trip.

Not everything is compressed, and that is the point: rank-2-and-up float tensors
above a size floor, which is the embedding table and the `Linear` weights.
LayerNorm gains and biases are rank 1, tiny, and the parameters least tolerant
of precision loss, so they stay `f32`. `int8` is scaled per row rather than per
tensor — the embedding table's rows are individual tokens, and one shared scale
would let the largest vector in the vocabulary set the resolution for all 30522.

A quantized bundle declares format version 2, so an older build refuses it
outright instead of reading `f16` bytes as though they were `f32`. Unquantized
bundles are still version 1 and unchanged.

**What is not measured here.** Six probes say no answer flipped; they do not
bound accuracy on your data. int8's score drift (2.4e-3) is a hundred times
f16's, so a model whose decisions sit close to its threshold should prefer
f16. `tests/quantize.rs` bounds the drift on the toy model (f16 ~1e-5, int8
~1e-4 on scores), which catches corruption, not quality.

## Features

| Feature   | Purpose                            | wasm | Exercised by tests |
| --------- | ---------------------------------- | ---- | ------------------ |
| `ndarray` | CPU backend                        | yes  | yes                |
| `wgpu`    | GPU backend                        | opt-in below | yes — `--test wgpu`, on software Vulkan |
| `wgpu-wasm-unverified` | `wgpu` on `wasm32`    | yes  | **barely** — once, headless, on a software GPU (via excel-embed); never on real hardware |
| `train`   | Both training stages               | yes  | yes                |
| `native`  | HuggingFace download, filesystem   | no   | yes (`--ignored`)  |

`native` is filesystem and network, and is quarantined to one module for that
reason — see the design notes.

### A GPU backend in a browser

WebGPU has no synchronous readback. A browser cannot block a thread on a GPU
buffer map, so Burn's `Tensor::into_data()` polls the map future once, finds it
pending and panics. Lazy wgpu device acquisition goes through the same
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

- **Bring the device up first**, with `Device::wgpu_options().init_async()`.
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
| `src/quantize.rs` | Narrowing weights at rest. Storage only — never what the model computes |
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
