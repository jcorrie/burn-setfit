# Browser smoke test

Answers the question [#1](https://github.com/jcorrie/burn-setfit/issues/1) asks:
does any of this actually run in a browser? It trains a model from a handful of
examples in the page, packs a bundle, loads it back, and classifies a document
long enough to chunk — then fails the run if the main thread was ever blocked
long enough to make the tab feel stuck.

## Running it

```bash
cargo build --release --target wasm32-unknown-unknown -p setfit-wasm
wasm-bindgen --target web --out-dir browser-test/public/pkg \
  target/wasm32-unknown-unknown/release/setfit_wasm.wasm
python3 browser-test/make_toy_checkpoint.py browser-test/public/checkpoint
cd browser-test && npm install && node run.mjs
```

`npm install` needs `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1` where Chromium is
already provided; `run.mjs` takes the binary from `CHROME_PATH`.

## The toy checkpoint

`make_toy_checkpoint.py` writes the three files a real HuggingFace checkpoint
has — `config.json`, `model.safetensors`, `tokenizer.json` — for a 32-wide,
2-layer body over a 30-token vocabulary. 146 KB rather than 90 MB, and no
network.

Nothing this harness measures depends on the weights being any good. Threads,
entropy, event-loop yielding and memory are properties of the code, not the
checkpoint, so a toy body exercises them exactly as a real one would. It follows
HuggingFace naming and PyTorch's `[out, in]` Linear layout, so the loader's key
remap and transpose run too, and it pads to 128 tokens like a published
`tokenizer.json` does, so `Tokenizer::from_bytes` has padding to strip.

What it does **not** cover is fidelity: whether the browser reproduces the
published `all-MiniLM-L6-v2` similarity matrix needs the real checkpoint, and
that is the part of #1 still open.

## What the run asserts

| Check | Why |
| ----- | --- |
| No page errors or console errors | A panic in wasm surfaces as `unreachable`, which says nothing on its own — `console_error_panic_hook` turns it into a message |
| Every held-in example classifies to its own label | Proves the whole chain ran, not merely that it returned |
| The long document produced more than one chunk | The chunker and the reducer are on the path, not just a single forward pass |
| Worst main-thread frame gap < 250 ms | The reason `Trainer` is a step-wise state machine rather than a `fit()` loop. A CSS animation would not prove this: it can run on the compositor thread while the main thread is wedged, so the harness clocks `requestAnimationFrame` on the main thread instead |
| A backend this build cannot run is refused, by name and with a reason | The failure mode is a caller who asks for the GPU, is quietly given the CPU, and reads the timings as GPU timings. Checked on `requireBackend` and on the `Trainer` constructor, since the helper is the easier one to forget to call |
