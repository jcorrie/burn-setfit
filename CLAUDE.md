# burn-setfit

SetFit few-shot text classification in Rust on [Burn](https://github.com/tracel-ai/burn),
running natively and in the browser, over documents of unbounded length.

Read `README.md` for what the crate does and why. This file is for working on it.

## The loop

```bash
cargo test --features ndarray,train,native
```

148 tests, about two seconds, no network. Stay in this loop. Anything touching
`--ignored` or the examples downloads ~90 MB of MiniLM and trains, so it is slow
and a poor fit for a constrained connection.

```bash
cargo clippy --all-targets --features ndarray,train,native   # expected: zero warnings
cargo build --target wasm32-unknown-unknown --no-default-features --features ndarray,train
cargo build --target wasm32-unknown-unknown --no-default-features --features wgpu-wasm-unverified,train
cargo build --release --target wasm32-unknown-unknown -p setfit-wasm
```

The `wgpu` backend has its own suite, kept out of the default run because it
needs the feature and a Vulkan adapter:

```bash
cargo test --features ndarray,wgpu,train --test wgpu    # 4 tests, ~20 s
```

No GPU is needed. Mesa's lavapipe (`apt-get install mesa-vulkan-drivers`,
providing `/usr/share/vulkan/icd.d/lvp_icd.json`) is a software Vulkan 1.4
implementation, and wgpu runs on it as `llvmpipe`, `device_type: Cpu`. That
exercises the whole backend — naga codegen, buffers, dispatch, readback — and
is what catches a wgpu path that does not run or disagrees with `ndarray`. It
proves nothing about real hardware or about WebGPU in a browser.

Before anything that touches the browser path, run the harness — it catches what
compiling cannot:

```bash
cd browser-test && node run.mjs     # see browser-test/README.md for the setup
```

**Both wasm targets must keep building.** That is the point of the crate, and it
is easy to break by adding a dependency that assumes a filesystem or threads.
`native` is the only feature allowed to do either, and it is confined to
`Checkpoint::download`.

The `wgpu-wasm-unverified` build is there to keep the async path honest: it is
the only configuration where a blocking readback is a real failure rather than a
slow one, so it catches an `into_data()` that crept back in. It has never been
executed on a GPU — the feature's name is load-bearing, do not rename it to
something reassuring.

## Things that will bite

These are all mistakes already made once here. Each is guarded by a test; the
test names are given so a failure is self-explaining.

- **Tokenizer padding.** `Tokenizer::from_bytes` strips padding and truncation
  baked into `tokenizer.json`. `all-MiniLM-L6-v2` pads to 128, and this crate
  derives the attention mask from sequence length — so upstream padding is
  indistinguishable from content by the time that mask is built. Leaving it in
  made every sentence ~94% padding and pushed unrelated-sentence similarity from
  0.10 to 0.90. Nothing fails to compile if it comes back.
  Guard: `tests/pretrained.rs::tokenizer_padding_is_stripped_on_load`.

- **`toy_checkpoint()` returns a different random model every call.** Any test
  comparing two models must share one bundle. `toy_bundle` memoises for this
  reason, and holds its lock *across training* — releasing it lets two threads
  each train a different model and compare them, which fails intermittently.

- **Reducer ordering is not a theorem in single-label space.** `max >= mean`
  holds per label in logit space, but single-label scores are a softmax, which
  depends on the *gap* between logits; raising both can narrow it. The property
  only survives under a monotone link, i.e. multi-label sigmoids.
  Guard: `tests/infer.rs::stronger_reducers_dominate_per_label_under_multi_label`.

- **Chunk windows must fit the body's position table.** Position ids run
  `0..seq_len`, so a window longer than `max_position_embeddings` indexes past
  the table and Burn reports it as an out-of-bounds `select` — a panic from
  inside the backend, and only once a document long enough to fill a window
  turns up. Both MiniLM checkpoints have 512 positions against a 256-token
  window, so this is unreachable with a real body and immediate with a small
  one. `check_sequence_budget` rejects it at `Trainer::new` and
  `Manifest::validate`; `toy_body_config` carries 512 positions for the same
  reason. Guards:
  `tests/bundle.rs::a_manifest_whose_windows_outrun_the_position_table_is_refused`,
  `tests/train.rs::a_sequence_budget_beyond_the_bodys_positions_is_rejected_before_training`.

- **Seeded runs reproduce weights, not bytes.** Burn writes the safetensors
  `__metadata__` map in `HashMap` order. Compare tensors, never file hashes.

- **`min_final_tokens` governs the trailing chunk only.** Intermediate windows are
  boundary-aligned and routinely short; a blanket minimum would discard ordinary
  content.

- **`Tensor::into_data()` is a panic on wasm with a GPU backend.** It polls the
  read future once and gives up; a browser cannot block on a buffer map, so the
  future is always pending there. Lazy `WgpuDevice` acquisition goes through the
  same `block_on`, which is why the trap lands while a tensor is being *built*,
  before any readback, and reports only `unreachable`. Every readback therefore
  goes through `src/readback.rs`, and the `_async` methods are the real
  implementations with the blocking ones as wrappers — write a new one that way
  round, or the two paths drift. Guards:
  `tests/async_api.rs::classify_async_agrees_with_classify`,
  `tests/async_api.rs::stepping_asynchronously_trains_the_same_model`.

- **An `async fn` that awaits itself does not compile.** `step()` used to recurse
  into itself to hand the caller a head step once the body stage ended; the
  future that produces would be infinitely sized. The stage transition is a
  fall-through in `step_async` for that reason, with the epoch bookkeeping
  pulled out into `body_epoch_rollover` so it happens before any future exists.
  Guard: `tests/async_api.rs::step_async_crosses_the_body_to_head_boundary`.

## Conventions

- Validation lives on the config types and is called at every entry point that
  consumes one (`Bundle::pack`, `Classifier::new`, `Trainer::new`). Prefer
  rejecting a bad value to silently clamping it — reducers used to clamp
  `k.max(1)` at point of use, which turned a config error into a model quietly
  doing something else.
- One error type crosses the whole crate, and `?` composes with `std::io`.
  `SetFitError::Io` keeps the underlying error rather than flattening it, so
  `ErrorKind` still works; the variants built from foreign errors fold those
  messages into their own, because naming the file or the example is worth more
  than the chain. The enum is `#[non_exhaustive]` — adding a variant is not a
  breaking change, and `Io` was added after the fact.
- Make illegal states unrepresentable where it is cheap. The multi-label
  threshold lives *inside* `TaskMode::MultiLabel` because an argmax has nothing
  to threshold.
- Comments explain *why*, especially where the code looks odd on purpose. Several
  are load-bearing — the ones above exist because the alternative was silent
  wrongness.
- Reduction, hierarchy and threshold are decode-time choices. Adding a knob that
  needs retraining when it could be a `with_*` on `Classifier` is a regression.

## Verifying model changes

Anything touching `src/minilm/`, `src/tokenize.rs`, or pooling must still
reproduce the published `all-MiniLM-L6-v2` similarity matrix to four decimals:

```bash
cargo test --release --features ndarray,train,native -- --ignored
```

`examples/reference.rs` is an independent BERT forward pass written from the raw
safetensors, for bisecting a disagreement between Burn and the checkpoint. Note
it once agreed with Burn while both were wrong, because both consumed the same
bad token ids — agreement with it is necessary, not sufficient. Ground truth is
the published matrix.

## Open work

See [issues](https://github.com/jcorrie/burn-setfit/issues). #1 is now half
answered — `browser-test/` proves the wasm trains and classifies in a browser,
but only against a toy checkpoint; the real model has never been fetched into a
page. #2 (quantize the 86.5 MB bundle) is the other one that matters; the rest
are smaller or speculative.
