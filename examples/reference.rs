//! An independent BERT forward pass, to check the Burn pipeline against.
//!
//! This reads the safetensors directly and computes `all-MiniLM-L6-v2` with plain
//! `Vec<f32>` arithmetic — no Burn, no shared code with the library beyond loading
//! the file. If the two agree, the vendored body composes its (verified-correct)
//! weights the way BERT does. If they diverge, this says where.
//!
//! Run with: `cargo run --release --example reference`

use burn::backend::NdArray;
use burn_setfit::checkpoint::Checkpoint;
use burn_setfit::minilm::MiniLmVariant;
use burn_setfit::model::embed_body;
use burn_setfit::tokenize::pad_batch;
use safetensors::SafeTensors;

type B = NdArray<f32>;

struct Weights<'a> {
    st: SafeTensors<'a>,
}

impl<'a> Weights<'a> {
    fn get(&self, name: &str) -> Vec<f32> {
        let t = self
            .st
            .tensor(name)
            .unwrap_or_else(|_| panic!("missing tensor {name}"));
        t.data()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

/// `y[i][o] = Σ_k x[i][k] · w[o][k] + b[o]`, with PyTorch's `[out, in]` layout.
fn linear(x: &[f32], rows: usize, w: &[f32], b: &[f32], d_in: usize, d_out: usize) -> Vec<f32> {
    let mut y = vec![0.0; rows * d_out];
    for i in 0..rows {
        for o in 0..d_out {
            let mut acc = b[o] as f64;
            for k in 0..d_in {
                acc += (x[i * d_in + k] * w[o * d_in + k]) as f64;
            }
            y[i * d_out + o] = acc as f32;
        }
    }
    y
}

fn layer_norm(x: &mut [f32], rows: usize, d: usize, gamma: &[f32], beta: &[f32], eps: f32) {
    for i in 0..rows {
        let row = &mut x[i * d..(i + 1) * d];
        let mean = row.iter().map(|v| *v as f64).sum::<f64>() / d as f64;
        let var = row.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
        let denom = (var + eps as f64).sqrt();
        for (j, v) in row.iter_mut().enumerate() {
            *v = (((*v as f64 - mean) / denom) as f32) * gamma[j] + beta[j];
        }
    }
}

fn gelu(x: f32) -> f32 {
    // The exact erf formulation, which is what BERT's `hidden_act: "gelu"` means.
    0.5 * x * (1.0 + erf(x / core::f32::consts::SQRT_2))
}

/// Abramowitz & Stegun 7.1.26 — ample precision for a cross-check.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1_f32 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row = &mut x[i * cols..(i + 1) * cols];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v as f64;
        }
        for v in row.iter_mut() {
            *v = (*v as f64 / sum) as f32;
        }
    }
}

/// Full reference forward for one sequence, returning the mean-pooled unit vector.
fn reference_embed(w: &Weights, ids: &[u32], cfg: &burn_setfit::minilm::MiniLmConfig) -> Vec<f32> {
    let n = ids.len();
    let d = cfg.hidden_size;
    let heads = cfg.num_attention_heads;
    let dh = d / heads;
    let eps = cfg.layer_norm_eps as f32;

    // Embeddings: word + position + token type, then LayerNorm.
    let word = w.get("embeddings.word_embeddings.weight");
    let pos = w.get("embeddings.position_embeddings.weight");
    let typ = w.get("embeddings.token_type_embeddings.weight");
    let mut x = vec![0.0f32; n * d];
    for (i, &id) in ids.iter().enumerate() {
        for j in 0..d {
            x[i * d + j] = word[id as usize * d + j] + pos[i * d + j] + typ[j];
        }
    }
    layer_norm(
        &mut x,
        n,
        d,
        &w.get("embeddings.LayerNorm.weight"),
        &w.get("embeddings.LayerNorm.bias"),
        eps,
    );

    for l in 0..cfg.num_hidden_layers {
        let p = format!("encoder.layer.{l}");

        let q = linear(
            &x,
            n,
            &w.get(&format!("{p}.attention.self.query.weight")),
            &w.get(&format!("{p}.attention.self.query.bias")),
            d,
            d,
        );
        let k = linear(
            &x,
            n,
            &w.get(&format!("{p}.attention.self.key.weight")),
            &w.get(&format!("{p}.attention.self.key.bias")),
            d,
            d,
        );
        let v = linear(
            &x,
            n,
            &w.get(&format!("{p}.attention.self.value.weight")),
            &w.get(&format!("{p}.attention.self.value.bias")),
            d,
            d,
        );

        // Per-head scaled dot-product attention. No padding here: one sequence,
        // no batch, so every position is real.
        let mut context = vec![0.0f32; n * d];
        for h in 0..heads {
            let off = h * dh;
            let mut scores = vec![0.0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for t in 0..dh {
                        acc += (q[i * d + off + t] * k[j * d + off + t]) as f64;
                    }
                    scores[i * n + j] = (acc / (dh as f64).sqrt()) as f32;
                }
            }
            softmax_rows(&mut scores, n, n);
            for i in 0..n {
                for t in 0..dh {
                    let mut acc = 0.0f64;
                    for j in 0..n {
                        acc += (scores[i * n + j] * v[j * d + off + t]) as f64;
                    }
                    context[i * d + off + t] = acc as f32;
                }
            }
        }

        // Attention output projection, residual, LayerNorm (post-LN).
        let attn = linear(
            &context,
            n,
            &w.get(&format!("{p}.attention.output.dense.weight")),
            &w.get(&format!("{p}.attention.output.dense.bias")),
            d,
            d,
        );
        for i in 0..n * d {
            x[i] += attn[i];
        }
        layer_norm(
            &mut x,
            n,
            d,
            &w.get(&format!("{p}.attention.output.LayerNorm.weight")),
            &w.get(&format!("{p}.attention.output.LayerNorm.bias")),
            eps,
        );

        // Feed-forward, residual, LayerNorm.
        let inter = cfg.intermediate_size;
        let mut hidden = linear(
            &x,
            n,
            &w.get(&format!("{p}.intermediate.dense.weight")),
            &w.get(&format!("{p}.intermediate.dense.bias")),
            d,
            inter,
        );
        for v in hidden.iter_mut() {
            *v = gelu(*v);
        }
        let ff = linear(
            &hidden,
            n,
            &w.get(&format!("{p}.output.dense.weight")),
            &w.get(&format!("{p}.output.dense.bias")),
            inter,
            d,
        );
        for i in 0..n * d {
            x[i] += ff[i];
        }
        layer_norm(
            &mut x,
            n,
            d,
            &w.get(&format!("{p}.output.LayerNorm.weight")),
            &w.get(&format!("{p}.output.LayerNorm.bias")),
            eps,
        );
    }

    // Mean pool over all real tokens, then L2 normalise.
    let mut pooled = vec![0.0f32; d];
    for i in 0..n {
        for j in 0..d {
            pooled[j] += x[i * d + j];
        }
    }
    for v in pooled.iter_mut() {
        *v /= n as f32;
    }
    let norm = pooled.iter().map(|v| (v * v) as f64).sum::<f64>().sqrt() as f32;
    for v in pooled.iter_mut() {
        *v /= norm;
    }
    pooled
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    let checkpoint = Checkpoint::download(MiniLmVariant::L6, None)?;
    let tokenizer = checkpoint.tokenizer()?;
    let body = checkpoint.body::<B>(&device)?;
    let weights = Weights {
        st: SafeTensors::deserialize(&checkpoint.weights)?,
    };

    let sentences = [
        "The cat sat on the mat.",
        "A feline rested upon the rug.",
        "Quantum chromodynamics describes the strong interaction.",
    ];

    let mut ours = Vec::new();
    let mut theirs = Vec::new();
    for s in &sentences {
        let ids = tokenizer.encode_full(s, 256)?;

        // Burn path, one sentence at a time so no padding is involved.
        let (input_ids, mask) = pad_batch::<B>(core::slice::from_ref(&ids), 0, &device);
        let e = embed_body(&body, input_ids, mask);
        ours.push(e.into_data().into_vec::<f32>().unwrap());

        theirs.push(reference_embed(&weights, &ids, &checkpoint.config));
    }

    println!("agreement between Burn and the reference implementation:");
    let mut worst: f32 = 1.0;
    for (i, s) in sentences.iter().enumerate() {
        let agreement = cos(&ours[i], &theirs[i]);
        let max_abs = ours[i]
            .iter()
            .zip(&theirs[i])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.min(agreement);
        println!("  cos {agreement:.6}   max |Δ| {max_abs:.6}   {s}");
    }

    println!("\ncosine similarities, Burn vs reference:");
    for i in 0..sentences.len() {
        for j in i + 1..sentences.len() {
            println!(
                "  [{i}]~[{j}]   burn {:.4}   reference {:.4}",
                cos(&ours[i], &ours[j]),
                cos(&theirs[i], &theirs[j])
            );
        }
    }

    if worst > 0.999 {
        println!("\nOK: the Burn pipeline matches an independent BERT forward pass.");
        Ok(())
    } else {
        Err(format!("pipelines disagree (worst cos {worst:.6})").into())
    }
}
