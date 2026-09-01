//! Shrinking a bundle's weights at rest.
//!
//! A `.setfit` bundle for MiniLM-L6 is 86.5 MB against 8.5 MB of wasm, so the
//! weights are the payload and everything else is rounding. Roughly half of
//! those parameters are the `30522 x 384` word-embedding table, which is
//! exactly the kind of tensor that survives losing precision.
//!
//! # Why this is storage, not compute
//!
//! Burn has [`Tensor::quantize`](burn::tensor::Tensor::quantize), which
//! produces a `QFloat` tensor that arithmetic then runs *on*. That is a
//! different goal: it needs quantized-op support from every backend the crate
//! claims to run on, and it changes what the model computes. What #2 asks for
//! is a smaller download.
//!
//! So quantization here happens **to the file and nowhere else**. Weights are
//! compressed when a bundle is packed and expanded back to `f32` before Burn
//! ever sees them, which means inference is bit-for-bit the same code on every
//! backend, and the only cost is the precision lost in the round trip. It also
//! means a bundle is not a new tensor format — it is still safetensors, just
//! with narrower dtypes inside.
//!
//! # What gets compressed
//!
//! Not everything, and the distinction is the point. [`worth_compressing`]
//! takes rank-2-and-up float tensors above a size floor: the embedding table
//! and the `Linear` weights, where the bytes actually are. LayerNorm gains and
//! biases are rank 1, tiny, and the parameters least tolerant of precision
//! loss — quantizing them would buy a fraction of a percent and risk the whole
//! forward pass.
//!
//! [`expand`] does not need to be told any of this. It reads each tensor's
//! dtype and reverses whatever it finds, so a blob where some tensors were
//! skipped — or where one was skipped because its values were not finite —
//! round-trips correctly without a policy of its own.

use crate::error::{Result, SetFitError};
use burn::tensor::f16;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::borrow::Cow;

/// Scales for an int8 tensor travel beside it, under this prefix.
///
/// In the blob rather than the manifest: `int8` is scaled per row, and the
/// embedding table has 30522 of them. As JSON in the manifest that is a few
/// hundred KB of text on a file whose entire purpose is being smaller.
/// [`expand`] consumes these, so Burn never sees a tensor it does not
/// recognise.
const SCALE_PREFIX: &str = "__q_scale__";

/// Below this many elements, a tensor is not worth the loss.
///
/// Every `Linear` weight in MiniLM-L6 is far above it and every norm and bias
/// far below, so the floor is doing less work than [`worth_compressing`]'s rank
/// test — it is here for a checkpoint whose shapes are not MiniLM's.
const MIN_ELEMENTS: usize = 4096;

/// How a bundle's weights are stored.
///
/// Decode-time behaviour is unaffected: whatever is chosen here, the model that
/// comes back is `f32`. The choice trades bundle size against how far the
/// weights drift from the ones that were trained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Quantization {
    /// Exactly the trained weights. The default, and what every bundle written
    /// before this existed contains.
    #[default]
    None,
    /// IEEE half precision. Halves the weights, and for a model whose values
    /// sit well inside half's range the error is a rounding step.
    F16,
    /// 8-bit integers with a per-row `f32` scale. Quarters the weights.
    Int8,
}

impl Quantization {
    /// Whether this mode changes the bytes at all.
    pub fn is_lossless(self) -> bool {
        self == Quantization::None
    }

    /// The name used in the manifest and in error messages.
    pub fn as_str(self) -> &'static str {
        match self {
            Quantization::None => "none",
            Quantization::F16 => "f16",
            Quantization::Int8 => "int8",
        }
    }
}

/// Whether a tensor is worth compressing.
///
/// See the module docs: this is where the size is, and skipping the rest costs
/// almost nothing while keeping the delicate parameters exact.
fn worth_compressing(view: &TensorView<'_>) -> bool {
    view.dtype() == Dtype::F32
        && view.shape().len() >= 2
        && view.shape().iter().product::<usize>() >= MIN_ELEMENTS
}

fn store_err(e: impl core::fmt::Display) -> SetFitError {
    SetFitError::Store(format!("safetensors: {e}"))
}

/// Read an `f32` tensor's data. Safetensors is little-endian on the wire.
fn f32_data(view: &TensorView<'_>) -> Vec<f32> {
    view.data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// One tensor, ready to be handed back to `safetensors::serialize`.
struct Owned {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl safetensors::tensor::View for Owned {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Rewrite an `f32` safetensors blob with narrower weights.
///
/// A tensor that [`worth_compressing`] rejects, or whose values are not finite,
/// is copied through untouched — so this never turns a broken checkpoint into a
/// differently broken one, and [`expand`] still reverses the result.
pub(crate) fn compress(weights: &[u8], mode: Quantization) -> Result<Vec<u8>> {
    if mode.is_lossless() {
        return Ok(weights.to_vec());
    }

    let parsed = SafeTensors::deserialize(weights).map_err(store_err)?;
    let (_, metadata) = SafeTensors::read_metadata(weights).map_err(store_err)?;

    let mut out: Vec<(String, Owned)> = Vec::new();

    for (name, view) in parsed.iter() {
        if !worth_compressing(&view) {
            out.push((
                name.to_string(),
                Owned {
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    data: view.data().to_vec(),
                },
            ));
            continue;
        }

        let values = f32_data(&view);
        // A non-finite weight is a broken model, and finding out at pack time
        // by silently rescaling everything to inf would be the worst way to
        // learn it. Leave the tensor alone and let it stay diagnosable.
        if !values.iter().all(|v| v.is_finite()) {
            out.push((
                name.to_string(),
                Owned {
                    dtype: Dtype::F32,
                    shape: view.shape().to_vec(),
                    data: view.data().to_vec(),
                },
            ));
            continue;
        }

        match mode {
            Quantization::None => unreachable!("handled above"),
            Quantization::F16 => {
                let mut data = Vec::with_capacity(values.len() * 2);
                for v in &values {
                    data.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
                }
                out.push((
                    name.to_string(),
                    Owned {
                        dtype: Dtype::F16,
                        shape: view.shape().to_vec(),
                        data,
                    },
                ));
            }
            Quantization::Int8 => {
                let shape = view.shape().to_vec();
                let rows = shape[0];
                let row_len = values.len() / rows.max(1);

                let mut data = Vec::with_capacity(values.len());
                let mut scales = Vec::with_capacity(rows * 4);

                for row in values.chunks(row_len.max(1)) {
                    // Symmetric, per row. Per row rather than per tensor
                    // because the embedding table's rows are individual
                    // tokens: one shared scale would let the largest vector in
                    // the vocabulary set the resolution for all 30522.
                    let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    // An all-zero row has no scale to speak of. Any non-zero
                    // value works; 1.0 keeps the dequantized zeros exact.
                    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                    scales.extend_from_slice(&scale.to_le_bytes());
                    for v in row {
                        let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
                        data.push(q as u8);
                    }
                }

                out.push((
                    name.to_string(),
                    Owned {
                        dtype: Dtype::I8,
                        shape,
                        data,
                    },
                ));
                out.push((
                    format!("{SCALE_PREFIX}{name}"),
                    Owned {
                        dtype: Dtype::F32,
                        shape: vec![rows],
                        data: scales,
                    },
                ));
            }
        }
    }

    // Sorted so the header key order is at least stable within a run. It is
    // still not canonical across runs — see the note in `bundle.rs` and #6.
    out.sort_by(|a, b| a.0.cmp(&b.0));
    safetensors::serialize(out, metadata.metadata().clone()).map_err(store_err)
}

/// Undo [`compress`], returning a blob whose float tensors are all `f32`.
///
/// Driven entirely by what is in the blob rather than by the mode that produced
/// it, so a mixed blob — some tensors compressed, some skipped — comes back
/// correctly.
pub(crate) fn expand(weights: &[u8]) -> Result<Vec<u8>> {
    let parsed = SafeTensors::deserialize(weights).map_err(store_err)?;
    let (_, metadata) = SafeTensors::read_metadata(weights).map_err(store_err)?;

    let mut out: Vec<(String, Owned)> = Vec::new();

    for (name, view) in parsed.iter() {
        // Scales are consumed by the tensors they belong to.
        if name.starts_with(SCALE_PREFIX) {
            continue;
        }

        let restored = match view.dtype() {
            Dtype::F16 => {
                let mut data = Vec::with_capacity(view.data().len() * 2);
                for b in view.data().chunks_exact(2) {
                    let h = f16::from_le_bytes([b[0], b[1]]);
                    data.extend_from_slice(&h.to_f32().to_le_bytes());
                }
                Owned {
                    dtype: Dtype::F32,
                    shape: view.shape().to_vec(),
                    data,
                }
            }
            Dtype::I8 => {
                let scale_name = format!("{SCALE_PREFIX}{name}");
                let scales = parsed.tensor(&scale_name).map_err(|_| {
                    SetFitError::Bundle(format!(
                        "quantized tensor `{name}` has no `{scale_name}` beside it, so it \
                         cannot be decoded. The bundle is corrupt or was written by a build \
                         that quantizes differently."
                    ))
                })?;
                let scales = f32_data(&scales);

                let raw = view.data();
                let rows = scales.len();
                let row_len = if rows == 0 { 0 } else { raw.len() / rows };

                let mut data = Vec::with_capacity(raw.len() * 4);
                for (r, row) in raw.chunks(row_len.max(1)).enumerate() {
                    let scale = scales.get(r).copied().unwrap_or(1.0);
                    for q in row {
                        let v = (*q as i8) as f32 * scale;
                        data.extend_from_slice(&v.to_le_bytes());
                    }
                }
                Owned {
                    dtype: Dtype::F32,
                    shape: view.shape().to_vec(),
                    data,
                }
            }
            _ => Owned {
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
                data: view.data().to_vec(),
            },
        };
        out.push((name.to_string(), restored));
    }

    out.sort_by(|a, b| a.0.cmp(&b.0));
    safetensors::serialize(out, metadata.metadata().clone()).map_err(store_err)
}
