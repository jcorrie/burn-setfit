//! Weight loading for the MiniLM body.
//!
//! Unlike the upstream `minilm-burn` loader this is byte-oriented rather than
//! path-oriented, so the exact same code path serves a native file read and a
//! browser `fetch()` response.

use super::model::MiniLmModel;
use crate::error::SetFitError;
use burn::tensor::backend::Backend;
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

/// Maps HuggingFace BERT parameter names onto Burn's `TransformerEncoder` layout.
///
/// Applied in order, so more specific patterns come first. Adapted from
/// `tracel-ai/models` (`minilm-burn`), MIT OR Apache-2.0.
pub fn hf_key_mappings() -> Vec<(&'static str, &'static str)> {
    vec![
        // Strip the `bert.` prefix used by some checkpoints.
        ("^bert\\.(.+)", "$1"),
        ("encoder\\.layer\\.([0-9]+)", "encoder.layers.$1"),
        // Attention
        ("attention\\.self\\.query", "mha.query"),
        ("attention\\.self\\.key", "mha.key"),
        ("attention\\.self\\.value", "mha.value"),
        ("attention\\.output\\.dense", "mha.output"),
        ("attention\\.output\\.LayerNorm", "norm_1"),
        // Feed-forward. The `layers.N.` anchor keeps this from also matching
        // the attention block's `output.dense`, which was rewritten above.
        ("intermediate\\.dense", "pwff.linear_inner"),
        ("(layers\\.[0-9]+)\\.output\\.dense", "$1.pwff.linear_outer"),
        ("(layers\\.[0-9]+)\\.output\\.LayerNorm", "$1.norm_2"),
        ("embeddings\\.LayerNorm", "embeddings.layer_norm"),
    ]
}

/// Which naming convention a set of body weights uses.
///
/// HuggingFace checkpoints need both a key remap and PyTorch's transposed `Linear`
/// layout undone; weights this crate wrote itself need neither, and applying the
/// adapter to them would transpose correct weights into wrong ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Naming {
    /// `encoder.layer.0.attention.self.query.weight`, PyTorch tensor layout.
    HuggingFace,
    /// `encoder.layers.0.mha.query.weight`, Burn tensor layout.
    Burn,
}

/// Load safetensors weights into an initialised model.
pub fn load_weights<B: Backend>(
    model: &mut MiniLmModel<B>,
    bytes: Vec<u8>,
    naming: Naming,
) -> Result<(), SetFitError> {
    match naming {
        Naming::HuggingFace => {
            let remapper = KeyRemapper::from_patterns(hf_key_mappings())
                .map_err(|e| SetFitError::Store(e.to_string()))?;
            let mut store = SafetensorsStore::from_bytes(Some(bytes))
                .with_from_adapter(PyTorchToBurnAdapter)
                .remap(remapper);
            model
                .load_from(&mut store)
                .map_err(|e| SetFitError::Store(e.to_string()))?;
        }
        Naming::Burn => {
            let mut store = SafetensorsStore::from_bytes(Some(bytes));
            model
                .load_from(&mut store)
                .map_err(|e| SetFitError::Store(e.to_string()))?;
        }
    }
    Ok(())
}

/// Load HuggingFace-format safetensors weights into an initialised model.
pub fn load_hf_weights<B: Backend>(
    model: &mut MiniLmModel<B>,
    bytes: Vec<u8>,
) -> Result<(), SetFitError> {
    load_weights(model, bytes, Naming::HuggingFace)
}
