//! Loading a HuggingFace-format checkpoint, offline.
//!
//! `Checkpoint::from_files` is the browser's path: three files fetched by the
//! caller, in HuggingFace naming with PyTorch's `[out, in]` Linear layout. The
//! key remap and the transpose are otherwise exercised only by the `--ignored`
//! tests against the real download. Here a toy body is rewritten into that
//! format by hand and must load back bit-for-bit.

mod common;

use burn_setfit::checkpoint::Checkpoint;
use burn_setfit::minilm::{MiniLmConfig, MiniLmVariant};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use common::{cpu, toy_body_config, toy_checkpoint, toy_tokenizer_json};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use std::collections::BTreeMap;

/// Every tensor in a safetensors blob, as `name -> (shape, f32 values)`.
fn tensors(bytes: &[u8]) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
    SafeTensors::deserialize(bytes)
        .expect("valid safetensors")
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            assert_eq!(view.dtype(), Dtype::F32, "{name}");
            let (words, _) = view.data().as_chunks::<4>();
            let values = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
            (name, (view.shape().to_vec(), values))
        })
        .collect()
}

/// A Burn parameter name as HuggingFace's BERT checkpoints spell it.
fn hf_name(burn: &str) -> String {
    let mut name = burn
        .replace("encoder.layers.", "encoder.layer.")
        .replace(".mha.query", ".attention.self.query")
        .replace(".mha.key", ".attention.self.key")
        .replace(".mha.value", ".attention.self.value")
        .replace(".mha.output", ".attention.output.dense")
        .replace(".norm_1", ".attention.output.LayerNorm")
        .replace(".pwff.linear_inner", ".intermediate.dense")
        .replace(".pwff.linear_outer", ".output.dense")
        .replace(".norm_2", ".output.LayerNorm")
        .replace("embeddings.layer_norm", "embeddings.LayerNorm");
    // PyTorch's LayerNorm calls its parameters weight and bias.
    if let Some(stem) = name.strip_suffix(".gamma") {
        name = format!("{stem}.weight");
    } else if let Some(stem) = name.strip_suffix(".beta") {
        name = format!("{stem}.bias");
    }
    name
}

/// Is this a `Linear` weight, which PyTorch stores transposed?
fn is_linear_weight(burn: &str) -> bool {
    burn.ends_with(".weight") && (burn.contains(".mha.") || burn.contains(".pwff."))
}

fn transpose(shape: &[usize], values: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let (rows, cols) = (shape[0], shape[1]);
    let mut out = vec![0.0; values.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    (vec![cols, rows], out)
}

/// Rewrite Burn-named weights as a HuggingFace checkpoint would ship them,
/// with `prefix` before every name and an unused pooler, as real ones have.
fn to_huggingface(burn_weights: &[u8], prefix: &str) -> Vec<u8> {
    let mut hf: Vec<(String, Vec<usize>, Vec<u8>)> = tensors(burn_weights)
        .into_iter()
        .map(|(name, (shape, values))| {
            let (shape, values) = if is_linear_weight(&name) {
                transpose(&shape, &values)
            } else {
                (shape, values)
            };
            let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            (format!("{prefix}{}", hf_name(&name)), shape, bytes)
        })
        .collect();
    let hidden = toy_body_config().hidden_size;
    hf.push((
        format!("{prefix}pooler.dense.weight"),
        vec![hidden, hidden],
        vec![0; hidden * hidden * 4],
    ));
    let views: Vec<(String, TensorView<'_>)> = hf
        .iter()
        .map(|(name, shape, bytes)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes).expect("view");
            (name.clone(), view)
        })
        .collect();
    safetensors::serialize(views, None).expect("serialise")
}

/// The body a checkpoint loads, saved back in Burn naming.
fn loaded_body(checkpoint: &Checkpoint) -> Vec<u8> {
    let body = checkpoint.body(&cpu()).expect("the body loads");
    let mut store = SafetensorsStore::from_bytes(None);
    body.save_into(&mut store).expect("saves");
    store.get_bytes().expect("bytes")
}

fn config_json(config: &MiniLmConfig) -> String {
    // A real config.json carries more than the body needs; the extras are ignored.
    format!(
        r#"{{
            "architectures": ["BertModel"],
            "model_type": "bert",
            "hidden_act": "gelu",
            "hidden_size": {},
            "num_attention_heads": {},
            "num_hidden_layers": {},
            "intermediate_size": {},
            "vocab_size": {},
            "max_position_embeddings": {},
            "type_vocab_size": {},
            "hidden_dropout_prob": {},
            "layer_norm_eps": {}
        }}"#,
        config.hidden_size,
        config.num_attention_heads,
        config.num_hidden_layers,
        config.intermediate_size,
        config.vocab_size,
        config.max_position_embeddings,
        config.type_vocab_size,
        config.hidden_dropout_prob,
        config.layer_norm_eps
    )
}

#[test]
fn a_huggingface_checkpoint_loads_into_exactly_the_body_it_describes() {
    let original = toy_checkpoint();
    let expected = tensors(&original.weights);

    for prefix in ["", "bert."] {
        let hf = Checkpoint::from_files(
            MiniLmVariant::L6,
            &config_json(&original.config),
            to_huggingface(&original.weights, prefix),
            toy_tokenizer_json(),
        )
        .expect("from_files accepts a HuggingFace config.json");
        assert_eq!(
            format!("{:?}", hf.config),
            format!("{:?}", original.config),
            "config.json parsed"
        );

        let got = tensors(&loaded_body(&hf));
        assert_eq!(
            got.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>(),
            "prefix {prefix:?}: every parameter loaded, the pooler ignored"
        );
        for (name, want) in &expected {
            assert_eq!(&got[name], want, "prefix {prefix:?}: {name} differs");
        }
    }
}

#[test]
fn burn_named_weights_are_not_remapped_or_transposed() {
    // The mistake `Naming` exists to prevent: running the HuggingFace adapter
    // over weights this crate wrote turns correct weights into wrong ones.
    let original = toy_checkpoint();
    assert_eq!(tensors(&loaded_body(&original)), tensors(&original.weights));
}

#[test]
fn a_config_json_that_is_not_json_is_a_config_error() {
    let err = Checkpoint::from_files(
        MiniLmVariant::L6,
        "{ not json",
        toy_checkpoint().weights,
        toy_tokenizer_json(),
    )
    .unwrap_err();
    assert!(
        matches!(err, burn_setfit::error::SetFitError::Config(_)),
        "{err:?}"
    );
}

#[test]
fn each_variant_names_its_huggingface_repository() {
    assert_eq!(
        MiniLmVariant::L6.model_id(),
        "sentence-transformers/all-MiniLM-L6-v2"
    );
    assert_eq!(
        MiniLmVariant::L12.model_id(),
        "sentence-transformers/all-MiniLM-L12-v2"
    );
}
