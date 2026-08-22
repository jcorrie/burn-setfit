//! The trained artifact: a MiniLM body paired with a classification head.

use crate::head::{SetFitHead, SetFitHeadConfig};
use crate::minilm::{MiniLmConfig, MiniLmModel, mean_pooling, normalize_l2};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Body and head as one module, so both save and load through a single store
/// under stable `body.*` / `head.*` prefixes.
#[derive(Module, Debug)]
pub struct SetFitModule<B: Backend> {
    /// The sentence-transformer body, contrastively fine-tuned in stage one.
    pub body: MiniLmModel<B>,
    /// The linear classifier fitted in stage two.
    pub head: SetFitHead<B>,
}

/// Encode a padded batch with a bare body, without a head attached.
///
/// Stage one trains the body alone, so it needs this before a head exists.
pub fn embed_body<B: Backend>(
    body: &MiniLmModel<B>,
    input_ids: Tensor<B, 2, Int>,
    attention_mask: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let output = body.forward(input_ids, attention_mask.clone(), None);
    normalize_l2(mean_pooling(output.hidden_states, attention_mask))
}

impl<B: Backend> SetFitModule<B> {
    /// Build from a body config and a label count, with a randomly initialised head.
    pub fn init(body_config: &MiniLmConfig, num_labels: usize, device: &B::Device) -> Self {
        Self {
            body: body_config.init(device),
            head: SetFitHeadConfig::new(body_config.hidden_size, num_labels).init(device),
        }
    }

    /// Encode a padded batch into L2-normalised sentence embeddings.
    ///
    /// Mean pooling then L2 normalisation is what `sentence-transformers` does for
    /// the `all-MiniLM-*` checkpoints, and the contrastive stage optimises cosine
    /// similarity, which is only meaningful on unit vectors.
    pub fn embed(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        embed_body(&self.body, input_ids, attention_mask)
    }

    /// Encode a batch and score it, returning logits `[batch, num_labels]`.
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.head.forward(self.embed(input_ids, attention_mask))
    }
}
