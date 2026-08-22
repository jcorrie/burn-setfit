//! MiniLM sentence-transformer body.
//!
//! Vendored and adapted from [`tracel-ai/models`](https://github.com/tracel-ai/models)
//! (`minilm-burn`, MIT OR Apache-2.0). Vendored rather than depended upon because
//! that crate pins `tokenizers` with the `onig` feature, and Cargo features are
//! additive — an upstream `onig` cannot be switched off downstream, and oniguruma
//! does not build for `wasm32-unknown-unknown`. The adaptations here are:
//!
//! - byte-based weight and config loading (no `std::fs`, no `hf-hub`)
//! - no forced regex backend
//! - `MiniLmEmbeddingsConfig` made public for bundle round-trips

mod embedding;
mod loader;
mod model;
mod pooling;

pub use embedding::*;
pub use loader::*;
pub use model::*;
pub use pooling::*;

/// The two published `all-MiniLM-*-v2` checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MiniLmVariant {
    /// 6 layers, ~22.7M parameters. The sane default for wasm.
    #[default]
    L6,
    /// 12 layers, ~33M parameters. Better quality, ~2x the cost and payload.
    L12,
}

impl MiniLmVariant {
    /// HuggingFace repository id for this variant.
    pub fn model_id(&self) -> &'static str {
        match self {
            MiniLmVariant::L6 => "sentence-transformers/all-MiniLM-L6-v2",
            MiniLmVariant::L12 => "sentence-transformers/all-MiniLM-L12-v2",
        }
    }
}

/// Hidden size of both MiniLM variants, and therefore the head's input width.
pub const EMBEDDING_DIM: usize = 384;

/// Reject a sequence budget the body's position embeddings cannot cover.
///
/// Position ids run `0..seq_len`, so a window longer than the position table
/// indexes past the end of it. Burn reports that as an out-of-bounds `select`
/// deep inside the backend — a panic, not an error, and only once a document
/// long enough to fill a window turns up. Both published MiniLM checkpoints
/// carry 512 positions against a 256-token window, so the gap never closes in
/// practice; a smaller body makes it reachable.
pub fn check_sequence_budget(
    body: &MiniLmConfig,
    max_tokens: usize,
    what: &str,
) -> crate::Result<()> {
    if max_tokens > body.max_position_embeddings {
        return Err(crate::SetFitError::Config(format!(
            "{what} of {max_tokens} tokens exceeds the {} position embeddings this body has; \
             a sequence that long would index past the position table",
            body.max_position_embeddings
        )));
    }
    Ok(())
}

/// Sequence length these checkpoints were actually trained at.
///
/// The position embeddings run to 512, but `sentence_bert_config.json` caps
/// `max_seq_length` at 256 — beyond that you are extrapolating past the
/// training distribution. This is the ceiling the chunker plans against.
pub const TRAINED_SEQ_LEN: usize = 256;
