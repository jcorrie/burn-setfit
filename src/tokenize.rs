//! Tokenizer wrapper.
//!
//! Thin layer over HuggingFace `tokenizers`, with two jobs beyond delegation:
//! keeping special-token handling explicit (the chunker packs raw token ids and
//! only wraps them in `[CLS]`/`[SEP]` at the end, so it must not double-add), and
//! keeping us off the parallel code paths that do not survive the browser.

use crate::error::{Result, SetFitError};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// The special token ids a BERT-family tokenizer needs to build model input.
#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    /// Classification token prepended to every sequence.
    pub cls: u32,
    /// Separator token appended to every sequence.
    pub sep: u32,
    /// Padding token used to square off a batch.
    pub pad: u32,
}

/// A loaded WordPiece tokenizer.
#[derive(Clone)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    special: SpecialTokens,
}

impl Tokenizer {
    /// Build from the contents of a `tokenizer.json`.
    ///
    /// Any padding or truncation baked into the file is stripped. Published
    /// `tokenizer.json` files often pad to a fixed length — `all-MiniLM-L6-v2`
    /// pads to 128 — and that is actively wrong here: this crate packs raw token
    /// ids into windows and pads once per batch, deriving the attention mask from
    /// each sequence's true length. Padding applied upstream is indistinguishable
    /// from content by the time it reaches that mask, so the encoder would attend
    /// to it and mean pooling would average it in.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| SetFitError::Tokenizer(e.to_string()))?;
        inner.with_padding(None);
        inner
            .with_truncation(None)
            .map_err(|e| SetFitError::Tokenizer(e.to_string()))?;

        let id = |t: &str| {
            inner
                .token_to_id(t)
                .ok_or_else(|| SetFitError::Tokenizer(format!("vocab is missing {t}")))
        };
        let special = SpecialTokens {
            cls: id("[CLS]")?,
            sep: id("[SEP]")?,
            pad: id("[PAD]")?,
        };

        Ok(Self { inner, special })
    }

    /// The special token ids for this vocabulary.
    pub fn special_tokens(&self) -> SpecialTokens {
        self.special
    }

    /// Tokenize without adding `[CLS]`/`[SEP]`.
    ///
    /// The chunker packs many of these together before wrapping the result once,
    /// so adding specials here would corrupt both the packing arithmetic and the
    /// resulting sequence.
    pub fn encode_bare(&self, text: &str) -> Result<Vec<u32>> {
        // Deliberately `encode`, not `encode_batch`: the batch path goes through
        // rayon, which has no threads to spawn under wasm32-unknown-unknown.
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| SetFitError::Tokenizer(e.to_string()))
    }

    /// Wrap packed token ids into a full model input sequence.
    pub fn add_specials(&self, ids: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(ids.len() + 2);
        out.push(self.special.cls);
        out.extend_from_slice(ids);
        out.push(self.special.sep);
        out
    }

    /// Tokenize a short text into a complete model input sequence.
    ///
    /// For training examples and other already-short inputs; long documents go
    /// through the chunker instead.
    pub fn encode_full(&self, text: &str, max_tokens: usize) -> Result<Vec<u32>> {
        let mut ids = self.encode_bare(text)?;
        // Leave room for [CLS] and [SEP].
        ids.truncate(max_tokens.saturating_sub(2));
        Ok(self.add_specials(&ids))
    }
}

/// Pad a batch of variable-length sequences into `(input_ids, attention_mask)`.
///
/// Both tensors are `[batch_size, max_len]`; the mask is 1.0 for real tokens and
/// 0.0 for padding, matching what the encoder's pad-mask conversion expects.
pub fn pad_batch<B: Backend>(
    sequences: &[Vec<u32>],
    pad_id: u32,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
    let batch_size = sequences.len();
    let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);

    let mut ids = vec![pad_id as i64; batch_size * max_len];
    let mut mask = vec![0.0f32; batch_size * max_len];

    for (i, seq) in sequences.iter().enumerate() {
        for (j, &id) in seq.iter().enumerate() {
            ids[i * max_len + j] = id as i64;
            mask[i * max_len + j] = 1.0;
        }
    }

    let input_ids =
        Tensor::<B, 1, Int>::from_data(ids.as_slice(), device).reshape([batch_size, max_len]);
    let attention_mask =
        Tensor::<B, 1>::from_data(mask.as_slice(), device).reshape([batch_size, max_len]);

    (input_ids, attention_mask)
}
