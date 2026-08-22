//! A MiniLM checkpoint held in memory: config, weights and tokenizer together.
//!
//! Deliberately backend- and platform-agnostic. Natively the three files come from
//! a HuggingFace download; in a browser they come from three `fetch` calls; in a
//! test they are generated. Every consumer below this point sees the same type, so
//! nothing downstream needs a `cfg` for where the bytes came from — which is what
//! kept the browser and native paths from drifting apart.

use crate::error::{Result, SetFitError};
use crate::minilm::{MiniLmConfig, MiniLmModel, MiniLmVariant, Naming, load_weights};
use crate::tokenize::Tokenizer;
use burn::tensor::backend::Backend;

/// Everything needed to start from a pretrained body.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Which checkpoint this is.
    pub variant: MiniLmVariant,
    /// Body architecture.
    pub config: MiniLmConfig,
    /// Safetensors weights, in the convention named by [`Self::naming`].
    pub weights: Vec<u8>,
    /// Raw `tokenizer.json`.
    pub tokenizer_json: Vec<u8>,
    /// Which parameter-naming convention [`Self::weights`] uses.
    pub naming: Naming,
}

impl Checkpoint {
    /// Assemble from an already-parsed config and HuggingFace-format weights.
    pub fn new(
        variant: MiniLmVariant,
        config: MiniLmConfig,
        weights: Vec<u8>,
        tokenizer_json: Vec<u8>,
    ) -> Self {
        Self {
            variant,
            config,
            weights,
            tokenizer_json,
            naming: Naming::HuggingFace,
        }
    }

    /// Assemble from the three raw files of a HuggingFace repo.
    ///
    /// This is the browser path: `config.json`, `model.safetensors` and
    /// `tokenizer.json`, fetched by the caller and handed over as bytes.
    pub fn from_files(
        variant: MiniLmVariant,
        config_json: &str,
        weights: Vec<u8>,
        tokenizer_json: Vec<u8>,
    ) -> Result<Self> {
        Ok(Self::new(
            variant,
            MiniLmConfig::from_hf_json(config_json)?,
            weights,
            tokenizer_json,
        ))
    }

    /// Note that the weights use Burn's own parameter naming, not HuggingFace's.
    ///
    /// Weights this crate wrote need no key remap and no transpose; running the
    /// HuggingFace adapter over them would turn correct weights into wrong ones.
    pub fn with_burn_naming(mut self) -> Self {
        self.naming = Naming::Burn;
        self
    }

    /// Initialise a body and load the weights into it.
    pub fn body<B: Backend>(&self, device: &B::Device) -> Result<MiniLmModel<B>> {
        let mut model = self.config.init(device);
        load_weights(&mut model, self.weights.clone(), self.naming)?;
        Ok(model)
    }

    /// Build the tokenizer.
    pub fn tokenizer(&self) -> Result<Tokenizer> {
        Tokenizer::from_bytes(&self.tokenizer_json)
    }

    /// Reject a checkpoint that could not produce a usable body.
    pub fn validate(&self) -> Result<()> {
        if self.weights.is_empty() {
            return Err(SetFitError::Config("checkpoint has no weights".into()));
        }
        if self.tokenizer_json.is_empty() {
            return Err(SetFitError::Config("checkpoint has no tokenizer".into()));
        }
        if !self
            .config
            .hidden_size
            .is_multiple_of(self.config.num_attention_heads)
        {
            return Err(SetFitError::Config(format!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                self.config.hidden_size, self.config.num_attention_heads
            )));
        }
        Ok(())
    }

    /// Download from HuggingFace, or read from the local cache.
    #[cfg(feature = "native")]
    pub fn download(variant: MiniLmVariant, cache_dir: Option<std::path::PathBuf>) -> Result<Self> {
        use std::path::PathBuf;

        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("burn-models")
        });
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .build()
            .map_err(|e| SetFitError::Download(format!("could not reach HuggingFace: {e}")))?;
        let repo = api.model(variant.model_id().to_string());

        let read = |name: &str| -> Result<Vec<u8>> {
            let path = repo
                .get(name)
                .map_err(|e| SetFitError::Download(format!("{name}: {e}")))?;
            std::fs::read(&path)
                .map_err(|e| SetFitError::Download(format!("{}: {e}", path.display())))
        };

        let config_json = read("config.json")?;
        let config_json = core::str::from_utf8(&config_json)
            .map_err(|e| SetFitError::Config(format!("config.json is not UTF-8: {e}")))?;

        Self::from_files(
            variant,
            config_json,
            read("model.safetensors")?,
            read("tokenizer.json")?,
        )
    }
}
