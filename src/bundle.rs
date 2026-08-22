//! The `.setfit` bundle: everything needed to classify, in one file.
//!
//! A trained model is four things — body weights, head weights, the tokenizer,
//! and the configuration that ties them together. Shipping four files means four
//! round trips before a browser can classify anything, so they travel as one.
//!
//! The container is deliberately plain: a magic number, a length-prefixed JSON
//! manifest, then two length-prefixed blobs. All little-endian. It parses with a
//! handful of slice reads and no allocator tricks, which is what you want in the
//! one environment where the file arrives as a single `ArrayBuffer`.
//!
//! ```text
//! Note on reproducibility: a seeded training run produces identical *weights*,
//! but not identical *bytes*. Burn writes the safetensors `__metadata__` map in
//! `HashMap` order, and Rust seeds each map differently, so the header keys come
//! out in a different order each time. Compare models by their tensors, not by
//! hashing the file — content-addressing a bundle will not work.
//!
//! ```text
//! "BSETFIT\x00"   8 bytes   magic
//! u32                       format version
//! u32 + bytes               manifest (JSON)
//! u64 + bytes               model weights (safetensors)
//! u64 + bytes               tokenizer.json
//! ```

use crate::config::ClassifierConfig;
use crate::error::{Result, SetFitError};
use crate::head::TaskMode;
use crate::minilm::{MiniLmConfig, MiniLmVariant, check_sequence_budget};
use crate::model::SetFitModule;
use crate::tokenize::Tokenizer;
use burn::tensor::backend::Backend;
use burn_store::{ModuleSnapshot, SafetensorsStore};

const MAGIC: &[u8; 8] = b"BSETFIT\x00";
/// Bundle format version written by this crate.
pub const FORMAT_VERSION: u32 = 1;

/// Everything about a trained model that is not a weight.
///
/// Splits along the line of who chose what: [`Self::classifier`] is the caller's
/// configuration, while [`Self::variant`] and [`Self::body`] describe the
/// checkpoint it was built on and are filled in from that checkpoint.
///
/// ```no_run
/// # use burn::backend::NdArray;
/// # use burn_setfit::Classifier;
/// # fn main() -> burn_setfit::Result<()> {
/// # let classifier = Classifier::<NdArray<f32>>::from_bundle(&[], Default::default())?;
/// let manifest = classifier.manifest();
///
/// println!("{:?} over {:?}", manifest.task(), manifest.labels());
/// println!("body: {:?}, {} wide", manifest.variant, manifest.body.hidden_size);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    /// Which MiniLM checkpoint the body came from.
    pub variant: MiniLmVariant,
    /// Body architecture, so the module can be reconstructed before loading weights.
    pub body: MiniLmConfig,
    /// How the trained model classifies.
    pub classifier: ClassifierConfig,
}

impl Manifest {
    /// Pair a classifier configuration with the checkpoint it was trained on.
    pub fn new(variant: MiniLmVariant, body: MiniLmConfig, classifier: ClassifierConfig) -> Self {
        Self {
            variant,
            body,
            classifier,
        }
    }

    /// Number of classes.
    pub fn num_labels(&self) -> usize {
        self.classifier.num_labels()
    }

    /// Class names, in head-output order.
    pub fn labels(&self) -> &[String] {
        &self.classifier.labels
    }

    /// Single- or multi-label.
    pub fn task(&self) -> TaskMode {
        self.classifier.task
    }

    /// Check the configuration, and that it describes a usable body.
    pub fn validate(&self) -> Result<()> {
        self.classifier.validate()?;
        if self.body.hidden_size == 0 || self.body.num_hidden_layers == 0 {
            return Err(SetFitError::Config(
                "manifest describes a body with no width or no layers".into(),
            ));
        }
        if !self
            .body
            .hidden_size
            .is_multiple_of(self.body.num_attention_heads)
        {
            return Err(SetFitError::Config(format!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                self.body.hidden_size, self.body.num_attention_heads
            )));
        }
        check_sequence_budget(
            &self.body,
            self.classifier.chunk.max_tokens,
            "chunk windows",
        )?;
        Ok(())
    }
}

/// Confirm a manifest actually describes the weights it travels with.
///
/// A manifest and a set of weights are serialised side by side but validated
/// separately, so nothing else stops a three-label manifest shipping with a
/// four-label head. That mismatch would not fail loudly — it would silently
/// misattribute every score to the wrong class name.
fn check_against_module<B: Backend>(manifest: &Manifest, module: &SetFitModule<B>) -> Result<()> {
    let head_labels = module.head.num_labels();
    if head_labels != manifest.num_labels() {
        return Err(SetFitError::Config(format!(
            "manifest names {} labels but the head has {head_labels} outputs",
            manifest.num_labels()
        )));
    }
    let body_width = module.body.hidden_size();
    if body_width != manifest.body.hidden_size {
        return Err(SetFitError::Config(format!(
            "manifest says the body is {} wide but the weights are {body_width}",
            manifest.body.hidden_size
        )));
    }
    Ok(())
}

/// A parsed bundle, before the weights are materialised onto a device.
///
/// Most callers never name this type — [`crate::Classifier::from_bundle`] does
/// the unpacking. Reach for it to inspect a model without loading it onto a
/// device:
///
/// ```no_run
/// use burn_setfit::Bundle;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let bundle = Bundle::unpack(&std::fs::read("support.setfit")?)?;
///
/// println!("{:?}", bundle.manifest.labels());
/// println!("trained on {:?}", bundle.manifest.variant);
/// println!("{} bytes of weights", bundle.weights.len());
/// # Ok(())
/// # }
/// ```
///
/// Anything that is not a bundle, or is a truncated one, is refused rather than
/// misread:
///
/// ```
/// use burn_setfit::Bundle;
///
/// assert!(Bundle::unpack(b"").is_err());
/// assert!(Bundle::unpack(b"BSETFIT\x00 but nothing after it").is_err());
/// ```
#[derive(Debug, Clone)]
pub struct Bundle {
    /// Model configuration and metadata.
    pub manifest: Manifest,
    /// Safetensors payload holding `body.*` and `head.*`.
    pub weights: Vec<u8>,
    /// Raw `tokenizer.json`.
    pub tokenizer: Vec<u8>,
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32> {
    bytes
        .get(at..at + 4)
        .and_then(|s| s.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| SetFitError::Bundle("truncated bundle header".into()))
}

fn read_u64(bytes: &[u8], at: usize) -> Result<usize> {
    bytes
        .get(at..at + 8)
        .and_then(|s| s.try_into().ok())
        .map(|b| u64::from_le_bytes(b) as usize)
        .ok_or_else(|| SetFitError::Bundle("truncated bundle header".into()))
}

fn read_blob(bytes: &[u8], at: usize, len: usize, what: &str) -> Result<Vec<u8>> {
    bytes
        .get(at..at + len)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| SetFitError::Bundle(format!("bundle truncated inside {what}")))
}

impl Bundle {
    /// Serialise a trained model into the container format.
    pub fn pack<B: Backend>(
        module: &SetFitModule<B>,
        manifest: &Manifest,
        tokenizer_json: &[u8],
    ) -> Result<Vec<u8>> {
        manifest.validate()?;
        check_against_module(manifest, module)?;
        if tokenizer_json.is_empty() {
            return Err(SetFitError::Config(
                "a bundle without a tokenizer cannot classify anything".into(),
            ));
        }

        let mut store = SafetensorsStore::from_bytes(None);
        module
            .save_into(&mut store)
            .map_err(|e| SetFitError::Store(e.to_string()))?;
        let weights = store
            .get_bytes()
            .map_err(|e| SetFitError::Store(e.to_string()))?;

        let manifest_json =
            serde_json::to_vec(manifest).map_err(|e| SetFitError::Config(e.to_string()))?;

        let mut out = Vec::with_capacity(
            8 + 4 + 4 + manifest_json.len() + 8 + weights.len() + 8 + tokenizer_json.len(),
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(manifest_json.len() as u32).to_le_bytes());
        out.extend_from_slice(&manifest_json);
        out.extend_from_slice(&(weights.len() as u64).to_le_bytes());
        out.extend_from_slice(&weights);
        out.extend_from_slice(&(tokenizer_json.len() as u64).to_le_bytes());
        out.extend_from_slice(tokenizer_json);

        Ok(out)
    }

    /// Parse a bundle. Does not touch a device.
    pub fn unpack(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err(SetFitError::Bundle("not a .setfit bundle".into()));
        }

        let version = read_u32(bytes, 8)?;
        if version != FORMAT_VERSION {
            return Err(SetFitError::Bundle(format!(
                "bundle format version {version}, but this build reads version {FORMAT_VERSION}"
            )));
        }

        let manifest_len = read_u32(bytes, 12)? as usize;
        let mut at = 16;
        let manifest_bytes = read_blob(bytes, at, manifest_len, "the manifest")?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| SetFitError::Bundle(format!("malformed manifest: {e}")))?;
        manifest.validate()?;
        at += manifest_len;

        let weights_len = read_u64(bytes, at)?;
        at += 8;
        let weights = read_blob(bytes, at, weights_len, "the weights")?;
        at += weights_len;

        let tokenizer_len = read_u64(bytes, at)?;
        at += 8;
        let tokenizer = read_blob(bytes, at, tokenizer_len, "the tokenizer")?;

        Ok(Bundle {
            manifest,
            weights,
            tokenizer,
        })
    }

    /// Materialise the module onto a device.
    pub fn load_module<B: Backend>(&self, device: &B::Device) -> Result<SetFitModule<B>> {
        let mut module =
            SetFitModule::<B>::init(&self.manifest.body, self.manifest.num_labels(), device);
        let mut store = SafetensorsStore::from_bytes(Some(self.weights.clone()));
        module
            .load_from(&mut store)
            .map_err(|e| SetFitError::Store(e.to_string()))?;
        check_against_module(&self.manifest, &module)?;
        Ok(module)
    }

    /// Build the tokenizer.
    pub fn load_tokenizer(&self) -> Result<Tokenizer> {
        Tokenizer::from_bytes(&self.tokenizer)
    }
}
