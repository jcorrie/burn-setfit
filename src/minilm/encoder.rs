//! The BERT encoder stack, owned here so its GELU differentiates without `tanh`.
//!
//! This is Burn's post-norm `TransformerEncoder` with one thing changed, and it
//! keeps that module's field names so the same parameter paths
//! (`encoder.layers.N.mha.query`, `…pwff.linear_inner`, `…norm_1`) load the
//! same weights. The one change is [`gelu`].
//!
//! Burn computes GELU forward from `erf`, but its backward is the derivative of
//! the *tanh approximation*: it calls `tanh` on `0.0357x³ + 0.798x`, which
//! passes 43 for pre-activations above about 9. Metal returns NaN from `tanh`
//! for inputs that large. cubecl guards it behind `cfg(target_os = "macos")`,
//! which is the Rust target — a wasm build running in a browser on a Mac also
//! lands on Metal, through WebGPU, and gets no guard. One GPU training step
//! there filled MiniLM's weights with NaN, and every prediction after it
//! landed on one label. Still unguarded in cubecl 0.11 / Burn 0.22.
//!
//! Composing GELU from `erf` instead lets autodiff differentiate `erf`, whose
//! backward is `2/√π · exp(-x²)`: it underflows to zero for large inputs rather
//! than producing NaN, on every target, and it is the exact derivative rather
//! than an approximation's. The forward pass is the same arithmetic Burn's
//! `gelu` does, in the same order.

use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{
    Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::{Bool, Device, Tensor};

/// Encoder stack configuration.
#[derive(Config, Debug)]
pub struct MiniLmEncoderConfig {
    /// Hidden size.
    pub d_model: usize,
    /// Feed-forward intermediate size.
    pub d_ff: usize,
    /// Attention heads.
    pub n_heads: usize,
    /// Encoder layers.
    pub n_layers: usize,
    /// Dropout probability, for attention output, feed-forward and residuals.
    pub dropout: f64,
    /// Layer-norm epsilon inside the encoder.
    ///
    /// `1e-5`, which is what Burn's `TransformerEncoder` used here before this
    /// module replaced it and what the published similarity matrix was
    /// reproduced with — not the checkpoint's `layer_norm_eps` of `1e-12`.
    /// Moving to that is a numerical change of its own, to be checked against
    /// the published matrix rather than slipped in alongside this one.
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
    /// Weight initialiser for a body built from scratch.
    pub initializer: Initializer,
}

/// A stack of post-norm BERT layers.
#[derive(Module, Debug)]
pub struct MiniLmEncoder {
    /// The layers, applied in order.
    pub layers: Vec<MiniLmLayer>,
}

/// One post-norm BERT layer: attention, add & norm, feed-forward, add & norm.
#[derive(Module, Debug)]
pub struct MiniLmLayer {
    /// Self-attention.
    pub mha: MultiHeadAttention,
    /// Position-wise feed-forward network.
    pub pwff: FeedForward,
    /// Norm after the attention residual.
    pub norm_1: LayerNorm,
    /// Norm after the feed-forward residual.
    pub norm_2: LayerNorm,
    /// Dropout on both residual branches.
    pub dropout: Dropout,
}

/// `Linear → GELU → Dropout → Linear`.
#[derive(Module, Debug)]
pub struct FeedForward {
    /// `d_model → d_ff`.
    pub linear_inner: Linear,
    /// `d_ff → d_model`.
    pub linear_outer: Linear,
    /// Dropout after the activation.
    pub dropout: Dropout,
}

impl MiniLmEncoderConfig {
    /// Initialise with random weights.
    pub fn init(&self, device: &Device) -> MiniLmEncoder {
        MiniLmEncoder {
            layers: (0..self.n_layers)
                .map(|_| self.init_layer(device))
                .collect(),
        }
    }

    fn init_layer(&self, device: &Device) -> MiniLmLayer {
        let norm = || {
            LayerNormConfig::new(self.d_model)
                .with_epsilon(self.layer_norm_eps)
                .init(device)
        };
        let linear = |d_in, d_out| {
            LinearConfig::new(d_in, d_out)
                .with_initializer(self.initializer.clone())
                .init(device)
        };
        MiniLmLayer {
            mha: MultiHeadAttentionConfig::new(self.d_model, self.n_heads)
                .with_initializer(self.initializer.clone())
                .with_dropout(self.dropout)
                .init(device),
            pwff: FeedForward {
                linear_inner: linear(self.d_model, self.d_ff),
                linear_outer: linear(self.d_ff, self.d_model),
                dropout: DropoutConfig::new(self.dropout).init(),
            },
            norm_1: norm(),
            norm_2: norm(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl MiniLmEncoder {
    /// Encode `[batch, seq, d_model]`, ignoring positions where `mask_pad` is true.
    pub fn forward(&self, input: Tensor<3>, mask_pad: Tensor<2, Bool>) -> Tensor<3> {
        self.layers
            .iter()
            .fold(input, |x, layer| layer.forward(x, mask_pad.clone()))
    }
}

impl MiniLmLayer {
    fn forward(&self, x: Tensor<3>, mask_pad: Tensor<2, Bool>) -> Tensor<3> {
        let attended = self
            .mha
            .forward(MhaInput::self_attn(x.clone()).mask_pad(mask_pad))
            .context;
        let x = self.norm_1.forward(x + self.dropout.forward(attended));

        let fed = self.pwff.forward(x.clone());
        self.norm_2.forward(x + self.dropout.forward(fed))
    }
}

impl FeedForward {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = gelu(self.linear_inner.forward(x));
        self.linear_outer.forward(self.dropout.forward(x))
    }
}

/// Exact GELU, `x · Φ(x)`, built from `erf` so its gradient never calls `tanh`.
///
/// See the module docs for why this is not `burn::tensor::activation::gelu`.
pub fn gelu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let cdf = (x.clone() / core::f32::consts::SQRT_2).erf() + 1.0;
    (x * cdf) / 2.0
}

#[cfg(all(test, feature = "ndarray", feature = "train"))]
mod tests {
    use super::*;

    #[allow(deprecated)] // `ndarray` is deprecated upstream in favour of `flex`
    fn cpu() -> Device {
        Device::ndarray()
    }

    fn floats<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data().try_into_vec::<f32>().expect("floats")
    }

    /// `d/dx [x · Φ(x)] = Φ(x) + x · φ(x)`.
    fn exact_derivative(x: f64) -> f64 {
        let cdf = 0.5 * (1.0 + libm_erf(x / core::f64::consts::SQRT_2));
        let pdf = (-x * x / 2.0).exp() / (2.0 * core::f64::consts::PI).sqrt();
        cdf + x * pdf
    }

    /// Abramowitz & Stegun 7.1.26 is too coarse to test against; this is the
    /// series and continued fraction, good to well past f32.
    fn libm_erf(x: f64) -> f64 {
        if x < 0.0 {
            return -libm_erf(-x);
        }
        if x < 3.0 {
            // Maclaurin series.
            let mut term = x;
            let mut sum = x;
            for n in 1..200 {
                term *= -x * x / n as f64;
                sum += term / (2 * n + 1) as f64;
            }
            sum * 2.0 / core::f64::consts::PI.sqrt()
        } else {
            // erfc continued fraction, evaluated backwards.
            let mut f = 0.0;
            for k in (1..60).rev() {
                f = k as f64 / 2.0 / (x + f);
            }
            1.0 - (-x * x).exp() / core::f64::consts::PI.sqrt() / (x + f)
        }
    }

    #[test]
    fn gelu_forward_matches_burns() {
        let device = cpu();
        let x = Tensor::<1>::from_floats([-6.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0, 50.0], &device);
        let ours = floats(gelu(x.clone()));
        let theirs = floats(burn::tensor::activation::gelu(x));
        assert_eq!(ours, theirs, "same arithmetic in the same order");
    }

    /// The guard on this module's reason to exist.
    ///
    /// Burn's GELU backward is the tanh approximation's derivative, which is up
    /// to 8.7e-4 away from the exact one near `x = 2`. Matching the exact
    /// derivative to 1e-5 therefore fails if anyone swaps Burn's `Gelu` back in
    /// — which is the change that brings `tanh` back into training, and with
    /// it NaN on Metal through WebGPU. Large inputs are where that NaN came
    /// from; here they must give a finite gradient of one.
    #[test]
    fn gelu_gradient_is_exact_and_finite_where_tanh_was_not() {
        let device = cpu().autodiff();
        let points = [
            -6.0f32, -3.0, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 2.02, 3.0, 9.5, 50.0,
        ];
        let x = Tensor::<1>::from_floats(points, &device).require_grad();
        let grads = gelu(x.clone()).sum().backward();
        let got = floats(x.grad(&grads).expect("x requires grad"));

        for (&p, &g) in points.iter().zip(&got) {
            assert!(g.is_finite(), "gradient at {p} is {g}");
            let want = exact_derivative(p as f64) as f32;
            assert!(
                (g - want).abs() < 1e-5,
                "gradient at {p}: {g}, exact {want} — the tanh approximation is back?"
            );
        }
    }

    /// The paths are what lets this replace Burn's encoder without touching a
    /// single checkpoint or bundle; the forward pass is what makes it a
    /// replacement rather than a different model.
    #[test]
    fn loads_burns_encoder_weights_and_agrees_with_it() {
        use burn::nn::transformer::{TransformerEncoderConfig, TransformerEncoderInput};
        use burn_store::{ModuleSnapshot, SafetensorsStore};

        let device = cpu();
        let (d_model, d_ff, heads, layers) = (16, 32, 2, 2);
        let initializer = Initializer::KaimingUniform {
            gain: 1.0 / 3.0f64.sqrt(),
            fan_out_only: false,
        };

        let reference = TransformerEncoderConfig::new(d_model, d_ff, heads, layers)
            .with_dropout(0.0)
            .with_norm_first(false)
            .with_initializer(initializer.clone())
            .init(&device);
        let mut store = SafetensorsStore::from_bytes(None);
        reference.save_into(&mut store).expect("saves");
        let bytes = store.get_bytes().expect("serialises");

        let mut ours =
            MiniLmEncoderConfig::new(d_model, d_ff, heads, layers, 0.0, initializer).init(&device);
        let mut store = SafetensorsStore::from_bytes(Some(bytes)).allow_partial(false);
        ours.load_from(&mut store)
            .expect("every one of Burn's paths has a home");

        let x = Tensor::<3>::random([2, 5, d_model], Default::default(), &device);
        let mask =
            Tensor::<2, Bool>::from_data([[false, false, false, true, true], [false; 5]], &device);
        let want = floats(
            reference.forward(TransformerEncoderInput::new(x.clone()).mask_pad(mask.clone())),
        );
        let got = floats(ours.forward(x, mask));
        for (i, (w, g)) in want.iter().zip(&got).enumerate() {
            assert!((w - g).abs() < 1e-5, "element {i}: Burn {w}, ours {g}");
        }
    }
}
