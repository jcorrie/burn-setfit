//! Fidelity checks against the real checkpoint.
//!
//! These download `all-MiniLM-L6-v2`, so they are `#[ignore]`d by default:
//!
//! ```text
//! cargo test --release --features ndarray,train,native -- --ignored
//! ```

use burn::backend::NdArray;
use burn_setfit::checkpoint::Checkpoint;
use burn_setfit::minilm::MiniLmVariant;
use burn_setfit::model::embed_body;
use burn_setfit::tokenize::pad_batch;

type B = NdArray<f32>;

fn embeddings(sentences: &[&str]) -> Vec<Vec<f32>> {
    let device = Default::default();
    let checkpoint = Checkpoint::download(MiniLmVariant::L6, None).expect("download");
    let tokenizer = checkpoint.tokenizer().expect("tokenizer");
    let body = checkpoint.body::<B>(&device).expect("weights");

    sentences
        .iter()
        .map(|s| {
            let ids = tokenizer.encode_full(s, 256).expect("encode");
            let (input_ids, mask) = pad_batch::<B>(&[ids], 0, &device);
            embed_body(&body, input_ids, mask)
                .into_data()
                .into_vec::<f32>()
                .unwrap()
        })
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The published similarity matrix for these three sentences under this exact
/// model. Reproducing it to three decimals means the whole path — tokenizer,
/// weight remapping, encoder, pooling, normalisation — matches upstream.
#[test]
#[ignore = "downloads all-MiniLM-L6-v2"]
fn reproduces_the_reference_similarity_matrix() {
    let e = embeddings(&[
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]);

    let expected = [
        [1.0000f32, 0.6660, 0.1046],
        [0.6660, 1.0000, 0.1411],
        [0.1046, 0.1411, 1.0000],
    ];

    for i in 0..3 {
        for j in 0..3 {
            let got = cos(&e[i], &e[j]);
            assert!(
                (got - expected[i][j]).abs() < 1e-3,
                "similarity[{i}][{j}] = {got:.4}, expected {:.4}",
                expected[i][j]
            );
        }
    }
}

/// The specific regression that produced that matrix wrongly the first time.
///
/// `all-MiniLM-L6-v2`'s `tokenizer.json` pads to 128 tokens. Left in place, the
/// padding is indistinguishable from content by the time the attention mask is
/// derived from sequence length, so the encoder attends to it and mean pooling
/// averages it in — every sentence ends up ~94% identical padding.
#[test]
#[ignore = "downloads all-MiniLM-L6-v2"]
fn tokenizer_padding_is_stripped_on_load() {
    let checkpoint = Checkpoint::download(MiniLmVariant::L6, None).expect("download");
    let tokenizer = checkpoint.tokenizer().expect("tokenizer");

    let ids = tokenizer
        .encode_full("The weather is lovely today.", 256)
        .expect("encode");
    assert_eq!(
        ids.len(),
        8,
        "expected a tight sequence, got {} tokens: {ids:?}",
        ids.len()
    );

    let special = tokenizer.special_tokens();
    assert_eq!(*ids.first().unwrap(), special.cls);
    assert_eq!(*ids.last().unwrap(), special.sep);
    assert!(
        !ids[1..ids.len() - 1].contains(&special.pad),
        "padding leaked into the sequence: {ids:?}"
    );
}

#[test]
#[ignore = "downloads all-MiniLM-L6-v2"]
fn paraphrases_outrank_unrelated_text() {
    let e = embeddings(&[
        "The cat sat on the mat.",
        "A feline rested upon the rug.",
        "Quantum chromodynamics describes the strong interaction.",
    ]);

    let paraphrase = cos(&e[0], &e[1]);
    let unrelated = cos(&e[0], &e[2]);
    assert!(
        paraphrase > 0.4 && unrelated < 0.3,
        "paraphrase {paraphrase:.3} should be high and unrelated {unrelated:.3} low"
    );
}
