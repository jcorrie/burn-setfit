//! Tokenization and batching — where the padding bug lived.

mod common;

use burn::backend::NdArray;
use burn_setfit::tokenize::{Tokenizer, pad_batch};
use common::{toy_tokenizer, toy_tokenizer_json_string};

type B = NdArray<f32>;

#[test]
fn special_token_ids_come_from_the_vocabulary() {
    let s = toy_tokenizer().special_tokens();
    assert_eq!((s.pad, s.cls, s.sep), (0, 2, 3));
}

#[test]
fn a_malformed_tokenizer_is_rejected_not_ignored() {
    assert!(Tokenizer::from_bytes(b"").is_err());
    assert!(Tokenizer::from_bytes(b"{}").is_err());
    assert!(Tokenizer::from_bytes(b"not json").is_err());
}

#[test]
fn bare_encoding_carries_no_special_tokens() {
    let tk = toy_tokenizer();
    let s = tk.special_tokens();
    let ids = tk.encode_bare("a b c").expect("encodes");

    assert_eq!(ids.len(), 3);
    assert!(!ids.contains(&s.cls) && !ids.contains(&s.sep) && !ids.contains(&s.pad));
}

#[test]
fn adding_specials_wraps_exactly_once() {
    let tk = toy_tokenizer();
    let s = tk.special_tokens();
    let wrapped = tk.add_specials(&[4, 5, 6]);

    assert_eq!(wrapped, vec![s.cls, 4, 5, 6, s.sep]);
}

#[test]
fn full_encoding_respects_the_token_budget_including_specials() {
    let tk = toy_tokenizer();
    let s = tk.special_tokens();

    for budget in [3usize, 8, 16, 50] {
        let ids = tk
            .encode_full(&common::words(200), budget)
            .expect("encodes");
        assert!(
            ids.len() <= budget,
            "budget {budget} exceeded: {} tokens",
            ids.len()
        );
        assert_eq!(*ids.first().unwrap(), s.cls);
        assert_eq!(*ids.last().unwrap(), s.sep);
    }
}

#[test]
fn short_input_is_not_padded_up_to_the_budget() {
    // The regression that made every sentence 94% identical padding.
    let tk = toy_tokenizer();
    let ids = tk.encode_full("a b c", 256).expect("encodes");
    assert_eq!(ids.len(), 5, "expected [CLS] a b c [SEP], got {ids:?}");
    assert!(!ids.contains(&tk.special_tokens().pad));
}

#[test]
fn padding_baked_into_a_tokenizer_file_is_stripped_on_load() {
    // Same toy vocabulary, but the file asks for fixed-length padding.
    let padded_json = toy_tokenizer_json_string().replace(
        "\"padding\": null",
        "\"padding\":{\"strategy\":{\"Fixed\":64},\"direction\":\"Right\",\
         \"pad_to_multiple_of\":null,\"pad_id\":0,\"pad_type_id\":0,\"pad_token\":\"[PAD]\"}",
    );
    assert_ne!(
        padded_json,
        toy_tokenizer_json_string(),
        "the fixture must differ"
    );

    let tk = Tokenizer::from_bytes(padded_json.as_bytes()).expect("loads");
    let ids = tk.encode_bare("a b c").expect("encodes");
    assert_eq!(ids.len(), 3, "padding survived the load: {ids:?}");
}

#[test]
fn empty_text_encodes_to_nothing_but_the_specials() {
    let tk = toy_tokenizer();
    assert!(tk.encode_bare("").expect("encodes").is_empty());
    assert_eq!(tk.encode_full("", 32).expect("encodes").len(), 2);
}

// ── batching ────────────────────────────────────────────────────────────────

#[test]
fn a_ragged_batch_pads_to_the_longest_and_masks_the_rest() {
    let sequences = vec![vec![4u32, 5, 6], vec![7], vec![8, 9]];
    let (ids, mask) = pad_batch::<B>(&sequences, 99, &Default::default());

    assert_eq!(ids.dims(), [3, 3]);
    assert_eq!(mask.dims(), [3, 3]);

    let ids = ids.into_data().into_vec::<i64>().unwrap();
    let mask = mask.into_data().into_vec::<f32>().unwrap();

    assert_eq!(ids, vec![4, 5, 6, 7, 99, 99, 8, 9, 99]);
    // Exactly one mask entry per real token, and nothing else.
    assert_eq!(mask, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn a_uniform_batch_needs_no_padding_at_all() {
    let sequences = vec![vec![4u32, 5], vec![6, 7]];
    let (_, mask) = pad_batch::<B>(&sequences, 0, &Default::default());
    let mask = mask.into_data().into_vec::<f32>().unwrap();
    assert!(mask.iter().all(|m| *m == 1.0), "nothing to mask: {mask:?}");
}

#[test]
fn the_mask_counts_exactly_the_real_tokens() {
    // Mean pooling divides by this sum, so an off-by-one here rescales every
    // embedding in the batch.
    let sequences = vec![vec![4u32; 7], vec![5u32; 2], vec![6u32; 13]];
    let (_, mask) = pad_batch::<B>(&sequences, 0, &Default::default());
    let mask = mask.into_data().into_vec::<f32>().unwrap();

    let width = 13;
    for (row, expected) in [7usize, 2, 13].iter().enumerate() {
        let sum: f32 = mask[row * width..(row + 1) * width].iter().sum();
        assert_eq!(sum as usize, *expected);
    }
}

#[test]
fn a_batch_of_one_is_shaped_like_a_batch() {
    let (ids, mask) = pad_batch::<B>(&[vec![4u32, 5, 6]], 0, &Default::default());
    assert_eq!(ids.dims(), [1, 3]);
    assert_eq!(mask.dims(), [1, 3]);
}
