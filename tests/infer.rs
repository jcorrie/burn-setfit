//! Classification: decoding, evidence, batching, and the overrides.

mod common;
use burn_setfit::config::ClassifierConfig;
use burn_setfit::head::TaskMode;
use burn_setfit::infer::{Classifier, decode, to_probabilities};
use burn_setfit::reduce::Reducer;
use common::{cpu, small_chunking, toy_bundle, words};

fn classifier(config: ClassifierConfig) -> Classifier {
    Classifier::from_bundle(&toy_bundle(config), cpu()).expect("loads")
}

fn single_label() -> Classifier {
    classifier(ClassifierConfig::new(["alpha", "beta"]).with_chunking(small_chunking()))
}

// ── probabilities ───────────────────────────────────────────────────────────

#[test]
fn single_label_probabilities_form_a_distribution() {
    let p = to_probabilities(&[2.0, 1.0, -3.0], TaskMode::SingleLabel);
    let sum: f32 = p.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax must sum to one, got {sum}"
    );
    assert!(p.iter().all(|v| *v > 0.0 && *v < 1.0));
    // Order is preserved.
    assert!(p[0] > p[1] && p[1] > p[2]);
}

#[test]
fn multi_label_probabilities_are_independent() {
    let p = to_probabilities(&[3.0, 3.0, 3.0], TaskMode::multi_label());
    let sum: f32 = p.iter().sum();
    assert!(
        sum > 2.5,
        "independent sigmoids need not sum to one, got {sum}"
    );
    for v in &p {
        assert!(*v > 0.9, "each class scores on its own merits: {v}");
    }
}

#[test]
fn probabilities_stay_finite_at_saturation() {
    for task in [TaskMode::SingleLabel, TaskMode::multi_label()] {
        for logits in [
            vec![100.0f32, -100.0],
            vec![-1e30, 1e30],
            vec![0.0, 0.0],
            vec![f32::MIN, f32::MAX],
        ] {
            let p = to_probabilities(&logits, task);
            assert!(
                p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                "{task:?} on {logits:?} produced {p:?}"
            );
        }
    }
}

// ── decoding ────────────────────────────────────────────────────────────────

#[test]
fn single_label_always_picks_exactly_one() {
    assert_eq!(decode(&[0.1, 0.7, 0.2], TaskMode::SingleLabel), vec![1]);
    // Even when nothing is confident: an argmax has no way to abstain.
    assert_eq!(decode(&[0.34, 0.33, 0.33], TaskMode::SingleLabel), vec![0]);
}

#[test]
fn multi_label_can_pick_several_or_none() {
    let task = TaskMode::MultiLabel { threshold: 0.5 };
    assert_eq!(decode(&[0.9, 0.8, 0.1], task), vec![0, 1]);
    assert_eq!(decode(&[0.4, 0.3, 0.1], task), Vec::<usize>::new());
    assert_eq!(decode(&[0.9, 0.9, 0.9], task), vec![0, 1, 2]);
}

#[test]
fn the_threshold_is_inclusive_at_its_boundary() {
    let task = TaskMode::MultiLabel { threshold: 0.5 };
    assert_eq!(
        decode(&[0.5], task),
        vec![0],
        "at the threshold counts as present"
    );
    assert_eq!(decode(&[0.499_999], task), Vec::<usize>::new());
}

#[test]
fn a_higher_threshold_predicts_a_subset() {
    let scores = [0.95, 0.6, 0.45, 0.2];
    let lenient = decode(&scores, TaskMode::MultiLabel { threshold: 0.3 });
    let strict = decode(&scores, TaskMode::MultiLabel { threshold: 0.9 });
    assert!(strict.iter().all(|i| lenient.contains(i)));
    assert!(strict.len() < lenient.len());
}

// ── batching ────────────────────────────────────────────────────────────────

/// Batch size is a throughput knob. If it changed the answer it would be a bug
/// that only appears on inputs of certain lengths.
#[test]
fn batch_size_does_not_change_the_verdict() {
    let text = words(400);
    let reference = single_label()
        .with_batch_size(1)
        .classify(&text)
        .expect("classifies");

    for batch in [2, 3, 8, 64] {
        let got = single_label()
            .with_batch_size(batch)
            .classify(&text)
            .expect("classifies");
        assert_eq!(got.predicted, reference.predicted, "batch {batch}");
        assert_eq!(got.chunks_seen, reference.chunks_seen, "batch {batch}");
        for (a, b) in got.scores.iter().zip(&reference.scores) {
            assert!((a - b).abs() < 1e-4, "batch {batch}: {a} vs {b}");
        }
    }
}

#[test]
fn a_zero_batch_size_is_clamped_rather_than_dividing_by_zero() {
    let got = single_label()
        .with_batch_size(0)
        .classify("a b c")
        .expect("classifies");
    assert_eq!(got.chunks_seen, 1);
}

// ── streaming ───────────────────────────────────────────────────────────────

#[test]
fn a_streamed_document_classifies_the_same_as_a_whole_one() {
    let text = words(300);
    let whole = single_label().classify(&text).expect("classifies");

    let pieces: Vec<String> = text
        .as_bytes()
        .chunks(11)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect();
    let streamed = single_label()
        .classify_stream(pieces.into_iter())
        .expect("classifies");

    assert_eq!(whole.predicted, streamed.predicted);
    assert_eq!(whole.chunks_seen, streamed.chunks_seen);
    for (a, b) in whole.scores.iter().zip(&streamed.scores) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}

#[test]
fn an_empty_document_yields_no_chunks_and_a_neutral_verdict() {
    let p = single_label().classify("").expect("classifies");
    assert_eq!(p.chunks_seen, 0);
    assert!(p.evidence.is_empty());
    // Zero logits under softmax: a uniform distribution, not a confident guess.
    let sum: f32 = p.scores.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(p.scores.iter().all(|s| (s - 0.5).abs() < 1e-5));
}

#[test]
fn a_long_document_produces_proportionally_more_chunks() {
    let short = single_label().classify(&words(50)).expect("classifies");
    let long = single_label().classify(&words(500)).expect("classifies");
    assert!(
        long.chunks_seen > short.chunks_seen * 5,
        "{} vs {}",
        long.chunks_seen,
        short.chunks_seen
    );
}

// ── evidence ────────────────────────────────────────────────────────────────

#[test]
fn evidence_is_ranked_and_bounded() {
    let limit = 3;
    let p = single_label()
        .with_evidence_limit(limit)
        .classify(&words(600))
        .expect("classifies");

    assert!(p.evidence.len() <= limit);
    assert!(
        !p.evidence.is_empty(),
        "a long document should produce evidence"
    );
    for w in p.evidence.windows(2) {
        assert!(w[0].score >= w[1].score, "evidence must be ranked");
    }
}

#[test]
fn evidence_only_names_labels_that_were_predicted() {
    // A chunk confident about some other label is not evidence for this verdict.
    let p = single_label().classify(&words(600)).expect("classifies");
    for e in &p.evidence {
        assert!(
            p.predicted.contains(&e.label),
            "evidence cites label {} which was not predicted ({:?})",
            e.label,
            p.predicted
        );
    }
}

#[test]
fn evidence_byte_ranges_index_the_document_that_was_classified() {
    let text = words(600);
    let p = single_label().classify(&text).expect("classifies");
    for e in &p.evidence {
        assert!(e.byte_range.end <= text.len());
        // Must be a valid slice, or tracing a label back to its source panics.
        let excerpt = &text[e.byte_range.clone()];
        assert!(!excerpt.trim().is_empty());
    }
}

#[test]
fn a_zero_limit_disables_the_bookkeeping_entirely() {
    let p = single_label()
        .with_evidence_limit(0)
        .classify(&words(600))
        .expect("classifies");
    assert!(p.evidence.is_empty());
    // The verdict itself is unaffected.
    assert_eq!(p.predicted.len(), 1);
}

// ── decode-time overrides ───────────────────────────────────────────────────

#[test]
fn the_reducer_can_be_changed_without_repacking() {
    let bundle =
        toy_bundle(ClassifierConfig::new(["alpha", "beta"]).with_chunking(small_chunking()));
    let text = words(400);

    let load = |reducer| {
        Classifier::from_bundle(&bundle, cpu())
            .unwrap()
            .with_reducer(reducer)
            .classify(&text)
            .unwrap()
    };

    let mean = load(Reducer::MeanLogits);
    let max = load(Reducer::MaxLogits);

    // Same document, same weights, same chunks — only the fold differs.
    assert_eq!(mean.chunks_seen, max.chunks_seen);
    for p in [&mean, &max] {
        let sum: f32 = p.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "still a distribution: {sum}");
    }
}

/// Reducer ordering is only observable per label, and only under a monotone link.
///
/// In logit space `max >= mean` holds for every label separately. Single-label
/// scores are a softmax, whose top probability depends on the *gap* between
/// logits, and raising both logits can narrow that gap — so the ordering does not
/// survive into single-label probabilities. Multi-label scores are independent
/// sigmoids, which are monotone, so there it does.
#[test]
fn stronger_reducers_dominate_per_label_under_multi_label() {
    let bundle = toy_bundle(
        ClassifierConfig::new(["alpha", "beta"])
            .multi_label()
            .with_chunking(small_chunking()),
    );
    let text = words(400);

    let load = |reducer| {
        Classifier::from_bundle(&bundle, cpu())
            .unwrap()
            .with_reducer(reducer)
            .classify(&text)
            .unwrap()
            .scores
    };

    let mean = load(Reducer::MeanLogits);
    let max = load(Reducer::MaxLogits);
    let noisy_or = load(Reducer::NoisyOr);

    for i in 0..2 {
        assert!(
            max[i] >= mean[i] - 1e-6,
            "label {i}: max {} should not fall below mean {}",
            max[i],
            mean[i]
        );
        // Noisy-OR accumulates every chunk's evidence, so it dominates the single
        // strongest chunk. This is exactly the saturation that makes it a poor
        // default on long documents.
        assert!(
            noisy_or[i] >= max[i] - 1e-6,
            "label {i}: noisy-OR {} should not fall below max {}",
            noisy_or[i],
            max[i]
        );
    }
}

#[test]
fn the_threshold_can_be_changed_without_repacking() {
    let multi = classifier(
        ClassifierConfig::new(["alpha", "beta"])
            .multi_label()
            .with_chunking(small_chunking()),
    );
    let text = words(100);

    let permissive = multi.clone_with_threshold(0.01).classify(&text).unwrap();
    let strict = multi.clone_with_threshold(0.99).classify(&text).unwrap();

    assert!(
        strict.predicted.len() <= permissive.predicted.len(),
        "a stricter threshold cannot predict more: {:?} vs {:?}",
        strict.predicted,
        permissive.predicted
    );
}

#[test]
fn a_threshold_override_is_inert_on_a_single_label_model() {
    let text = words(100);
    let before = single_label().classify(&text).unwrap();
    let after = single_label().with_threshold(0.99).classify(&text).unwrap();
    assert_eq!(before.predicted, after.predicted);
}

#[test]
fn hierarchical_reduction_agrees_with_flat_for_the_linear_reducer() {
    // The invariant proven at unit level, checked here through the full pipeline.
    let bundle =
        toy_bundle(ClassifierConfig::new(["alpha", "beta"]).with_chunking(small_chunking()));
    let text = words(500);

    let flat = Classifier::from_bundle(&bundle, cpu())
        .unwrap()
        .with_reducer(Reducer::MeanLogits)
        .classify(&text)
        .unwrap();
    let tree = Classifier::from_bundle(&bundle, cpu())
        .unwrap()
        .with_reducer(Reducer::MeanLogits)
        .with_hierarchy(4)
        .classify(&text)
        .unwrap();

    for (a, b) in flat.scores.iter().zip(&tree.scores) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}

// ── embedding ───────────────────────────────────────────────────────────────

#[test]
fn embeddings_are_unit_vectors_of_the_body_width() {
    let embeddings = single_label().embed(&["a b c", "x y z"]).expect("embeds");
    assert_eq!(embeddings.len(), 2);

    for e in &embeddings {
        assert_eq!(e.len(), common::toy_body_config().hidden_size);
        let norm: f32 = e.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected a unit vector, got {norm}"
        );
    }
}

#[test]
fn embedding_is_independent_of_the_rest_of_the_batch() {
    let c = single_label();
    let together = c.embed(&["a b c", "x y z z z z"]).expect("embeds");
    let alone = c.embed(&["a b c"]).expect("embeds");

    // Padding must not leak: a short input's embedding cannot depend on how long
    // its neighbours happened to be.
    for (a, b) in together[0].iter().zip(&alone[0]) {
        assert!(
            (a - b).abs() < 1e-4,
            "padding leaked into the embedding: {a} vs {b}"
        );
    }
}

#[test]
fn embedding_no_texts_yields_no_embeddings() {
    assert!(
        single_label()
            .embed(&[])
            .expect("embeds nothing")
            .is_empty()
    );
}

// ── the smaller surface ─────────────────────────────────────────────────────

#[test]
fn a_prediction_names_its_labels_and_its_top_score() {
    let c = single_label();
    let p = c.classify(&words(10)).unwrap();
    let names = p.labels(c.manifest());
    assert_eq!(names.len(), 1, "single-label picks one");
    let (top, score) = p.top().expect("there are scores");
    assert_eq!(names[0], c.labels()[top]);
    assert_eq!(score, p.scores[top]);
    assert!(p.scores.iter().all(|&s| s <= score));
}

#[test]
fn a_classifier_assembled_from_parts_matches_one_loaded_from_bytes() {
    use burn_setfit::bundle::Bundle;

    let config = ClassifierConfig::new(["alpha", "beta"]).with_chunking(small_chunking());
    let bytes = toy_bundle(config);
    let bundle = Bundle::unpack(&bytes).unwrap();
    let from_parts = Classifier::new(
        bundle.load_module(&cpu()).unwrap(),
        bundle.load_tokenizer().unwrap(),
        bundle.manifest.clone(),
        cpu(),
    )
    .unwrap();
    let from_bytes = Classifier::from_bundle(&bytes, cpu()).unwrap();

    let text = words(40);
    assert_eq!(
        from_parts.classify(&text).unwrap().scores,
        from_bytes.classify(&text).unwrap().scores
    );
}

#[test]
fn classifier_new_validates_the_manifest_it_is_given() {
    use burn_setfit::bundle::Bundle;

    let bundle = Bundle::unpack(&toy_bundle(
        ClassifierConfig::new(["alpha", "beta"]).with_chunking(small_chunking()),
    ))
    .unwrap();
    let mut broken = bundle.manifest.clone();
    broken.body.num_hidden_layers = 0;
    let result = Classifier::new(
        bundle.load_module(&cpu()).unwrap(),
        bundle.load_tokenizer().unwrap(),
        broken,
        cpu(),
    );
    assert!(result.is_err());
}

#[test]
fn per_chunk_output_is_the_same_whatever_the_batch_size() {
    // More chunks than a batch holds, so the full-batch flush runs mid-document.
    let c = single_label();
    let text = words(400);
    let whole = c.classify_chunks(&text).unwrap();
    assert!(whole.len() > 3, "the document must span several batches");
    let batched = single_label()
        .with_batch_size(2)
        .classify_chunks(&text)
        .unwrap();
    assert_eq!(whole.len(), batched.len());
    for (a, b) in whole.iter().zip(&batched) {
        assert_eq!(a.byte_range, b.byte_range);
        assert_eq!(a.predicted, b.predicted);
        for (x, y) in a.scores.iter().zip(&b.scores) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }
}
