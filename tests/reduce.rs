//! Numerical properties the reducers are supposed to have.

use burn_setfit::head::TaskMode;
use burn_setfit::reduce::{HierarchicalAccumulator, Reducer};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn assert_close(a: f32, b: f32, tol: f32) {
    assert!((a - b).abs() < tol, "expected {b}, got {a}");
}

/// The documented equivalence: a weighted mean of logits is a weighted mean.
#[test]
fn mean_logits_is_token_weighted() {
    let mut acc = Reducer::MeanLogits.accumulator(2);
    acc.push(300.0, &[1.0, -1.0]);
    acc.push(100.0, &[5.0, 3.0]);

    let out = acc.finish();
    // (300*1 + 100*5) / 400 = 2.0, (300*-1 + 100*3) / 400 = 0.0
    assert_close(out[0], 2.0, 1e-5);
    assert_close(out[1], 0.0, 1e-5);
}

/// Noisy-OR must reproduce `1 - Π(1 - pᵢ)` after decoding through a sigmoid.
#[test]
fn noisy_or_matches_closed_form() {
    let chunk_logits = [-1.0f32, 0.5, 2.0, -3.0];

    let mut acc = Reducer::NoisyOr.accumulator(1);
    for &l in &chunk_logits {
        acc.push(10.0, &[l]);
    }
    let got = sigmoid(acc.finish()[0]);

    let expected = 1.0
        - chunk_logits
            .iter()
            .map(|&l| 1.0 - sigmoid(l))
            .product::<f32>();
    assert_close(got, expected, 1e-5);
}

/// The case mean-pooling gets wrong: one strong chunk buried in noise.
#[test]
fn noisy_or_survives_dilution_where_mean_does_not() {
    let mut mean = Reducer::MeanLogits.accumulator(1);
    let mut noisy = Reducer::NoisyOr.accumulator(1);

    // One clearly positive chunk, then 200 clearly negative ones.
    for acc in [&mut mean, &mut noisy] {
        acc.push(200.0, &[6.0]);
        for _ in 0..200 {
            acc.push(200.0, &[-6.0]);
        }
    }

    assert!(
        sigmoid(mean.finish()[0]) < 0.01,
        "mean should dilute the signal away — that is the failure mode being demonstrated"
    );
    assert!(
        sigmoid(noisy.finish()[0]) > 0.99,
        "noisy-OR should still fire on the one relevant chunk"
    );
}

/// Max over logits is max over probabilities, since sigmoid is monotone.
#[test]
fn max_logits_picks_the_strongest_chunk() {
    let mut acc = Reducer::MaxLogits.accumulator(2);
    acc.push(1.0, &[-2.0, 4.0]);
    acc.push(1.0, &[3.0, 1.0]);
    acc.push(1.0, &[0.0, -5.0]);

    assert_eq!(acc.finish(), vec![3.0, 4.0]);
}

#[test]
fn top_k_mean_averages_only_the_best_k() {
    let mut acc = Reducer::TopKMeanLogits { k: 2 }.accumulator(1);
    for l in [1.0, 9.0, -4.0, 7.0, 0.0] {
        acc.push(1.0, &[l]);
    }
    // Mean of the two largest: (9 + 7) / 2
    assert_close(acc.finish()[0], 8.0, 1e-5);
}

/// Log-sum-exp is supposed to interpolate between mean and max.
#[test]
fn log_sum_exp_interpolates_between_mean_and_max() {
    let logits = [1.0f32, 3.0, 5.0];
    let mean = logits.iter().sum::<f32>() / 3.0;
    let max = 5.0;

    let run = |t: f32| {
        let mut acc = Reducer::LogSumExp { temperature: t }.accumulator(1);
        for &l in &logits {
            acc.push(1.0, &[l]);
        }
        acc.finish()[0]
    };

    assert_close(run(100.0), mean, 0.05);
    assert_close(run(0.01), max, 0.05);
    let mid = run(1.0);
    assert!(mid > mean && mid < max, "expected {mean} < {mid} < {max}");
}

/// Documented invariant: the hierarchy is a no-op for the linear reducer.
#[test]
fn hierarchical_mean_equals_flat_mean() {
    let chunks: Vec<(f32, Vec<f32>)> = (0..37)
        .map(|i| {
            (
                100.0 + i as f32,
                vec![i as f32 * 0.1 - 1.0, 2.0 - i as f32 * 0.05],
            )
        })
        .collect();

    let mut flat = Reducer::MeanLogits.accumulator(2);
    let mut tree = HierarchicalAccumulator::new(4, Reducer::MeanLogits, Reducer::MeanLogits, 2);
    for (w, l) in &chunks {
        flat.push(*w, l);
        tree.push(*w, l);
    }

    let (flat, tree) = (flat.finish(), tree.finish());
    for (a, b) in flat.iter().zip(&tree) {
        assert_close(*a, *b, 1e-3);
    }
}

/// The hierarchy changes the answer for nonlinear reducers — though for noisy-OR
/// the effect is small, because saturation happens within the first block.
#[test]
fn hierarchical_noisy_or_is_less_trigger_happy_than_flat() {
    // Weak evidence spread thinly across many chunks.
    let chunks = vec![vec![-2.0f32]; 64];

    let mut flat = Reducer::NoisyOr.accumulator(1);
    let mut tree = HierarchicalAccumulator::new(8, Reducer::NoisyOr, Reducer::MeanLogits, 1);
    for l in &chunks {
        flat.push(50.0, l);
        tree.push(50.0, l);
    }

    let (flat, tree) = (sigmoid(flat.finish()[0]), sigmoid(tree.finish()[0]));
    assert!(
        flat > tree,
        "flat noisy-OR accumulates weak evidence across all 64 chunks ({flat}); \
         reducing within blocks and then averaging across them should temper that ({tree})"
    );
}

#[test]
fn empty_document_is_neutral_not_confident() {
    for reducer in [
        Reducer::MeanLogits,
        Reducer::MaxLogits,
        Reducer::NoisyOr,
        Reducer::TopKMeanLogits { k: 3 },
        Reducer::LogSumExp { temperature: 1.0 },
    ] {
        let acc = reducer.accumulator(3);
        assert_eq!(acc.count(), 0);
        assert_eq!(acc.finish(), vec![0.0; 3], "{reducer:?}");
    }
}

/// Extreme logits must not produce NaN or infinity anywhere.
#[test]
fn stays_finite_under_saturated_logits() {
    for reducer in [
        Reducer::MeanLogits,
        Reducer::MaxLogits,
        Reducer::NoisyOr,
        Reducer::TopKMeanLogits { k: 2 },
        Reducer::LogSumExp { temperature: 0.5 },
    ] {
        let mut acc = reducer.accumulator(2);
        for l in [-80.0f32, 80.0, -40.0, 40.0, 0.0] {
            acc.push(128.0, &[l, -l]);
        }
        for v in acc.finish() {
            assert!(v.is_finite(), "{reducer:?} produced {v}");
        }
    }
}

#[test]
fn defaults_match_the_task_semantics() {
    assert_eq!(
        Reducer::default_for(TaskMode::SingleLabel),
        Reducer::MeanLogits
    );
    // Max, not noisy-OR: multiplicative combination saturates on long documents.
    assert_eq!(
        Reducer::default_for(TaskMode::multi_label()),
        Reducer::MaxLogits
    );
}

// ── properties every reducer should have ────────────────────────────────────

const ALL: [Reducer; 5] = [
    Reducer::MeanLogits,
    Reducer::MaxLogits,
    Reducer::NoisyOr,
    Reducer::TopKMeanLogits { k: 3 },
    Reducer::LogSumExp { temperature: 1.0 },
];

/// A single chunk leaves nothing to combine, so every reducer must return it
/// unchanged — except noisy-OR, whose combination of one term is still itself in
/// probability space but not in logit space.
#[test]
fn one_chunk_reduces_to_itself() {
    let logits = [1.5f32, -0.5, 3.0];
    for reducer in ALL {
        let mut acc = reducer.accumulator(3);
        acc.push(100.0, &logits);
        let out = acc.finish();

        assert_eq!(acc.count(), 1, "{reducer:?}");
        for (got, want) in out.iter().zip(&logits) {
            assert!(
                (got - want).abs() < 1e-4,
                "{reducer:?} changed a lone chunk: {got} vs {want}"
            );
        }
    }
}

#[test]
fn chunk_counts_are_reported_accurately() {
    for reducer in ALL {
        let mut flat = reducer.accumulator(2);
        let mut tree = HierarchicalAccumulator::new(4, reducer, Reducer::MeanLogits, 2);
        for i in 0..37 {
            flat.push(10.0, &[i as f32, -(i as f32)]);
            tree.push(10.0, &[i as f32, -(i as f32)]);
        }
        assert_eq!(flat.count(), 37, "{reducer:?}");
        assert_eq!(tree.count(), 37, "{reducer:?} (hierarchical)");
    }
}

/// Only the mean consults the token count. If another reducer did, a long
/// irrelevant chunk would outvote a short decisive one.
#[test]
fn only_the_mean_is_weighted_by_chunk_length() {
    let chunks = [[2.0f32], [-2.0], [0.5]];

    let run = |reducer: Reducer, weights: [f32; 3]| {
        let mut acc = reducer.accumulator(1);
        for (w, l) in weights.iter().zip(&chunks) {
            acc.push(*w, l);
        }
        acc.finish()[0]
    };

    for reducer in [
        Reducer::MaxLogits,
        Reducer::NoisyOr,
        Reducer::TopKMeanLogits { k: 2 },
        Reducer::LogSumExp { temperature: 1.0 },
    ] {
        let even = run(reducer, [1.0, 1.0, 1.0]);
        let skewed = run(reducer, [1000.0, 1.0, 1.0]);
        assert!(
            (even - skewed).abs() < 1e-6,
            "{reducer:?} was influenced by chunk length: {even} vs {skewed}"
        );
    }

    // The mean, by contrast, must move.
    let even = run(Reducer::MeanLogits, [1.0, 1.0, 1.0]);
    let skewed = run(Reducer::MeanLogits, [1000.0, 1.0, 1.0]);
    assert!((even - skewed).abs() > 0.5, "the mean should be weighted");
}

/// Chunks arrive in document order, but none of these folds should care.
#[test]
fn reduction_does_not_depend_on_chunk_order() {
    let chunks: Vec<Vec<f32>> = (0..25)
        .map(|i| vec![(i as f32) * 0.3 - 3.0, 2.0 - (i as f32) * 0.15])
        .collect();

    for reducer in ALL {
        let forward = {
            let mut acc = reducer.accumulator(2);
            for c in &chunks {
                acc.push(50.0, c);
            }
            acc.finish()
        };
        let backward = {
            let mut acc = reducer.accumulator(2);
            for c in chunks.iter().rev() {
                acc.push(50.0, c);
            }
            acc.finish()
        };

        for (a, b) in forward.iter().zip(&backward) {
            assert!(
                (a - b).abs() < 1e-3,
                "{reducer:?} is order-dependent: {a} vs {b}"
            );
        }
    }
}

// ── against naive reference implementations ─────────────────────────────────

/// The accumulators fold online, in logit space, with stability tricks. These
/// check each against the plain definition it is supposed to implement.
#[test]
fn online_folds_match_their_definitions() {
    let logits: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0, -0.25, 2.5, -1.5];
    let weights: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];

    let fold = |reducer: Reducer| {
        let mut acc = reducer.accumulator(1);
        for (w, l) in weights.iter().zip(&logits) {
            acc.push(*w, &[*l]);
        }
        acc.finish()[0]
    };

    // Weighted mean.
    let total: f32 = weights.iter().sum();
    let expected: f32 = weights.iter().zip(&logits).map(|(w, l)| w * l).sum::<f32>() / total;
    assert_close(fold(Reducer::MeanLogits), expected, 1e-4);

    // Maximum.
    assert_close(
        fold(Reducer::MaxLogits),
        logits.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        1e-6,
    );

    // Noisy-OR, compared in probability space where the definition lives.
    let expected = 1.0 - logits.iter().map(|l| 1.0 - sigmoid(*l)).product::<f32>();
    assert_close(sigmoid(fold(Reducer::NoisyOr)), expected, 1e-5);

    // Mean of the k largest.
    let mut sorted = logits.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    for k in 1..=logits.len() {
        let expected = sorted[..k].iter().sum::<f32>() / k as f32;
        assert_close(fold(Reducer::TopKMeanLogits { k }), expected, 1e-4);
    }

    // Log-sum-exp, against the direct formula.
    for t in [0.25f32, 1.0, 4.0] {
        let expected =
            t * (logits.iter().map(|l| (l / t).exp()).sum::<f32>() / logits.len() as f32).ln();
        assert_close(fold(Reducer::LogSumExp { temperature: t }), expected, 1e-3);
    }
}

#[test]
fn asking_for_more_than_there_is_averages_what_there_is() {
    // k larger than the chunk count must not pad with zeros and drag the mean down.
    let mut acc = Reducer::TopKMeanLogits { k: 100 }.accumulator(1);
    for l in [4.0f32, 6.0] {
        acc.push(1.0, &[l]);
    }
    assert_close(acc.finish()[0], 5.0, 1e-5);
}

#[test]
fn each_label_is_reduced_independently() {
    // A strong signal on one label must not bleed into another.
    let mut acc = Reducer::MaxLogits.accumulator(3);
    acc.push(1.0, &[9.0, -9.0, 0.0]);
    acc.push(1.0, &[-9.0, 9.0, 0.0]);
    assert_eq!(acc.finish(), vec![9.0, 9.0, 0.0]);
}
