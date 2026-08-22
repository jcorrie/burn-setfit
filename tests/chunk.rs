//! Chunker behaviour: windowing, overlap, boundaries, and streaming.

mod common;

use burn_setfit::chunk::{ChunkConfig, Chunker, chunk_str};
use common::{toy_tokenizer, words};

fn collect(text: &str, config: ChunkConfig) -> Vec<burn_setfit::chunk::Chunk> {
    let tk = toy_tokenizer();
    chunk_str(text, &tk, config)
        .collect::<Result<Vec<_>, _>>()
        .expect("chunking should succeed")
}

#[test]
fn short_text_is_a_single_chunk() {
    let chunks = collect(&words(5), ChunkConfig::default());
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].token_count, 5);
    // [CLS] + 5 + [SEP]
    assert_eq!(chunks[0].ids.len(), 7);
}

#[test]
fn every_chunk_respects_the_window() {
    let config = ChunkConfig {
        max_tokens: 16,
        overlap_tokens: 4,
        min_final_tokens: 1,
    };
    // No sentence boundaries at all: forces the hard-split path.
    let chunks = collect(&words(200), config);

    assert!(
        chunks.len() > 1,
        "200 tokens must not fit in one 16-token window"
    );
    for c in &chunks {
        assert!(
            c.ids.len() <= config.max_tokens,
            "chunk of {} exceeds max_tokens {}",
            c.ids.len(),
            config.max_tokens
        );
        assert!(c.token_count <= config.capacity());
    }
}

#[test]
fn specials_wrap_each_chunk_exactly_once() {
    let tk = toy_tokenizer();
    let special = tk.special_tokens();
    let config = ChunkConfig {
        max_tokens: 12,
        overlap_tokens: 2,
        min_final_tokens: 1,
    };

    for c in collect(&words(100), config) {
        assert_eq!(*c.ids.first().unwrap(), special.cls);
        assert_eq!(*c.ids.last().unwrap(), special.sep);
        assert_eq!(c.ids.iter().filter(|&&i| i == special.cls).count(), 1);
        assert_eq!(c.ids.iter().filter(|&&i| i == special.sep).count(), 1);
        assert_eq!(c.ids.len(), c.token_count + 2);
    }
}

#[test]
fn consecutive_chunks_overlap() {
    let config = ChunkConfig {
        max_tokens: 20,
        overlap_tokens: 6,
        min_final_tokens: 1,
    };
    // Sentence boundaries give the packer whole segments to carry over.
    let text = (0..40)
        .map(|i| format!("{} {} {} .", words(3), words(2), i % 10))
        .collect::<Vec<_>>()
        .join(" ");

    let chunks = collect(&text, config);
    assert!(chunks.len() >= 3);

    let overlapping = chunks
        .windows(2)
        .filter(|w| w[1].byte_range.start < w[0].byte_range.end)
        .count();
    assert!(
        overlapping > 0,
        "expected overlapping byte ranges between consecutive chunks"
    );
}

#[test]
fn chunks_cover_the_document_in_order() {
    let config = ChunkConfig {
        max_tokens: 24,
        overlap_tokens: 4,
        min_final_tokens: 1,
    };
    let text = words(300);
    let chunks = collect(&text, config);

    assert_eq!(chunks[0].byte_range.start, 0);
    for w in chunks.windows(2) {
        assert!(
            w[1].byte_range.start >= w[0].byte_range.start,
            "byte ranges must advance monotonically"
        );
        assert!(
            w[1].byte_range.start <= w[0].byte_range.end,
            "gap between chunks would silently drop text"
        );
    }
    assert!(chunks.last().unwrap().byte_range.end <= text.len());
}

#[test]
fn byte_ranges_index_the_original_text() {
    let config = ChunkConfig {
        max_tokens: 32,
        overlap_tokens: 0,
        min_final_tokens: 1,
    };
    let text = (0..20)
        .map(|_| format!("{} .", words(8)))
        .collect::<Vec<_>>()
        .join(" ");

    for c in collect(&text, config) {
        // Must be a valid slice — the whole point of tracking provenance.
        let slice = &text[c.byte_range.clone()];
        assert!(!slice.trim().is_empty());
    }
}

#[test]
fn prefers_sentence_boundaries_over_arbitrary_cuts() {
    let config = ChunkConfig {
        max_tokens: 14,
        overlap_tokens: 0,
        min_final_tokens: 1,
    };
    // Each sentence is 4 letters + a period = 5 tokens; 2 fit per 12-token window.
    let text = "a b c d. e f g h. i j k l. m n o p.";
    let tk = toy_tokenizer();
    let chunks: Vec<_> = chunk_str(text, &tk, config).map(Result::unwrap).collect();

    for c in &chunks {
        let slice = text[c.byte_range.clone()].trim();
        assert!(
            slice.ends_with('.'),
            "chunk {slice:?} was cut mid-sentence despite the window having room"
        );
    }
}

#[test]
fn a_single_oversized_segment_is_split_rather_than_dropped() {
    let config = ChunkConfig {
        max_tokens: 10,
        overlap_tokens: 0,
        min_final_tokens: 1,
    };
    // One "sentence" far longer than a window, with no internal boundary.
    let chunks = collect(&words(95), config);

    let total: usize = chunks.iter().map(|c| c.token_count).sum();
    assert!(chunks.len() >= 10);
    assert_eq!(
        total, 95,
        "hard-splitting must not lose or duplicate tokens"
    );
}

#[test]
fn text_with_no_boundaries_at_all_stays_bounded() {
    let config = ChunkConfig {
        max_tokens: 64,
        overlap_tokens: 0,
        min_final_tokens: 1,
    };
    // 200k bytes, no whitespace, no punctuation — the pathological streaming case.
    let text = "a".repeat(200_000);
    let chunks = collect(&text, config);

    assert!(!chunks.is_empty());
    for c in &chunks {
        assert!(c.ids.len() <= config.max_tokens);
    }
}

#[test]
fn a_streaming_source_chunks_the_same_as_the_whole_string() {
    let config = ChunkConfig {
        max_tokens: 18,
        overlap_tokens: 4,
        min_final_tokens: 1,
    };
    let text = (0..30)
        .map(|_| format!("{} .", words(6)))
        .collect::<Vec<_>>()
        .join(" ");
    let tk = toy_tokenizer();

    let whole: Vec<_> = chunk_str(&text, &tk, config).map(Result::unwrap).collect();

    // The same document arriving in 7-byte dribbles.
    let pieces: Vec<String> = text
        .as_bytes()
        .chunks(7)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect();
    let streamed: Vec<_> = Chunker::new(pieces.into_iter(), &tk, config)
        .map(Result::unwrap)
        .collect();

    assert_eq!(whole.len(), streamed.len());
    for (a, b) in whole.iter().zip(&streamed) {
        assert_eq!(a.ids, b.ids);
        assert_eq!(a.byte_range, b.byte_range);
    }
}

#[test]
fn lazy_enough_for_an_endless_source() {
    let config = ChunkConfig {
        max_tokens: 16,
        overlap_tokens: 2,
        min_final_tokens: 1,
    };
    let tk = toy_tokenizer();

    // An infinite source. Taking a finite prefix must terminate, which it only
    // can if chunking never tries to see the end of the document.
    let endless = std::iter::repeat_with(|| format!("{} . ", words(4)));
    let chunks: Vec<_> = Chunker::new(endless, &tk, config)
        .take(50)
        .map(Result::unwrap)
        .collect();

    assert_eq!(chunks.len(), 50);
    assert!(chunks.iter().all(|c| c.ids.len() <= config.max_tokens));
}

#[test]
fn empty_input_yields_nothing() {
    assert!(collect("", ChunkConfig::default()).is_empty());
    assert!(collect("   \n\n  \t ", ChunkConfig::default()).is_empty());
}

// ── unicode ─────────────────────────────────────────────────────────────────

/// Byte ranges are used to slice the source, so a range landing mid-character
/// is not an inaccuracy — it is a panic at the call site.
#[test]
fn byte_ranges_land_on_character_boundaries() {
    let config = ChunkConfig::new(24).with_overlap(4);
    // Multi-byte throughout: accents (2 bytes), CJK (3), emoji (4).
    let text = "café ünïcodé. 日本語のテキストです。 🎉🚀 emoji! Ende.".repeat(40);

    for c in collect(&text, config) {
        assert!(
            text.is_char_boundary(c.byte_range.start),
            "start {} is mid-character",
            c.byte_range.start
        );
        assert!(
            text.is_char_boundary(c.byte_range.end),
            "end {} is mid-character",
            c.byte_range.end
        );
        let _ = &text[c.byte_range.clone()]; // must not panic
    }
}

#[test]
fn a_multi_byte_source_split_mid_character_still_streams() {
    // A byte-oriented reader hands over pieces that split characters; the
    // segmenter has to rejoin them rather than lose or mangle text.
    let config = ChunkConfig::new(24).with_overlap(0);
    let tk = toy_tokenizer();
    let text = "日本語のテキスト。".repeat(30);

    let pieces: Vec<String> = text
        .as_bytes()
        .chunks(5)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect();

    let chunks: Vec<_> = Chunker::new(pieces.into_iter(), &tk, config)
        .collect::<Result<Vec<_>, _>>()
        .expect("streaming multi-byte text should not fail");
    assert!(!chunks.is_empty());
    for c in &chunks {
        assert!(c.ids.len() <= config.max_tokens);
    }
}

// ── min_final_tokens ──────────────────────────────────────────────────────────────

/// A sentence of exactly 12 tokens: 11 letters and a full stop.
fn twelve_token_sentence() -> String {
    format!("{} .", words(11))
}

/// With a 22-token window only one such sentence fits, so *every* window is 12
/// tokens — comfortably short of capacity. That makes both behaviours visible at
/// once: intermediate windows survive a minimum of 13, and only the tail is cut.
#[test]
fn the_minimum_governs_the_tail_and_not_every_window() {
    let text = vec![twelve_token_sentence(); 10].join(" ");
    let window = ChunkConfig::new(24).with_overlap(0);

    let kept = collect(&text, window.with_min_final_tokens(0));
    assert_eq!(kept.len(), 10, "one sentence per window");
    assert!(
        kept.iter().all(|c| c.token_count == 12),
        "every window is short of the 22-token capacity"
    );

    let filtered = collect(&text, window.with_min_final_tokens(13));
    assert_eq!(
        filtered.len(),
        9,
        "only the trailing window should be dropped, not all ten"
    );
    assert!(
        filtered.iter().all(|c| c.token_count == 12),
        "short intermediate windows must survive the minimum"
    );
}

#[test]
fn a_document_shorter_than_the_minimum_still_yields_its_one_chunk() {
    // Dropping every chunk would silently classify nothing at all.
    let config = ChunkConfig::new(64).with_min_final_tokens(50);
    let chunks = collect("a b c", config);
    assert_eq!(chunks.len(), 1, "the only chunk must survive the minimum");
    assert_eq!(chunks[0].token_count, 3);
}

// ── overlap ─────────────────────────────────────────────────────────────────

#[test]
fn zero_overlap_produces_disjoint_chunks() {
    let config = ChunkConfig::new(24).with_overlap(0);
    let chunks = collect(&words(300), config);

    for w in chunks.windows(2) {
        assert!(
            w[1].byte_range.start >= w[0].byte_range.end,
            "chunks overlap despite overlap_tokens = 0: {:?} then {:?}",
            w[0].byte_range,
            w[1].byte_range
        );
    }
}

// ── boundaries ──────────────────────────────────────────────────────────────

#[test]
fn windows_pack_across_paragraph_breaks() {
    let config = ChunkConfig::new(64).with_overlap(0);
    let text = (0..12)
        .map(|_| format!("{} .", words(6)))
        .collect::<Vec<_>>()
        .join("\n\n");

    let chunks = collect(&text, config);
    assert!(!chunks.is_empty());
    let total: usize = chunks.iter().map(|c| c.token_count).sum();
    assert!(
        total >= 12 * 7,
        "paragraph text should not be lost: {total}"
    );
}

#[test]
fn punctuation_only_input_does_not_loop_forever() {
    let config = ChunkConfig::new(16).with_overlap(2);
    let chunks = collect(&". ! ? . ! ? ".repeat(50), config);
    for c in &chunks {
        assert!(c.ids.len() <= config.max_tokens);
    }
}
