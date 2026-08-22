//! Splitting unbounded text into windows the encoder can actually see.
//!
//! MiniLM was trained at 256 tokens ([`crate::TRAINED_SEQ_LEN`]), so a document
//! of arbitrary length has to become a sequence of windows. Two properties matter
//! here beyond "cut it up":
//!
//! **It streams.** Input is an iterator of string pieces, not a `String`. Text is
//! segmented and tokenized incrementally and only a bounded buffer plus the
//! current window is ever resident, so a genuinely unbounded source costs
//! constant memory. Nothing ever holds the whole document.
//!
//! **It cuts at boundaries.** Windows are packed out of whole segments —
//! paragraphs, then sentences — so a chunk rarely severs a sentence. A single
//! segment longer than the window is hard-split as a fallback, which is the only
//! case where a chunk boundary lands mid-thought.

use crate::error::Result;
use crate::tokenize::Tokenizer;
use core::ops::Range;

/// Largest buffer the segmenter will accumulate before force-splitting.
///
/// Bounds memory when input contains no boundary characters at all — minified
/// data, a single enormous line — which is exactly the pathological case that
/// would otherwise defeat streaming.
const MAX_SEGMENT_BYTES: usize = 8 * 1024;

/// How documents are windowed.
///
/// ```
/// use burn_setfit::ChunkConfig;
///
/// let chunking = ChunkConfig::new(256).with_overlap(32).with_min_final_tokens(16);
///
/// // Two of the 256 tokens are spent on [CLS] and [SEP].
/// assert_eq!(chunking.capacity(), 254);
/// chunking.validate()?;
/// # Ok::<(), burn_setfit::SetFitError>(())
/// ```
///
/// Windowing that could not make progress is refused rather than adjusted:
///
/// ```
/// use burn_setfit::ChunkConfig;
///
/// // Carrying forward as much as a window holds means each window re-reads
/// // what the last one did, and the document never advances.
/// let err = ChunkConfig::new(32).with_overlap(30).validate().unwrap_err();
/// assert!(format!("{err}").contains("never advance"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChunkConfig {
    /// Maximum sequence length including `[CLS]` and `[SEP]`.
    pub max_tokens: usize,
    /// Tokens of trailing context carried into the next window.
    ///
    /// Overlap keeps a claim that straddles a boundary intact in at least one
    /// window. Costs proportionally more compute, so it is not free.
    pub overlap_tokens: usize,
    /// Shortest trailing chunk worth emitting.
    ///
    /// Applies to the **last** chunk of a document only, and never to the only
    /// one. Intermediate windows are packed out of whole segments, so they
    /// routinely stop short of capacity — a blanket minimum would discard
    /// ordinary content, and a document whose windows all happen to be
    /// boundary-aligned and short would classify as nothing at all. What is
    /// worth suppressing is the opposite case: a final scrap consisting mostly
    /// of carried-over overlap, which contributes almost no new text but votes
    /// as loudly as any other chunk.
    pub min_final_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: crate::TRAINED_SEQ_LEN,
            overlap_tokens: 32,
            min_final_tokens: 16,
        }
    }
}

impl ChunkConfig {
    /// Content tokens per window, after reserving room for `[CLS]` and `[SEP]`.
    pub fn capacity(&self) -> usize {
        self.max_tokens.saturating_sub(2).max(1)
    }

    /// A window of `max_tokens`, with no overlap and no minimum.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            overlap_tokens: 0,
            min_final_tokens: 0,
        }
    }

    /// Set the trailing context carried into the next window.
    pub fn with_overlap(mut self, overlap_tokens: usize) -> Self {
        self.overlap_tokens = overlap_tokens;
        self
    }

    /// Set the shortest trailing chunk worth keeping.
    pub fn with_min_final_tokens(mut self, min_final_tokens: usize) -> Self {
        self.min_final_tokens = min_final_tokens;
        self
    }

    /// Reject windowing that could not make progress.
    ///
    /// The overlap check is the one that matters: carrying forward as much as a
    /// window holds means each window re-reads what the last one did and the
    /// document never advances.
    pub fn validate(&self) -> crate::Result<()> {
        if self.max_tokens < 3 {
            return Err(crate::SetFitError::Config(format!(
                "max_tokens must leave room for [CLS], [SEP] and content, got {}",
                self.max_tokens
            )));
        }
        if self.overlap_tokens >= self.capacity() {
            return Err(crate::SetFitError::Config(format!(
                "overlap_tokens ({}) must be less than the {} content tokens a window holds, \
                 or windows would never advance",
                self.overlap_tokens,
                self.capacity()
            )));
        }
        if self.min_final_tokens > self.capacity() {
            return Err(crate::SetFitError::Config(format!(
                "min_final_tokens ({}) exceeds the {} content tokens a window holds, \
                 so the last chunk of every document would be discarded",
                self.min_final_tokens,
                self.capacity()
            )));
        }
        Ok(())
    }
}

/// One window of a document, ready to encode.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Token ids including `[CLS]`/`[SEP]`.
    pub ids: Vec<u32>,
    /// Content tokens, excluding specials. Used to weight the chunk's vote.
    pub token_count: usize,
    /// Byte span in the logical document, for tracing a label back to its source.
    pub byte_range: Range<usize>,
}

/// A boundary-delimited piece of source text.
#[derive(Debug, Clone)]
struct Segment {
    text: String,
    byte_range: Range<usize>,
}

/// Streaming, bounded-memory splitter over an iterator of text pieces.
struct Segmenter<I> {
    source: I,
    buffer: String,
    /// Byte offset, in the logical document, of `buffer[0]`.
    buffer_start: usize,
    exhausted: bool,
}

impl<I: Iterator<Item = String>> Segmenter<I> {
    fn new(source: I) -> Self {
        Self {
            source,
            buffer: String::new(),
            buffer_start: 0,
            exhausted: false,
        }
    }

    /// Byte index just past the first segment boundary in `s`, if any.
    ///
    /// A paragraph break wins over a sentence end; both consume the trailing
    /// whitespace so it does not open the next segment.
    fn find_boundary(s: &str) -> Option<usize> {
        let bytes = s.as_bytes();
        let mut para: Option<usize> = None;
        let mut sentence: Option<usize> = None;

        for (i, &b) in bytes.iter().enumerate() {
            if b == b'\n' && bytes.get(i + 1) == Some(&b'\n') {
                para = Some(consume_whitespace(s, i));
                break;
            }
            if matches!(b, b'.' | b'!' | b'?')
                && sentence.is_none()
                && bytes.get(i + 1).is_some_and(|c| c.is_ascii_whitespace())
            {
                sentence = Some(consume_whitespace(s, i + 1));
            }
        }

        para.or(sentence)
    }
}

/// Byte index of the first non-whitespace byte at or after `from`.
fn consume_whitespace(s: &str, from: usize) -> usize {
    let bytes = s.as_bytes();
    let mut i = from;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// Largest index `<= max` that sits on a UTF-8 character boundary.
fn floor_char_boundary(s: &str, max: usize) -> usize {
    if max >= s.len() {
        return s.len();
    }
    let mut i = max;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

impl<I: Iterator<Item = String>> Iterator for Segmenter<I> {
    type Item = Segment;

    fn next(&mut self) -> Option<Segment> {
        loop {
            if let Some(end) = Self::find_boundary(&self.buffer) {
                return Some(self.take(end));
            }

            // No boundary yet, and the buffer has grown past what we are willing
            // to hold. Cut it anyway — bounded memory beats a clean split here.
            if self.buffer.len() >= MAX_SEGMENT_BYTES {
                let end = floor_char_boundary(&self.buffer, MAX_SEGMENT_BYTES);
                return Some(self.take(end));
            }

            match self.source.next() {
                Some(piece) => self.buffer.push_str(&piece),
                None => {
                    self.exhausted = true;
                    if self.buffer.trim().is_empty() {
                        return None;
                    }
                    let end = self.buffer.len();
                    return Some(self.take(end));
                }
            }
        }
    }
}

impl<I> Segmenter<I> {
    /// Split `end` bytes off the front of the buffer as a segment.
    fn take(&mut self, end: usize) -> Segment {
        let text: String = self.buffer.drain(..end).collect();
        let range = self.buffer_start..self.buffer_start + end;
        self.buffer_start += end;
        Segment {
            text,
            byte_range: range,
        }
    }
}

/// A tokenized segment awaiting packing.
struct Packed {
    ids: Vec<u32>,
    byte_range: Range<usize>,
}

/// Packs segments into overlapping token windows.
///
/// Yields [`Chunk`]s lazily; pull as many as you like from an endless source.
pub struct Chunker<'a, I> {
    segments: Segmenter<I>,
    tokenizer: &'a Tokenizer,
    config: ChunkConfig,
    /// Segments accumulated toward the window currently being built.
    window: Vec<Packed>,
    window_tokens: usize,
    /// Ids split off an oversized segment, still waiting to be emitted.
    pending: Vec<Packed>,
    emitted: usize,
    done: bool,
}

impl<'a, I: Iterator<Item = String>> Chunker<'a, I> {
    /// Chunk a streaming source.
    pub fn new(source: I, tokenizer: &'a Tokenizer, config: ChunkConfig) -> Self {
        Self {
            segments: Segmenter::new(source),
            tokenizer,
            config,
            window: Vec::new(),
            window_tokens: 0,
            pending: Vec::new(),
            emitted: 0,
            done: false,
        }
    }

    /// Finish the current window and begin the next one carrying overlap.
    fn flush(&mut self) -> Option<Chunk> {
        if self.window.is_empty() {
            return None;
        }

        let mut ids = Vec::with_capacity(self.window_tokens);
        for p in &self.window {
            ids.extend_from_slice(&p.ids);
        }
        let byte_range =
            self.window[0].byte_range.start..self.window[self.window.len() - 1].byte_range.end;

        // Carry trailing segments into the next window. Whole segments only —
        // carrying part of one would reintroduce the mid-sentence cuts this
        // design exists to avoid.
        let capacity = self.config.capacity();
        let mut carry_count = 0usize;
        let mut carried = 0usize;
        for p in self.window.iter().rev() {
            // Never carry the entire window: the next one has to make progress.
            if carry_count + 1 >= self.window.len() {
                break;
            }
            if carried + p.ids.len() > self.config.overlap_tokens {
                break;
            }
            carried += p.ids.len();
            carry_count += 1;
        }

        // A segment larger than the whole overlap budget would otherwise mean a
        // caller who asked for overlap silently gets none. Carry one anyway, but
        // only up to half a window, so an oversized tail cannot crowd out the
        // content the next window is supposed to hold.
        if carry_count == 0 && self.config.overlap_tokens > 0 && self.window.len() > 1 {
            let last = self.window.last().expect("window is non-empty");
            if last.ids.len() <= capacity / 2 {
                carried = last.ids.len();
                carry_count = 1;
            }
        }

        let carry = self.window.split_off(self.window.len() - carry_count);
        self.window = carry;
        self.window_tokens = carried;
        self.emitted += 1;

        let token_count = ids.len();
        Some(Chunk {
            ids: self.tokenizer.add_specials(&ids),
            token_count,
            byte_range,
        })
    }

    /// Tokenize the next segment, hard-splitting it if it exceeds a whole window.
    fn next_packed(&mut self) -> Option<Result<Packed>> {
        if let Some(p) = self.pending.pop() {
            return Some(Ok(p));
        }

        let segment = self.segments.next()?;
        let ids = match self.tokenizer.encode_bare(&segment.text) {
            Ok(ids) => ids,
            Err(e) => return Some(Err(e)),
        };
        if ids.is_empty() {
            // Whitespace-only segment; skip it rather than emitting an empty chunk.
            return self.next_packed();
        }

        let capacity = self.config.capacity();
        if ids.len() <= capacity {
            return Some(Ok(Packed {
                ids,
                byte_range: segment.byte_range,
            }));
        }

        // One segment longer than a whole window — an unpunctuated wall of text.
        // Split on token count. Byte ranges are apportioned proportionally, so
        // they are approximate for these pieces only.
        let total = ids.len();
        let span = segment.byte_range.end - segment.byte_range.start;
        let mut pieces: Vec<Packed> = ids
            .chunks(capacity)
            .enumerate()
            .map(|(i, part)| {
                let start = segment.byte_range.start + span * (i * capacity) / total;
                let end = segment.byte_range.start
                    + (span * (i * capacity + part.len()) / total).min(span);
                Packed {
                    ids: part.to_vec(),
                    byte_range: start..end,
                }
            })
            .collect();

        let first = pieces.remove(0);
        pieces.reverse();
        self.pending = pieces;
        Some(Ok(first))
    }
}

impl<I: Iterator<Item = String>> Iterator for Chunker<'_, I> {
    type Item = Result<Chunk>;

    fn next(&mut self) -> Option<Result<Chunk>> {
        if self.done {
            return None;
        }

        let capacity = self.config.capacity();

        loop {
            match self.next_packed() {
                Some(Err(e)) => {
                    self.done = true;
                    return Some(Err(e));
                }
                Some(Ok(packed)) => {
                    if self.window_tokens + packed.ids.len() > capacity && !self.window.is_empty() {
                        let chunk = self.flush();
                        self.window_tokens += packed.ids.len();
                        self.window.push(packed);
                        if let Some(chunk) = chunk {
                            return Some(Ok(chunk));
                        }
                    } else {
                        self.window_tokens += packed.ids.len();
                        self.window.push(packed);
                    }
                }
                None => {
                    self.done = true;
                    // Drop a final scrap that is mostly carried-over overlap, but
                    // never return nothing at all for a non-empty document.
                    if self.window_tokens < self.config.min_final_tokens && self.emitted > 0 {
                        return None;
                    }
                    return self.flush().map(Ok);
                }
            }
        }
    }
}

/// Chunk a single in-memory string.
///
/// ```no_run
/// use burn_setfit::{ChunkConfig, Tokenizer};
/// use burn_setfit::chunk::chunk_str;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let tokenizer = Tokenizer::from_bytes(&[])?;
/// # let document = "";
/// for chunk in chunk_str(document, &tokenizer, ChunkConfig::default()) {
///     let chunk = chunk?;
///     // Each chunk knows where it came from, which is what lets a label be
///     // traced back to the passage that caused it.
///     println!("{} tokens from bytes {:?}", chunk.token_count, chunk.byte_range);
/// }
/// # Ok(())
/// # }
/// ```
pub fn chunk_str<'a>(
    text: &str,
    tokenizer: &'a Tokenizer,
    config: ChunkConfig,
) -> Chunker<'a, core::iter::Once<String>> {
    Chunker::new(core::iter::once(text.to_string()), tokenizer, config)
}
