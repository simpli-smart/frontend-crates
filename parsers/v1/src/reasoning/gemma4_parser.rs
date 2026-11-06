// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reasoning parser for Google Gemma 4 thinking models.
//!
//! Gemma 4 emits chain-of-thought reasoning between dedicated channel
//! delimiters:
//!
//! ```text
//! <|channel>thought
//! ...chain of thought reasoning...<channel|>
//! Final answer text.
//! ```
//!
//! Parsing semantics align with vLLM/SGLang ``Gemma4Detector`` (snapshot-based
//! streaming on incomplete-marker-safe prefixes).

use crate::ParserResult;
use crate::ReasoningParser;
use crate::tool_calling::gemma4::{
    diff_reasoning_streaming_snapshots, extract_reasoning_non_streaming, safe_prefix_len,
    OpenerLessNormalizer, ReasoningSnapshot, CHANNEL_END, CHANNEL_START, THOUGHT_PREFIX,
    TOOL_CALL_START,
};

/// How the snapshot of the last parsed prefix is held. The two common streaming states
/// are kept implicit so a chunk costs O(chunk) instead of re-extracting the whole text.
#[derive(Debug, Clone, Default)]
enum LastSnapshot {
    /// No `<|channel>` in the prefix: the snapshot is the text itself as content.
    #[default]
    Plain,
    /// Inside the first `<|channel>` span (no `<channel|>`, no `<|tool_call>` yet):
    /// reasoning is the trimmed body from `body_start`, no content.
    OpenThought { body_start: usize },
    /// Anything else: the stored snapshot.
    Full(ReasoningSnapshot),
}

#[derive(Debug, Clone)]
pub struct Gemma4ReasoningParser {
    /// Opener-less/stray-marker normalization; its output is append-only.
    norm: OpenerLessNormalizer,
    /// Length of `norm.output()` when the previous chunk was processed.
    prev_out_len: usize,
    /// Incomplete-marker-safe prefix of `norm.output()` already parsed.
    last_safe_len: usize,
    last_snapshot: LastSnapshot,
    /// Client-visible content already emitted (avoids duplicate pre-channel text).
    emitted_content: String,
    /// `norm.output()` scanned up to here for `<channel|>` / `<|tool_call>` after the
    /// first `<|channel>`.
    open_scan_to: usize,
    /// One of those markers was found: the in-thought fast path no longer applies.
    open_span_ended: bool,
    /// Use the O(chunk) paths for the plain and in-thought states (tests turn this off
    /// to check them against the general path).
    fast_paths: bool,
}

impl Gemma4ReasoningParser {
    pub fn new() -> Self {
        Self {
            norm: OpenerLessNormalizer::new(),
            prev_out_len: 0,
            last_safe_len: 0,
            last_snapshot: LastSnapshot::Plain,
            emitted_content: String::new(),
            open_scan_to: 0,
            open_span_ended: false,
            fast_paths: true,
        }
    }

    #[cfg(test)]
    fn without_fast_paths() -> Self {
        Self {
            fast_paths: false,
            ..Self::new()
        }
    }

    /// True while the first `<|channel>` span is open (no `<channel|>` yet).
    fn inside_open_first_channel(text: &str) -> bool {
        let Some((_, after_start)) = text.split_once(CHANNEL_START) else {
            return false;
        };
        !after_start.contains(CHANNEL_END)
    }

    fn snapshot_to_parser_result(snapshot: ReasoningSnapshot) -> ParserResult {
        ParserResult {
            normal_text: snapshot.content.unwrap_or_default(),
            reasoning_text: snapshot.reasoning.unwrap_or_default(),
        }
    }

    /// Body offset of the first thought span (after `<|channel>` and a complete
    /// `thought\n`), when `out[..len]` holds a complete `<|channel>`.
    fn open_body_start(&self, len: usize) -> Option<usize> {
        let out = self.norm.output();
        let p = self.norm.first_channel_start()? + CHANNEL_START.len();
        if p > len {
            return None;
        }
        Some(if out[p..len].starts_with(THOUGHT_PREFIX) { p + THOUGHT_PREFIX.len() } else { p })
    }

    /// Advance the scan for markers that end the open first span, up to `len`.
    fn scan_open_span(&mut self, len: usize) {
        let Some(start) = self.norm.first_channel_start().map(|p| p + CHANNEL_START.len()) else {
            return;
        };
        if self.open_span_ended || len <= start {
            return;
        }
        let out = self.norm.output();
        let overlap = CHANNEL_END.len().max(TOOL_CALL_START.len()) - 1;
        let mut from = self.open_scan_to.saturating_sub(overlap).max(start);
        while !out.is_char_boundary(from) {
            from -= 1;
        }
        let window = &out[from..len];
        if window.contains(CHANNEL_END) || window.contains(TOOL_CALL_START) {
            self.open_span_ended = true;
        }
        self.open_scan_to = len;
    }

    /// The snapshot of the previously parsed prefix `norm.output()[..len]`, exactly as
    /// `extract_reasoning_non_streaming` returns it.
    fn take_last_snapshot(&mut self, len: usize) -> ReasoningSnapshot {
        let prefix = &self.norm.output()[..len];
        match std::mem::take(&mut self.last_snapshot) {
            LastSnapshot::Full(s) => s,
            LastSnapshot::Plain => ReasoningSnapshot {
                reasoning: None,
                content: (!prefix.is_empty()).then(|| prefix.to_string()),
            },
            LastSnapshot::OpenThought { body_start } => {
                let r = prefix[body_start..].trim();
                ReasoningSnapshot {
                    reasoning: (!r.is_empty()).then(|| r.to_string()),
                    content: None,
                }
            }
        }
    }

    /// Reset for the next stream (keeps configuration).
    fn reset(&mut self) {
        *self = Self {
            fast_paths: self.fast_paths,
            ..Self::new()
        };
    }
}

impl Default for Gemma4ReasoningParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for Gemma4ReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, _token_ids: &[u32]) -> ParserResult {
        let text = OpenerLessNormalizer::normalize_all(text);
        Self::snapshot_to_parser_result(extract_reasoning_non_streaming(&text))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        _token_ids: &[u32],
    ) -> ParserResult {
        self.norm.push(text);
        // Output is held until the start is classified (see `OpenerLessNormalizer`).
        if !self.norm.is_decided() {
            return ParserResult::default();
        }
        let prev_out_len = std::mem::replace(&mut self.prev_out_len, self.norm.output().len());
        let safe_len = safe_prefix_len(self.norm.output());
        // The output is append-only, so the safe prefix only grows.
        if safe_len <= self.last_safe_len {
            return ParserResult::default();
        }
        let last_safe_len = std::mem::replace(&mut self.last_safe_len, safe_len);
        let first_channel = self.norm.first_channel_start();
        let channel_in = |len: usize| first_channel.is_some_and(|p| p + CHANNEL_START.len() <= len);
        let has_channel = channel_in(safe_len);
        self.scan_open_span(safe_len);

        if self.fast_paths {
            // Plain output: the snapshot is the text itself, so the delta is the new text.
            if !channel_in(self.prev_out_len) && matches!(self.last_snapshot, LastSnapshot::Plain) {
                let dc = self.norm.output()[last_safe_len..safe_len].to_string();
                self.emitted_content.push_str(&dc);
                return ParserResult {
                    normal_text: dc,
                    reasoning_text: String::new(),
                };
            }
            // Inside the first thought span: reasoning is the trimmed body, content is
            // held (suppressed) -- the delta is the newly trimmed reasoning text.
            if has_channel
                && !self.open_span_ended
                && matches!(self.last_snapshot, LastSnapshot::Plain | LastSnapshot::OpenThought { .. })
                && let Some(body_start) = self.open_body_start(safe_len)
            {
                let out = self.norm.output();
                // Trimmed reasoning of the previous prefix, as an `out` byte range.
                let prev_r = match self.last_snapshot {
                    LastSnapshot::OpenThought { body_start: b } => {
                        let t = out[b..last_safe_len].trim_start();
                        let start = last_safe_len - t.len();
                        start..start + t.trim_end().len()
                    }
                    _ => 0..0,
                };
                let t = out[body_start..safe_len].trim_start();
                let start = safe_len - t.len();
                let curr_r = start..start + t.trim_end().len();
                // Same buffer: a range with the same start is a prefix (O(delta)); any
                // other change falls back to the string comparison the general path uses.
                let dr = if prev_r.is_empty() {
                    &out[curr_r]
                } else if prev_r.start == curr_r.start && prev_r.end <= curr_r.end {
                    &out[prev_r.end..curr_r.end]
                } else {
                    let (c, p) = (&out[curr_r], &out[prev_r]);
                    c.strip_prefix(p).unwrap_or(c)
                }
                .to_string();
                self.last_snapshot = LastSnapshot::OpenThought { body_start };
                return ParserResult {
                    normal_text: String::new(),
                    reasoning_text: dr,
                };
            }
        }

        let mut prev_snapshot = self.take_last_snapshot(last_safe_len);
        // Pre-channel text must not be emitted twice once the channel opens.
        let channel_just_opened = !channel_in(prev_out_len) && channel_in(self.prev_out_len);
        if channel_just_opened {
            prev_snapshot.content = None;
        }

        let out = self.norm.output();
        let curr_snapshot = extract_reasoning_non_streaming(&out[..safe_len]);
        let (dr, mut dc) = diff_reasoning_streaming_snapshots(&curr_snapshot, &prev_snapshot);

        if Self::inside_open_first_channel(out) {
            dc.clear();
        } else if let Some(ref full) = curr_snapshot.content {
            if !self.emitted_content.is_empty() {
                if full.starts_with(&self.emitted_content) {
                    dc = full[self.emitted_content.len()..].to_string();
                } else if self.emitted_content.starts_with(full.as_str()) {
                    // Trimming made the snapshot shorter than what was already sent
                    // (e.g. `Hello <|channel>..<channel|>` + a whitespace-only chunk).
                    dc.clear();
                }
            }
            if !dc.is_empty() {
                self.emitted_content = full.clone();
            }
        }
        self.last_snapshot = LastSnapshot::Full(curr_snapshot);

        ParserResult {
            normal_text: dc,
            reasoning_text: dr,
        }
    }

    /// Flush what the stream has not emitted yet (the trait contract: only the
    /// remainder, like every other parser): output still held while the start was
    /// being classified, the tail kept back as a possible split marker, and content
    /// held back inside an unclosed thought span (e.g. a tool call).
    fn finish_reasoning_stream(&mut self) -> ParserResult {
        self.norm.finish();
        if self.norm.output().is_empty() {
            self.reset();
            return ParserResult::default();
        }
        let prev_snapshot = self.take_last_snapshot(self.last_safe_len);
        let final_snapshot = extract_reasoning_non_streaming(self.norm.output());
        let (reasoning_text, diff_content) =
            diff_reasoning_streaming_snapshots(&final_snapshot, &prev_snapshot);
        // Content: what the client has not seen. Content held back while streaming
        // (inside an unclosed span) is in the last snapshot but was never emitted, so
        // diff against what was emitted, not against the snapshot.
        let normal_text = match final_snapshot.content {
            Some(content) if content.starts_with(&self.emitted_content) => {
                content[self.emitted_content.len()..].to_string()
            }
            Some(content) if self.emitted_content.starts_with(content.as_str()) => String::new(),
            Some(_) => diff_content,
            None => String::new(),
        };
        self.reset();
        ParserResult {
            normal_text,
            reasoning_text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_basic_thinking() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning(
            "<|channel>thought\nstep one\nstep two<channel|>The answer is 42.",
            &[],
        );
        assert_eq!(r.reasoning_text, "step one\nstep two");
        assert_eq!(r.normal_text, "The answer is 42.");
    }

    #[test]
    fn detect_no_markers_passes_through() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning("just a plain answer", &[]);
        assert_eq!(r.reasoning_text, "");
        assert_eq!(r.normal_text, "just a plain answer");
    }

    #[test]
    fn detect_truncated_reasoning_open_only() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning("intro <|channel>thought\npartial", &[]);
        assert_eq!(r.reasoning_text, "partial");
        assert_eq!(r.normal_text, "");
    }

    #[test]
    fn detect_text_before_and_after() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning(
            "Hello. <|channel>thought\nrumination<channel|> Goodbye.",
            &[],
        );
        assert_eq!(r.reasoning_text, "rumination");
        assert_eq!(r.normal_text, "Hello.  Goodbye.");
    }

    // A `<channel|>` with no opener is never client-visible (vLLM absorbs it in
    // CONTENT state); the surrounding text is kept as content because it is not a role
    // label, so it cannot be told apart from a real answer.
    #[test]
    fn detect_dangling_end_marker_is_absorbed() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning("some thinking<channel|>final answer", &[]);
        assert_eq!(r.reasoning_text, "");
        assert_eq!(r.normal_text, "some thinkingfinal answer");
    }

    // Bare `thought\n` with no `<|channel>` opener is an implicitly opened channel
    // (matches vLLM's gemma4 parser): the label and `<channel|>` must not reach content.
    #[test]
    fn detect_bare_thought_prefix_is_reasoning() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning("thought\nrumination<channel|>final answer", &[]);
        assert_eq!(r.reasoning_text, "rumination");
        assert_eq!(r.normal_text, "final answer");
    }

    // Exact shape captured from the Exotel leakage suite (post-tool turns).
    #[test]
    fn detect_bare_empty_thought_post_tool() {
        for raw in [
            "thought\n<channel|>Great news, Rajesh!",
            "\nthought\n<channel|>Great news, Rajesh!",
        ] {
            let mut p = Gemma4ReasoningParser::new();
            let r = p.detect_and_parse_reasoning(raw, &[]);
            assert!(!r.normal_text.contains("thought"), "{raw:?} -> {:?}", r.normal_text);
            assert!(!r.normal_text.contains("<channel|>"), "{raw:?} -> {:?}", r.normal_text);
            assert_eq!(r.normal_text.trim(), "Great news, Rajesh!");
        }
    }

    #[test]
    fn streaming_bare_empty_thought_post_tool() {
        for chunks in [
            vec!["thought", "\n", "<channel|>", "Great", " news", ", Rajesh!"],
            vec!["\n", "thought", "\n", "<channel|>", "Great news, Rajesh!"],
            vec!["thou", "ght\n<channel|>Great news, Rajesh!"],
        ] {
            let mut p = Gemma4ReasoningParser::new();
            let mut normal = String::new();
            let mut reasoning = String::new();
            for c in &chunks {
                let r = p.parse_reasoning_streaming_incremental(c, &[]);
                normal.push_str(&r.normal_text);
                reasoning.push_str(&r.reasoning_text);
            }
            assert!(!normal.contains("thought"), "{chunks:?} -> {normal:?}");
            assert!(!normal.contains("<channel|>"), "{chunks:?} -> {normal:?}");
            assert_eq!(normal.trim(), "Great news, Rajesh!", "{chunks:?}");
            assert!(reasoning.trim().is_empty(), "{chunks:?} -> reasoning {reasoning:?}");
        }
    }

    // The word "thought" mid-answer is ordinary content.
    #[test]
    fn thought_word_in_answer_is_content() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning("I thought about it.\nThe answer is 42.", &[]);
        assert_eq!(r.reasoning_text, "");
        assert_eq!(r.normal_text, "I thought about it.\nThe answer is 42.");
    }

    // Short answers that are a prefix of "thought" must still be emitted at stream end.
    #[test]
    fn streaming_short_prefix_flushed_on_finish() {
        let mut p = Gemma4ReasoningParser::new();
        let r1 = p.parse_reasoning_streaming_incremental("th", &[]);
        let r2 = p.finish_reasoning_stream();
        assert_eq!(format!("{}{}", r1.normal_text, r2.normal_text), "th");
    }

    #[test]
    fn detect_no_thought_prefix() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning(
            "<|channel>raw reasoning without prefix<channel|>answer",
            &[],
        );
        assert_eq!(r.reasoning_text, "raw reasoning without prefix");
        assert_eq!(r.normal_text, "answer");
    }

    #[test]
    fn streaming_single_chunk() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.parse_reasoning_streaming_incremental(
            "<|channel>thought\nrumination<channel|>final",
            &[],
        );
        assert_eq!(r.reasoning_text, "rumination");
        assert_eq!(r.normal_text, "final");
    }

    #[test]
    fn streaming_thought_prefix_split_across_deltas() {
        let mut p = Gemma4ReasoningParser::new();
        let chunks = [
            "<|channel>",
            "thou",
            "ght\n",
            "real reasoning here",
            "<channel|>",
            "the answer.",
        ];
        let mut reasoning = String::new();
        let mut normal = String::new();
        for c in chunks {
            let r = p.parse_reasoning_streaming_incremental(c, &[]);
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        assert_eq!(reasoning, "real reasoning here");
        assert_eq!(normal, "the answer.");
    }

    #[test]
    fn streaming_start_marker_split() {
        let mut p = Gemma4ReasoningParser::new();
        let chunks = [
            "intro ",
            "<|chan",
            "nel>thought\n",
            "rumination",
            "<channel|>",
            "outro",
        ];
        let mut reasoning = String::new();
        let mut normal = String::new();
        for c in chunks {
            let r = p.parse_reasoning_streaming_incremental(c, &[]);
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        assert_eq!(reasoning, "rumination");
        assert_eq!(normal, "intro outro");
    }

    #[test]
    fn streaming_end_marker_split() {
        let mut p = Gemma4ReasoningParser::new();
        let chunks = [
            "<|channel>thought\n",
            "thinking",
            "<chan",
            "nel|>",
            "answer",
        ];
        let mut reasoning = String::new();
        let mut normal = String::new();
        for c in chunks {
            let r = p.parse_reasoning_streaming_incremental(c, &[]);
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        assert_eq!(reasoning, "thinking");
        assert_eq!(normal, "answer");
    }
    #[test]
    fn streaming_no_thought_prefix_streaming() {
        let mut p = Gemma4ReasoningParser::new();
        let chunks = [
            "<|channel>",
            "raw stream of consciousness",
            "<channel|>",
            "answer",
        ];
        let mut reasoning = String::new();
        let mut normal = String::new();
        for c in chunks {
            let r = p.parse_reasoning_streaming_incremental(c, &[]);
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        assert_eq!(reasoning, "raw stream of consciousness");
        assert_eq!(normal, "answer");
    }

    #[test]
    fn streaming_no_markers() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.parse_reasoning_streaming_incremental("plain text only", &[]);
        assert_eq!(r.reasoning_text, "");
        assert_eq!(r.normal_text, "plain text only");
    }

    #[test]
    fn detect_multiple_reasoning_spans_concatenated() {
        let mut p = Gemma4ReasoningParser::new();
        let input =
            "<|channel>thought\nfirst<channel|> middle <|channel>thought\nsecond<channel|> done";
        let r = p.detect_and_parse_reasoning(input, &[]);
        // Every closed span is reasoning, joined in order; only the text between
        // and after them is content. Matches upstream and the conformance suite.
        assert_eq!(r.reasoning_text, "firstsecond");
        assert!(r.normal_text.contains("middle"));
        assert!(r.normal_text.contains("done"));
        assert!(!r.normal_text.contains("second"));
    }

    #[test]
    fn detect_open_channel_tool_handoff() {
        let mut p = Gemma4ReasoningParser::new();
        let input = concat!(
            "<|channel>thought\nplanning",
            "<|tool_call>call:get_weather{location:<|\"|>SF<|\"|>}<tool_call|>",
        );
        let r = p.detect_and_parse_reasoning(input, &[]);
        assert_eq!(r.reasoning_text, "planning");
        assert_eq!(
            r.normal_text,
            r#"<|tool_call>call:get_weather{location:<|"|>SF<|"|>}<tool_call|>"#,
        );
    }

    #[test]
    fn paired_reasoning_then_tool_call_non_streaming() {
        let mut p = Gemma4ReasoningParser::new();
        let input = concat!(
            "<|channel>thought\nthinking about the request<channel|>",
            "<|tool_call>call:get_weather{location:<|\"|>Tokyo<|\"|>}<tool_call|>",
        );
        let r = p.detect_and_parse_reasoning(input, &[]);
        assert_eq!(r.reasoning_text, "thinking about the request");
        assert_eq!(
            r.normal_text, r#"<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>"#,
        );
    }

    #[test]
    fn streaming_suppressed_empty_thinking_not_in_content() {
        let mut p = Gemma4ReasoningParser::new();
        let chunks = [
            "<|channel>",
            "thought\n",
            "<channel|>",
            "<|channel>",
            "Hi. ",
        ];
        let mut normal = String::new();
        for c in chunks {
            let r = p.parse_reasoning_streaming_incremental(c, &[]);
            normal.push_str(&r.normal_text);
        }
        assert!(!normal.contains("<|channel>"));
        assert!(!normal.contains("<channel|>"));
        assert!(normal.starts_with("Hi."));
    }

    /// Every leaked `content` shape captured from the Exotel leakage suite on
    /// Gemma-4-26B-A4B-NVFP4 (post-tool turns), as raw model text, with the content a
    /// client must receive. Run through detect (non-streaming), token-like streaming,
    /// and byte-by-byte streaming; all three must agree.
    const CAPTURED: &[(&str, &str)] = &[
        ("thought\n<channel|>Great news, Rajesh!", "Great news, Rajesh!"),
        ("\nthought\n<channel|>Great news, Amit!", "Great news, Amit!"),
        ("<channel|>मुझे बहुत खेद है", "मुझे बहुत खेद है"),
        ("```thought\n<channel|>मुझे बहुत दुख है", "मुझे बहुत दुख है"),
        ("\u{fffd}thought\n<channel|>أنا أتفهم تماماً", "أنا أتفهم تماماً"),
        ("|thought\n|<channel|>أعتذر جداً على التأخير!", "أعتذر جداً على التأخير!"),
        ("<|\"|>thought\n<channel|>मुझे बहुत दुख है", "मुझे बहुत दुख है"),
        ("<|\"|>أعتذر جداً على التأخير!", "أعتذر جداً على التأخير!"),
        (".\"\"\"}<tool_call|>", ".\"\"\"}"),
        ("thought\nI'm really sorry that your shipment hasn't reached you yet.",
         "I'm really sorry that your shipment hasn't reached you yet."),
    ];

    fn stream_all(chunks: &[&str]) -> (String, String) {
        let mut p = Gemma4ReasoningParser::new();
        let (mut normal, mut reasoning) = (String::new(), String::new());
        for c in chunks {
            let r = p.parse_reasoning_streaming_incremental(c, &[]);
            normal.push_str(&r.normal_text);
            reasoning.push_str(&r.reasoning_text);
        }
        if normal.is_empty() && reasoning.is_empty() {
            let r = p.finish_reasoning_stream();
            normal.push_str(&r.normal_text);
            reasoning.push_str(&r.reasoning_text);
        }
        (normal, reasoning)
    }

    /// Split like a tokenizer would: special markers stay whole, text in small pieces.
    fn token_like_chunks(raw: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut rest = raw;
        let markers = ["<|channel>", "<channel|>", "<tool_call|>", "<|tool_call>", "<|\"|>"];
        while !rest.is_empty() {
            if let Some(m) = markers.iter().find(|m| rest.starts_with(**m)) {
                out.push(m.to_string());
                rest = &rest[m.len()..];
                continue;
            }
            let mut end = 0;
            for (i, c) in rest.char_indices() {
                if i > 0 && (markers.iter().any(|m| rest[i..].starts_with(m)) || i >= 3) {
                    break;
                }
                end = i + c.len_utf8();
            }
            out.push(rest[..end].to_string());
            rest = &rest[end..];
        }
        out
    }

    #[test]
    fn captured_leak_shapes_non_streaming() {
        for (raw, want) in CAPTURED {
            let mut p = Gemma4ReasoningParser::new();
            let r = p.detect_and_parse_reasoning(raw, &[]);
            assert_eq!(r.normal_text.trim(), *want, "raw={raw:?}");
            assert!(r.reasoning_text.trim().is_empty(), "raw={raw:?} reasoning={:?}", r.reasoning_text);
        }
    }

    #[test]
    fn captured_leak_shapes_streaming_token_like() {
        for (raw, want) in CAPTURED {
            let chunks = token_like_chunks(raw);
            let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let (normal, reasoning) = stream_all(&refs);
            assert_eq!(normal.trim(), *want, "raw={raw:?} chunks={chunks:?}");
            assert!(reasoning.trim().is_empty(), "raw={raw:?} reasoning={reasoning:?}");
        }
    }

    #[test]
    fn captured_leak_shapes_streaming_char_by_char() {
        for (raw, want) in CAPTURED {
            let chars: Vec<String> = raw.chars().map(|c| c.to_string()).collect();
            let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
            let (normal, _) = stream_all(&refs);
            assert_eq!(normal.trim(), *want, "raw={raw:?}");
        }
    }

    // Label followed by short opener-less reasoning that is then closed: reasoning, in
    // both modes (the shared ORPHAN_LABEL_HOLD window keeps them consistent).
    #[test]
    fn opener_less_label_with_short_reasoning_is_reasoning() {
        let raw = "thought\nUser wants order status.<channel|>Your order is on its way.";
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning(raw, &[]);
        assert_eq!(r.reasoning_text, "User wants order status.");
        assert_eq!(r.normal_text, "Your order is on its way.");
        let chunks = token_like_chunks(raw);
        let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let (normal, reasoning) = stream_all(&refs);
        assert_eq!(normal.trim(), "Your order is on its way.");
        assert_eq!(reasoning.trim(), "User wants order status.");
    }

    // Ordinary answers are never held back or altered.
    #[test]
    fn ordinary_answers_untouched() {
        for raw in [
            "The answer is 42.",
            "Thought experiment: imagine a train.",
            "thoughtful reply here",
            "I thought about it.\nThe answer is 42.",
            "मुझे बहुत खेद है",
            "4",
        ] {
            let mut p = Gemma4ReasoningParser::new();
            assert_eq!(p.detect_and_parse_reasoning(raw, &[]).normal_text, raw);
            let chunks = token_like_chunks(raw);
            let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            assert_eq!(stream_all(&refs).0, raw, "raw={raw:?}");
        }
    }

    // A first chunk that is clearly not a label is emitted immediately (no hold).
    #[test]
    fn non_label_start_is_not_held() {
        let mut p = Gemma4ReasoningParser::new();
        assert_eq!(p.parse_reasoning_streaming_incremental("Hello", &[]).normal_text, "Hello");
    }

    // Tool-call markup must survive the reasoning parser for the tool parser downstream.
    #[test]
    fn tool_markup_preserved_for_tool_parser() {
        for raw in [
            "<|tool_call>call:get_order_status{awb:<|\"|>123<|\"|>}<tool_call|>",
            "call:get_order_status{awb:<|\"|>123<|\"|>}",
            "thought\n<channel|><|tool_call>call:f{a:<|\"|>x<|\"|>}<tool_call|>",
        ] {
            let mut p = Gemma4ReasoningParser::new();
            let r = p.detect_and_parse_reasoning(raw, &[]);
            assert!(r.normal_text.contains("call:"), "raw={raw:?} -> {:?}", r.normal_text);
            assert!(r.normal_text.contains("<|\"|>"), "raw={raw:?} -> {:?}", r.normal_text);
        }
    }

    // Properly opened channels keep the existing behaviour.
    #[test]
    fn proper_channel_unchanged() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning("<|channel>thought\nplan<channel|>answer", &[]);
        assert_eq!(r.reasoning_text, "plan");
        assert_eq!(r.normal_text, "answer");
    }


    // finish() returns only what was not streamed yet (trait contract).
    #[test]
    fn finish_does_not_duplicate_streamed_content() {
        let mut p = Gemma4ReasoningParser::new();
        let mut normal = String::new();
        for c in ["Hello", " world", ", how are you?"] {
            normal.push_str(&p.parse_reasoning_streaming_incremental(c, &[]).normal_text);
        }
        let tail = p.finish_reasoning_stream();
        normal.push_str(&tail.normal_text);
        assert_eq!(normal, "Hello world, how are you?");
        assert!(tail.reasoning_text.is_empty());
    }

    // An orphan label + short answer is held (it might still be closed) and must be
    // released by finish(), label stripped.
    #[test]
    fn held_orphan_label_released_on_finish() {
        let mut p = Gemma4ReasoningParser::new();
        let mut normal = String::new();
        for c in ["thought", "\n", "OK."] {
            normal.push_str(&p.parse_reasoning_streaming_incremental(c, &[]).normal_text);
        }
        assert_eq!(normal, "", "held until the label is resolved");
        normal.push_str(&p.finish_reasoning_stream().normal_text);
        assert_eq!(normal, "OK.");
    }

    // An incomplete-marker tail kept back during streaming is flushed by finish().
    #[test]
    fn finish_flushes_trailing_incomplete_marker_text() {
        let mut p = Gemma4ReasoningParser::new();
        let mut normal = String::new();
        for c in ["Answer is 5 <", "|"] {
            normal.push_str(&p.parse_reasoning_streaming_incremental(c, &[]).normal_text);
        }
        normal.push_str(&p.finish_reasoning_stream().normal_text);
        assert_eq!(normal, "Answer is 5 <|");
    }

    /// Stream `chunks` through `p`, finish, and return the concatenated (normal, reasoning).
    fn stream_with(mut p: Gemma4ReasoningParser, chunks: &[&str]) -> (String, String) {
        let (mut normal, mut reasoning) = (String::new(), String::new());
        for c in chunks {
            let r = p.parse_reasoning_streaming_incremental(c, &[]);
            normal.push_str(&r.normal_text);
            reasoning.push_str(&r.reasoning_text);
        }
        let r = p.finish_reasoning_stream();
        normal.push_str(&r.normal_text);
        reasoning.push_str(&r.reasoning_text);
        (normal, reasoning)
    }

    /// Deterministic chunkings: per char, token-like, fixed widths, and pseudo-random.
    fn chunkings(raw: &str) -> Vec<Vec<String>> {
        let chars: Vec<char> = raw.chars().collect();
        let mut all = vec![
            chars.iter().map(|c| c.to_string()).collect(),
            token_like_chunks(raw),
            vec![raw.to_string()],
        ];
        for w in [2usize, 3, 5, 7, 11] {
            all.push(chars.chunks(w).map(|c| c.iter().collect()).collect());
        }
        let mut seed = 0x2545_f491_4f6c_dd1du64 ^ raw.len() as u64;
        for _ in 0..12 {
            let mut v = Vec::new();
            let mut i = 0;
            while i < chars.len() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let n = 1 + (seed % 9) as usize;
                v.push(chars[i..(i + n).min(chars.len())].iter().collect());
                i += n;
            }
            all.push(v);
        }
        all
    }

    /// Outputs covering every normalizer rule, the captured leak shapes, the review's
    /// counterexamples, and long plain / thinking text.
    fn corpus() -> Vec<String> {
        let mut v: Vec<String> = CAPTURED.iter().map(|(raw, _)| raw.to_string()).collect();
        v.extend(
            [
                "<|\"|>Sure. <|tool_call>call:f{a:<|\"|>x<|\"|>}<tool_call|>",
                "Ok<tool_call|> more<|tool_call>call:f{}<tool_call|>",
                "<|channel>thought\nplan <|tool_call>call:f{a:<|\"|>x<|\"|>}<tool_call|>",
                "thought\nshort plan<channel|>Answer.",
                "thought\nlabel then a real channel<|channel>thought\nr<channel|>z",
                "some thinking<channel|>final answer",
                "x<channel|>y<|channel>thought\nr<channel|>z",
                "Hello <|channel>thought\nrumination<channel|> Goodbye.",
                "<|channel>thought\nstep one\nstep two<channel|>The answer is 42.",
                "<|channel>thought\n  padded  \n<channel|>  answer  ",
                "<|channel>a<channel|>b<|channel>c<channel|>d",
                "<|channel>thought\n<channel|>Great news!<tool_call|>",
                "thought:\nIdeas matter.",
                "thought\n",
                "th",
                "",
                "   ",
                "Answer is 5 <|",
                "Use <tool_call this literally",
                "I'll call: <|\"|>quoted<|\"|> text",
                "call:get_order_status{awb:<|\"|>1<|\"|>}",
                "<|tool_call>call:get_order_status{awb:<|\"|>1<|\"|>}<tool_call|>",
                "Sure, checking.<|tool_call>call:f{q:<|\"|>a<channel|>b<|\"|>}<tool_call|>",
                "मुझे <|channel>thought\nसोच<channel|>उत्तर",
            ]
            .map(String::from),
        );
        let long_plain = "The quick brown fox jumps over the lazy dog. ".repeat(40);
        v.push(long_plain.clone());
        v.push(format!("<|channel>thought\n{long_plain}<channel|>{long_plain}"));
        v.push(format!("thought\n{long_plain}"));
        v.push(format!("<|channel>thought\n{long_plain}"));
        v
    }

    // The fast paths are an optimization only: same output as the general path for
    // every corpus entry and chunking.
    #[test]
    fn fast_paths_match_general_path() {
        for raw in corpus() {
            for chunks in chunkings(&raw) {
                let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
                let fast = stream_with(Gemma4ReasoningParser::new(), &refs);
                let slow = stream_with(Gemma4ReasoningParser::without_fast_paths(), &refs);
                assert_eq!(fast, slow, "raw={raw:?} chunks={chunks:?}");
            }
        }
    }

    // Normalization is a pure function of the text: identical output however the
    // input is chunked (this is what makes streaming append-only).
    #[test]
    fn normalizer_is_chunking_invariant() {
        for raw in corpus() {
            let whole = OpenerLessNormalizer::normalize_all(&raw);
            for chunks in chunkings(&raw) {
                let mut n = OpenerLessNormalizer::new();
                let mut seen = String::new();
                for c in &chunks {
                    n.push(c);
                    assert!(n.output().starts_with(&seen), "output retracted: raw={raw:?}");
                    seen = n.output().to_string();
                }
                n.finish();
                assert_eq!(n.output(), whole, "raw={raw:?} chunks={chunks:?}");
            }
        }
    }

    // Streaming (any chunking) delivers what non-streaming returns, and never repeats
    // text. Whitespace at the edges may differ: non-streaming trims, streaming cannot
    // take back a space it already sent.
    #[test]
    fn streaming_matches_non_streaming() {
        // Known limitation, by design: after the first span closes, a later
        // `<|channel>` with no end yet streams as content (the observed glitch is a stray
        // opener followed by the answer, see `streaming_suppressed_empty_thinking_not_in_content`),
        // so a second span that does close later has already been sent as content.
        let second_span = "<|channel>a<channel|>b<|channel>c<channel|>d";
        for raw in corpus().into_iter().filter(|r| r != second_span) {
            let want = Gemma4ReasoningParser::new().detect_and_parse_reasoning(&raw, &[]);
            for chunks in chunkings(&raw) {
                let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
                let (normal, reasoning) = stream_with(Gemma4ReasoningParser::new(), &refs);
                let squash = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
                assert_eq!(squash(&normal), squash(&want.normal_text), "normal: raw={raw:?} chunks={chunks:?}");
                assert_eq!(squash(&reasoning), squash(&want.reasoning_text), "reasoning: raw={raw:?} chunks={chunks:?}");
            }
        }
    }

    // Review finding 1: a stray marker before a real tool call made the normalization
    // flip once the call arrived, and the stream re-sent everything before it.
    #[test]
    fn stray_marker_before_tool_call_not_duplicated() {
        let cases: [(&[&str], &str); 2] = [
            (
                &["<|\"|>", "Sure.", " ", "<|tool_call>", "call:f{a:", "<|\"|>", "x", "<|\"|>", "}", "<tool_call|>"],
                "Sure. <|tool_call>call:f{a:<|\"|>x<|\"|>}<tool_call|>",
            ),
            (
                &["Ok", "<tool_call|>", " more", "<|tool_call>", "call:f{}", "<tool_call|>"],
                "Ok more<|tool_call>call:f{}<tool_call|>",
            ),
        ];
        for (chunks, want) in cases {
            let (normal, reasoning) = stream_with(Gemma4ReasoningParser::new(), chunks);
            assert_eq!(normal, want, "{chunks:?}");
            assert_eq!(reasoning, "");
            let whole: String = chunks.concat();
            let r = Gemma4ReasoningParser::new().detect_and_parse_reasoning(&whole, &[]);
            assert_eq!(r.normal_text, want);
        }
    }

    // Review finding 2: a tool call inside an unclosed thought span is held back while
    // streaming; finish must deliver it (it was lost).
    #[test]
    fn finish_delivers_tool_call_from_unclosed_span() {
        let raw = "<|channel>thought\nplan <|tool_call>call:f{a:<|\"|>x<|\"|>}<tool_call|>";
        let call = "<|tool_call>call:f{a:<|\"|>x<|\"|>}<tool_call|>";
        for chunks in chunkings(raw) {
            let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let (normal, reasoning) = stream_with(Gemma4ReasoningParser::new(), &refs);
            assert_eq!(normal, call, "{chunks:?}");
            assert_eq!(reasoning, "plan", "{chunks:?}");
        }
    }

    // Review finding 6: only a bare `thought` line is a label; `thought:` is an answer.
    #[test]
    fn thought_colon_line_is_content() {
        let raw = "thought:\nIdeas matter.";
        assert_eq!(Gemma4ReasoningParser::new().detect_and_parse_reasoning(raw, &[]).normal_text, raw);
        assert_eq!(stream_with(Gemma4ReasoningParser::new(), &["thought", ":\n", "Ideas matter."]).0, raw);
    }

    // A stray `<channel|>` before a real span no longer eats that span's own end marker.
    #[test]
    fn stray_end_before_real_span_keeps_span() {
        let r = Gemma4ReasoningParser::new()
            .detect_and_parse_reasoning("x<channel|>y<|channel>thought\nr<channel|>z", &[]);
        assert_eq!(r.reasoning_text, "r");
        assert_eq!(r.normal_text, "xyz");
    }

    // Per-chunk cost must not grow with the response length on the plain and
    // in-thought paths. Run with `--ignored --nocapture` (release) for numbers.
    #[test]
    #[ignore]
    fn bench_streaming_cost() {
        let word = "token ";
        let after = format!("<|channel>thought\n{}<channel|>", "plan ".repeat(200));
        for (label, prefix) in [
            ("plain", ""),
            ("think", "<|channel>thought\n"),
            ("answer-after-thought", after.as_str()),
        ] {
            for n in [1_000usize, 4_000, 16_000] {
                let mut best = std::time::Duration::MAX;
                for _ in 0..5 {
                    let mut p = Gemma4ReasoningParser::new();
                    let t = std::time::Instant::now();
                    p.parse_reasoning_streaming_incremental(prefix, &[]);
                    for _ in 0..n {
                        std::hint::black_box(p.parse_reasoning_streaming_incremental(word, &[]));
                    }
                    std::hint::black_box(p.finish_reasoning_stream());
                    best = best.min(t.elapsed());
                }
                println!("{label} {n:>6} tokens: {best:?} ({:.2} us/token)", best.as_secs_f64() * 1e6 / n as f64);
            }
        }
    }
}
