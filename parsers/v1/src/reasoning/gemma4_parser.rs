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
    diff_reasoning_streaming_snapshots, extract_reasoning_non_streaming, normalize_opener_less,
    opener_less_undecided, strip_trailing_incomplete_token, ReasoningSnapshot, CHANNEL_END,
    CHANNEL_START,
};

#[derive(Debug, Clone)]
pub struct Gemma4ReasoningParser {
    /// Full accumulated model text for snapshot-based streaming.
    cumulative_text: String,
    /// Previous cumulative text (for channel-open detection).
    prev_cumulative_text: String,
    /// Last incomplete-marker-safe prefix of `cumulative_text`.
    last_safe_text: String,
    /// Client-visible content already emitted (avoids duplicate pre-channel text).
    emitted_content: String,
    /// Whether the start of the stream has been classified (opener-less label / empty
    /// thought block vs. ordinary output). Output is held until it is.
    start_decided: bool,
}

impl Gemma4ReasoningParser {
    pub fn new() -> Self {
        Self {
            cumulative_text: String::new(),
            prev_cumulative_text: String::new(),
            last_safe_text: String::new(),
            emitted_content: String::new(),
            start_decided: false,
        }
    }

    fn snapshot_from_safe_text(safe_text: &str) -> ReasoningSnapshot {
        extract_reasoning_non_streaming(safe_text)
    }

    /// True while the first `<|channel>` span is open (no `<channel|>` yet).
    fn inside_open_first_channel(text: &str) -> bool {
        let Some((_, after_start)) = text.split_once(CHANNEL_START) else {
            return false;
        };
        !after_start.contains(CHANNEL_END)
    }

    fn snapshot_to_parser_result(snapshot: &ReasoningSnapshot) -> ParserResult {
        ParserResult {
            normal_text: snapshot.content.clone().unwrap_or_default(),
            reasoning_text: snapshot.reasoning.clone().unwrap_or_default(),
        }
    }
}

impl Default for Gemma4ReasoningParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for Gemma4ReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, _token_ids: &[u32]) -> ParserResult {
        let text = normalize_opener_less(text);
        Self::snapshot_to_parser_result(&extract_reasoning_non_streaming(&text))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        _token_ids: &[u32],
    ) -> ParserResult {
        self.prev_cumulative_text = self.cumulative_text.clone();
        self.cumulative_text.push_str(text);
        // Hold output until we know whether it opens with an opener-less role label or
        // empty thought block (see `normalize_opener_less`); decided once per stream.
        if !self.start_decided {
            if opener_less_undecided(&strip_trailing_incomplete_token(&self.cumulative_text)) {
                return ParserResult::default();
            }
            self.start_decided = true;
        }
        let norm_curr = normalize_opener_less(&self.cumulative_text).into_owned();
        let norm_prev = normalize_opener_less(&self.prev_cumulative_text).into_owned();
        let safe_curr = strip_trailing_incomplete_token(&norm_curr);

        if safe_curr == self.last_safe_text {
            return ParserResult::default();
        }

        let safe_prev = if self.last_safe_text.is_empty() {
            String::new()
        } else {
            self.last_safe_text.clone()
        };

        let mut prev_snapshot = if safe_prev.is_empty() {
            ReasoningSnapshot::default()
        } else {
            Self::snapshot_from_safe_text(&safe_prev)
        };

        // Pre-channel text must not be emitted twice once the channel opens.
        let channel_just_opened =
            !norm_prev.contains(CHANNEL_START) && norm_curr.contains(CHANNEL_START);
        if channel_just_opened {
            prev_snapshot.content = None;
        }

        let curr_snapshot = Self::snapshot_from_safe_text(&safe_curr);
        self.last_safe_text = safe_curr;

        let (dr, mut dc) = diff_reasoning_streaming_snapshots(&curr_snapshot, &prev_snapshot);

        let suppress_content = Self::inside_open_first_channel(&norm_curr);
        if suppress_content {
            dc.clear();
        } else if let Some(ref full) = curr_snapshot.content {
            if !self.emitted_content.is_empty() && full.starts_with(&self.emitted_content) {
                dc = full[self.emitted_content.len()..].to_string();
            }
            if !dc.is_empty() {
                self.emitted_content = full.clone();
            }
        }

        ParserResult {
            normal_text: dc,
            reasoning_text: dr,
        }
    }

    /// Flush what the stream has not emitted yet (the trait contract: only the
    /// remainder, like every other parser). Covers output still held as a possible
    /// role label and the tail kept back as an incomplete marker.
    fn finish_reasoning_stream(&mut self) -> ParserResult {
        if self.cumulative_text.is_empty() {
            return ParserResult::default();
        }

        let full = std::mem::take(&mut self.cumulative_text);
        let last_safe = std::mem::take(&mut self.last_safe_text);
        let emitted = std::mem::take(&mut self.emitted_content);
        self.prev_cumulative_text.clear();
        self.start_decided = false;

        let final_snapshot = extract_reasoning_non_streaming(&normalize_opener_less(&full));
        let prev_snapshot = if last_safe.is_empty() {
            ReasoningSnapshot::default()
        } else {
            Self::snapshot_from_safe_text(&last_safe)
        };
        let (reasoning_text, mut normal_text) =
            diff_reasoning_streaming_snapshots(&final_snapshot, &prev_snapshot);
        if let Some(ref content) = final_snapshot.content
            && !emitted.is_empty()
            && content.starts_with(&emitted)
        {
            normal_text = content[emitted.len()..].to_string();
        }
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

}
