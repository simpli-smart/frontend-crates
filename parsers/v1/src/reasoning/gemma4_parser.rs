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
    diff_reasoning_streaming_snapshots, extract_reasoning_non_streaming,
    strip_trailing_incomplete_token, ReasoningSnapshot, CHANNEL_END, CHANNEL_START,
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
}

impl Gemma4ReasoningParser {
    pub fn new() -> Self {
        Self {
            cumulative_text: String::new(),
            prev_cumulative_text: String::new(),
            last_safe_text: String::new(),
            emitted_content: String::new(),
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
        Self::snapshot_to_parser_result(&extract_reasoning_non_streaming(text))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        _token_ids: &[u32],
    ) -> ParserResult {
        self.prev_cumulative_text = self.cumulative_text.clone();
        self.cumulative_text.push_str(text);
        let safe_curr = strip_trailing_incomplete_token(&self.cumulative_text);

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
        let channel_just_opened = !self.prev_cumulative_text.contains(CHANNEL_START)
            && self.cumulative_text.contains(CHANNEL_START);
        if channel_just_opened {
            prev_snapshot.content = None;
        }

        let curr_snapshot = Self::snapshot_from_safe_text(&safe_curr);
        self.last_safe_text = safe_curr;

        let (dr, mut dc) = diff_reasoning_streaming_snapshots(&curr_snapshot, &prev_snapshot);

        let suppress_content = Self::inside_open_first_channel(&self.cumulative_text);
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

    fn finish_reasoning_stream(&mut self) -> ParserResult {
        if self.cumulative_text.is_empty() {
            return ParserResult::default();
        }

        let full = std::mem::take(&mut self.cumulative_text);
        self.prev_cumulative_text.clear();
        self.last_safe_text.clear();
        self.emitted_content.clear();

        let final_snapshot = extract_reasoning_non_streaming(&full);
        Self::snapshot_to_parser_result(&final_snapshot)
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

    #[test]
    fn detect_dangling_end_marker_passes_through_as_content() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning("some thinking<channel|>final answer", &[]);
        assert_eq!(r.reasoning_text, "");
        assert_eq!(r.normal_text, "some thinking<channel|>final answer");
    }

    #[test]
    fn detect_dangling_end_with_thought_prefix_in_content() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning("thought\nrumination<channel|>final answer", &[]);
        assert_eq!(r.reasoning_text, "");
        assert!(r.normal_text.contains("<channel|>"));
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
    fn detect_multiple_reasoning_spans_first_only() {
        let mut p = Gemma4ReasoningParser::new();
        let input =
            "<|channel>thought\nfirst<channel|> middle <|channel>thought\nsecond<channel|> done";
        let r = p.detect_and_parse_reasoning(input, &[]);
        assert_eq!(r.reasoning_text, "first");
        assert!(r.normal_text.contains("middle"));
        assert!(r.normal_text.contains("second"));
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
}
