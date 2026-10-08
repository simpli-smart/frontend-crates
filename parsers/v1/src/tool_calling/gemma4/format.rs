// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Content-cleaning helpers ported from vLLM's gemma4_format.py (SGLang-aligned).

use super::parser::{CALL_PREFIX, STRING_DELIM, TOOL_CALL_END, TOOL_CALL_START};

pub(crate) const CHANNEL_START: &str = "<|channel>";
pub(crate) const CHANNEL_END: &str = "<channel|>";
pub(crate) const THOUGHT_PREFIX: &str = "thought\n";

const TH_WORD: &str = "thought";
const LEN_TH: usize = TH_WORD.len();
const TH_PAIR: &str = "thoughtthought";
const LEN_PAIR: usize = TH_PAIR.len();

const EMPTY_THINKING_PATTERNS: &[&str] = &[
    "<|channel>thought\n<channel|>",
    "<|channel>thought\r\n<channel|>",
];

/// Orphan role labels the model sometimes emits with ``enable_thinking: false``.
const ORPHAN_THOUGHT_LABEL_PREFIXES: &[&str] = &[
    "__thought\n",
    "__thought\r\n",
    "__thought__\n",
    "__thought__\r\n",
];

/// Control tokens whose **proper prefixes** may appear as streaming tail fragments.
const INCOMPLETE_TOKENS: &[&str] = &[
    CHANNEL_START,
    CHANNEL_END,
    "<|channel>thought\n",   // CHANNEL_START + THOUGHT_PREFIX
    "<|channel>thought\r\n", // \r\n variant (Dynamo empty-thinking pattern)
    TOOL_CALL_START,
    TOOL_CALL_END,
    STRING_DELIM,
];

/// Longest proper-prefix suffix of known control tokens at the end of `text`.
fn longest_incomplete_suffix_len(text: &str) -> usize {
    let mut longest = 0usize;
    for tok in INCOMPLETE_TOKENS {
        // Exclude `i == tok.len()` so complete tokens (e.g. `<|"|>`) are never stripped.
        for i in 1..tok.len() {
            if text.ends_with(&tok[..i]) && i > longest {
                longest = i;
            }
        }
    }
    longest
}

/// Snapshot of reasoning vs visible content at a point in the stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReasoningSnapshot {
    pub reasoning: Option<String>,
    pub content: Option<String>,
}

/// Stateful prefix-diff cleaner for pre-tool streaming content.
#[derive(Debug, Clone, Default)]
pub struct StreamingContentCleaner {
    clean_content_prefix: String,
    clean_content_previous_text: Option<String>,
}

impl StreamingContentCleaner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Visible text delta before the first tool call, diffed on stripped prefixes.
    pub fn pre_tool_content_delta(
        &mut self,
        previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> Option<String> {
        let same_stream = self
            .clean_content_previous_text
            .as_deref()
            .is_some_and(|prev| prev == previous_text);

        let clean_prev = if same_stream && !previous_text.is_empty() {
            self.clean_content_prefix.clone()
        } else {
            strip_tool_call_suffix(&strip_leaked_empty_thinking(previous_text))
        };

        let clean_curr =
            strip_tool_call_suffix(&strip_leaked_empty_thinking(current_text));
        let clean_prev = strip_tool_call_suffix(&clean_prev);
        self.clean_content_prefix = clean_curr.clone();
        self.clean_content_previous_text = Some(current_text.to_string());

        if clean_curr.starts_with(&clean_prev) {
            let out = clean_curr[clean_prev.len()..].to_string();
            return if out.is_empty() { None } else { Some(out) };
        }

        let cleaned = strip_tool_call_suffix(&strip_leaked_empty_thinking(delta_text));
        if cleaned.is_empty() {
            None
        } else {
            Some(cleaned)
        }
    }
}

/// Index where Gemma4 tool-call markup begins, or `None`.
pub(crate) fn tool_call_markup_start(text: &str) -> Option<usize> {
    if let Some(i) = text.find(TOOL_CALL_START) {
        return Some(i);
    }
    let search_from = text
        .rfind(CHANNEL_END)
        .map(|i| i + CHANNEL_END.len())
        .unwrap_or(0);
    let slice_text = &text[search_from..];
    let call_rel = slice_text.find(CALL_PREFIX)?;
    let call_abs = search_from + call_rel;
    text[call_abs + CALL_PREFIX.len()..]
        .find('{')
        .map(|_| call_abs)
}

/// True if *text* contains Gemma4 tool-call delimiters or a bare `call:fn{` block.
pub(crate) fn has_tool_call_markup(text: &str) -> bool {
    text.contains(TOOL_CALL_START)
        || text.contains(TOOL_CALL_END)
        || text.contains(STRING_DELIM)
        || tool_call_markup_start(text).is_some()
}

/// Remove trailing tool-call markup so it is not emitted as client `content`.
pub(crate) fn strip_tool_call_suffix(text: &str) -> String {
    if let Some(idx) = tool_call_markup_start(text) {
        text[..idx].trim_end().to_string()
    } else {
        text.trim_end().to_string()
    }
}

/// Text from the first tool-call marker onward (reasoning→tool handoff).
pub fn extract_tool_handoff_text(text: &str) -> String {
    tool_call_markup_start(text)
        .map(|idx| text[idx..].to_string())
        .unwrap_or_default()
}

fn compact_cf_no_ws(core: &str) -> String {
    core.to_lowercase().split_whitespace().collect()
}

fn strip_one_thought_shard_line(core: &str, nl: &str) -> String {
    let compact = compact_cf_no_ws(core);
    if compact.is_empty() {
        return format!("{core}{nl}");
    }

    let mut i = 0usize;
    let limit = compact.len();
    while i + LEN_PAIR <= limit && compact[i..].starts_with(TH_PAIR) {
        i += LEN_PAIR;
    }
    let c = &compact[i..];
    let lc = c.len();

    if lc == 0 {
        return nl.to_string();
    }
    if c == TH_WORD {
        return nl.to_string();
    }
    if lc < LEN_TH {
        return if TH_WORD.starts_with(c) {
            nl.to_string()
        } else {
            format!("{core}{nl}")
        };
    }
    if lc <= 5 && TH_WORD.ends_with(c) && c != TH_WORD {
        return nl.to_string();
    }

    if lc > LEN_TH && c.starts_with(TH_WORD) {
        let tail = &c[LEN_TH..];
        if !tail.is_empty() && tail != TH_WORD && TH_WORD.starts_with(tail) {
            return nl.to_string();
        }
    }

    format!("{core}{nl}")
}

/// Strip glued `thought` shards from plain `content` (streaming cut artifacts).
pub(crate) fn strip_thought_shard_echoes(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let cf_full = text.to_lowercase();
    if !cf_full.contains("thought") {
        return text.to_string();
    }

    if !text.contains('\n') && !text.contains('\r') {
        return strip_one_thought_shard_line(text, "");
    }

    let mut rebuilt = String::new();
    for raw_line in text.split_inclusive('\n') {
        let (core, nl) = if let Some(stripped) = raw_line.strip_suffix('\n') {
            (stripped, "\n")
        } else {
            (raw_line, "")
        };
        rebuilt.push_str(&strip_one_thought_shard_line(core, nl));
    }
    rebuilt
}

/// True when *text* may contain Gemma4 control tokens that must not reach clients.
fn may_contain_gemma4_control_leak(text: &str) -> bool {
    text.contains(CHANNEL_START)
        || text.contains(CHANNEL_END)
        || text.to_lowercase().contains("thought")
        || text.contains(TOOL_CALL_START)
        || text.contains(TOOL_CALL_END)
        || text.contains(STRING_DELIM)
}

fn may_need_content_cleaning(text: &str) -> bool {
    may_contain_gemma4_control_leak(text)
        || text.contains("__thought")
        || text.lines().any(is_underscore_separator_line)
}

fn is_underscore_separator_line(line: &str) -> bool {
    let stripped = line.trim();
    stripped.len() >= 3 && stripped.chars().all(|c| c == '_')
}

/// Remove ``thought\\n<|channel>thought\\n<channel|>`` only at line boundaries.
fn strip_composite_empty_thinking(text: &str) -> String {
    let composite = format!("{THOUGHT_PREFIX}{CHANNEL_START}{THOUGHT_PREFIX}{CHANNEL_END}");
    let mut s = text.to_string();
    loop {
        if s.starts_with(&composite) {
            s = s[composite.len()..].to_string();
            continue;
        }
        let needle = format!("\n{composite}");
        if let Some(idx) = s.find(&needle) {
            s = format!("{}{}", &s[..idx], &s[idx + 1..]);
            continue;
        }
        break;
    }
    s
}

/// Remove echoed ``__thought`` role labels (markdown-style channel corruption).
fn strip_orphan_thought_labels(text: &str) -> String {
    let mut s = text.to_string();
    loop {
        let old = s.clone();
        for prefix in ORPHAN_THOUGHT_LABEL_PREFIXES {
            if s.starts_with(prefix) {
                s = s[prefix.len()..].to_string();
            }
        }
        if let Some(rest) = s.strip_prefix("__thought") {
            let rest = rest.strip_prefix("__").unwrap_or(rest);
            if rest.is_empty() || rest.starts_with('\n') || rest.starts_with("\r\n") {
                s = rest
                    .trim_start_matches('\r')
                    .trim_start_matches('\n')
                    .to_string();
            }
        }
        if s == old {
            break;
        }
    }
    s
}

/// Remove planning separator lines made of underscores only.
fn strip_underscore_separator_lines(text: &str) -> String {
    if !text.contains('_') {
        return text.to_string();
    }
    let mut out = String::new();
    for raw_line in text.split_inclusive('\n') {
        let (core, nl) = if let Some(stripped) = raw_line.strip_suffix('\n') {
            (stripped, "\n")
        } else {
            (raw_line, "")
        };
        if is_underscore_separator_line(core) {
            continue;
        }
        out.push_str(core);
        out.push_str(nl);
    }
    out.trim_start_matches('\n').to_string()
}

/// Remove orphaned Gemma4 tool-call grammar tokens from client-visible text.
///
/// Does not strip ``<|tool_call>`` — that marker is handled by
/// ``strip_tool_call_suffix`` on the full accumulated buffer.
fn strip_leaked_tool_grammar(text: &str) -> String {
    let mut s = text.to_string();
    for tok in [TOOL_CALL_END, STRING_DELIM] {
        if s.contains(tok) {
            s = s.replace(tok, "");
        }
    }
    s
}

/// Remove echoed empty thinking channels and orphan control-token prefixes.
pub(crate) fn strip_leaked_empty_thinking(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    if !may_need_content_cleaning(text) {
        return text.to_string();
    }

    let mut s = text.to_string();

    s = strip_orphan_thought_labels(&s);
    s = strip_composite_empty_thinking(&s);

    for pattern in EMPTY_THINKING_PATTERNS {
        if s.contains(pattern) {
            s = s.replace(pattern, "");
        }
    }

    while s.starts_with(THOUGHT_PREFIX)
        && (s[THOUGHT_PREFIX.len()..].contains(CHANNEL_START)
            || s[THOUGHT_PREFIX.len()..].contains(CHANNEL_END))
    {
        s = s[THOUGHT_PREFIX.len()..].to_string();
    }

    let orphan_close = format!("{THOUGHT_PREFIX}{CHANNEL_END}");
    if s.starts_with(&orphan_close) {
        s = s[orphan_close.len()..].to_string();
    }

    if s.contains(CHANNEL_START) || s.contains(CHANNEL_END) {
        loop {
            let old = s.clone();
            s = s.replace(CHANNEL_START, "").replace(CHANNEL_END, "");
            s = s.trim().to_string();
            if s == old {
                break;
            }
        }
    }

    let s = strip_leaked_tool_grammar(&s);
    let s = strip_thought_shard_echoes(&s);
    let s = strip_orphan_thought_labels(&s);
    let s = strip_underscore_separator_lines(&s);
    s.trim_start_matches('\n').trim().to_string()
}

/// Trim suffix that may be an incomplete Gemma4 marker (streaming-safe).
pub fn strip_trailing_incomplete_token(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let strip_len = longest_incomplete_suffix_len(text);
    if strip_len == 0 {
        text.to_string()
    } else if text.is_char_boundary(text.len() - strip_len) {
        text[..text.len() - strip_len].to_string()
    } else {
        text.to_string()
    }
}

fn norm_snapshot_part(x: Option<&str>) -> String {
    x.unwrap_or("").to_string()
}

/// Return `(reasoning_delta, content_delta)` from two parse snapshots.
pub fn diff_reasoning_streaming_snapshots(
    curr: &ReasoningSnapshot,
    prev: &ReasoningSnapshot,
) -> (String, String) {
    let nr = norm_snapshot_part(curr.reasoning.as_deref());
    let nc = norm_snapshot_part(curr.content.as_deref());
    let nr_prev = norm_snapshot_part(prev.reasoning.as_deref());
    let nc_prev = norm_snapshot_part(prev.content.as_deref());

    let dr = if nr.starts_with(&nr_prev) {
        nr[nr_prev.len()..].to_string()
    } else {
        nr
    };
    let dc = if nc.starts_with(&nc_prev) {
        nc[nc_prev.len()..].to_string()
    } else {
        nc
    };
    (dr, dc)
}

/// Strip `thought\n` from the start of reasoning body text.
pub fn strip_thought_label(text: &str) -> &str {
    text.strip_prefix(THOUGHT_PREFIX).unwrap_or(text)
}

/// Split Gemma4 assistant output into reasoning and visible content (SGLang-aligned).
pub fn extract_reasoning_non_streaming(model_output: &str) -> ReasoningSnapshot {
    if !model_output.contains(CHANNEL_START) {
        return ReasoningSnapshot {
            reasoning: None,
            content: if model_output.is_empty() {
                None
            } else {
                Some(model_output.to_string())
            },
        };
    }

    let Some((before_channel, after_channel)) = model_output.split_once(CHANNEL_START) else {
        return ReasoningSnapshot {
            reasoning: None,
            content: finalize_client_content(model_output),
        };
    };

    let mut rest = after_channel;
    if let Some(stripped) = rest.strip_prefix(THOUGHT_PREFIX) {
        rest = stripped;
    }

    if !rest.contains(CHANNEL_END) {
        if let Some(tool_idx) = rest.find(TOOL_CALL_START) {
            let reasoning_text = rest[..tool_idx].trim();
            let normal_text = format!("{}{}", before_channel, &rest[tool_idx..]).trim().to_string();
            return ReasoningSnapshot {
                reasoning: if reasoning_text.is_empty() {
                    None
                } else {
                    Some(reasoning_text.to_string())
                },
                content: if normal_text.is_empty() {
                    None
                } else if has_tool_call_markup(&normal_text) {
                    Some(normal_text)
                } else {
                    finalize_client_content(&normal_text)
                },
            };
        }
        let reasoning_text = rest.trim();
        return ReasoningSnapshot {
            reasoning: if reasoning_text.is_empty() {
                None
            } else {
                Some(reasoning_text.to_string())
            },
            content: None,
        };
    }

    let Some((reason_part, content_part)) = rest.split_once(CHANNEL_END) else {
        let reasoning_text = rest.trim();
        return ReasoningSnapshot {
            reasoning: if reasoning_text.is_empty() {
                None
            } else {
                Some(reasoning_text.to_string())
            },
            content: None,
        };
    };

    let reasoning_text = reason_part.trim();
    let merged_content = format!("{before_channel}{content_part}");
    let content = if content_part.is_empty() {
        // Channel closed but no post-marker text yet — hold content until more arrives.
        None
    } else {
        let merged_trimmed = merged_content.trim();
        if merged_trimmed.is_empty() {
            None
        } else if has_tool_call_markup(merged_trimmed) {
            Some(merged_trimmed.to_string())
        } else {
            finalize_client_content(merged_trimmed)
        }
    };
    ReasoningSnapshot {
        reasoning: Some(reasoning_text.to_string()),
        content,
    }
}

/// Client-visible content pipeline: empty-thinking suppression + tool-suffix strip.
pub(crate) fn finalize_client_content(text: &str) -> Option<String> {
    let after = strip_leaked_empty_thinking(text);
    if after.is_empty() {
        return None;
    }
    let after = strip_tool_call_suffix(&after);
    if after.is_empty() {
        None
    } else {
        Some(after)
    }
}

/// Clean prose prefix without stripping downstream tool-call markup.
pub(crate) fn clean_visible_prefix(text: &str) -> String {
    strip_leaked_empty_thinking(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_trailing_incomplete_token_preserves_complete_markers() {
        // `<|channel>` alone is a proper prefix of `<|channel>thought\n` and is stripped.
        assert_eq!(strip_trailing_incomplete_token("<|channel>"), "");
        assert_eq!(strip_trailing_incomplete_token("<channel|>"), "<channel|>");
        assert_eq!(strip_trailing_incomplete_token("<|tool_call>"), "<|tool_call>");
        assert_eq!(strip_trailing_incomplete_token("<tool_call|>"), "<tool_call|>");
        assert_eq!(strip_trailing_incomplete_token(STRING_DELIM), STRING_DELIM);
        assert_eq!(
            strip_trailing_incomplete_token("{location:<|\"|>NYC<|\"|>}"),
            "{location:<|\"|>NYC<|\"|>}"
        );
    }

    #[test]
    fn strip_trailing_incomplete_token_strips_partial_markers() {
        assert_eq!(strip_trailing_incomplete_token("Hi<|chan"), "Hi");
        assert_eq!(strip_trailing_incomplete_token("Hi<|\"|"), "Hi");
        assert_eq!(strip_trailing_incomplete_token("partial<tho"), "partial<tho");
        assert_eq!(
            strip_trailing_incomplete_token("<|channel>thought\n"),
            "<|channel>thought\n"
        );
    }

    #[test]
    fn strip_leaked_empty_thinking_preserves_tool_call_start_for_suffix_strip() {
        let input = "I'll help!<|tool_call>call:f{}<tool_call|>";
        assert_eq!(
            strip_tool_call_suffix(&strip_leaked_empty_thinking(input)),
            "I'll help!"
        );
    }

    #[test]
    fn strip_leaked_empty_thinking_removes_tool_grammar_leaks() {
        let banking = "<|\"|>Great news! You are eligible for a personal loan.";
        assert_eq!(
            strip_leaked_empty_thinking(banking),
            "Great news! You are eligible for a personal loan."
        );

        assert_eq!(strip_leaked_empty_thinking("thought<tool_call|>"), "");
        assert_eq!(strip_leaked_empty_thinking("<tool_call|>"), "");
        assert_eq!(strip_leaked_empty_thinking("<|\"|>"), "");
    }

    #[test]
    fn strip_leaked_empty_thinking_removes_echoed_channel() {
        let input = "Hello <|channel>thought\n<channel|>world";
        assert_eq!(strip_leaked_empty_thinking(input), "Hello world");

        let composite = "thought\n<|channel>thought\n<channel|>أعتذر جداً";
        assert_eq!(strip_leaked_empty_thinking(composite), "أعتذر جداً");

        let orphan = "thought\n<channel|>أعتذر";
        assert_eq!(strip_leaked_empty_thinking(orphan), "أعتذر");

        let arabic = "أعتذر جداً عن التأخير!";
        assert_eq!(strip_leaked_empty_thinking(arabic), arabic);

        let orphan_label = "__thought\nAll getting sometime this";
        assert_eq!(
            strip_leaked_empty_thinking(orphan_label),
            "All getting sometime this"
        );

        let composite_label =
            "__thought\n<|channel>thought\n<channel|>You're welcome.";
        assert_eq!(
            strip_leaked_empty_thinking(composite_label),
            "You're welcome."
        );

        let separators = "_________________\nHello there";
        assert_eq!(strip_leaked_empty_thinking(separators), "Hello there");

        let thought_then_sep = "thought\n_________________\nHi";
        assert_eq!(strip_leaked_empty_thinking(thought_then_sep), "Hi");
    }

    #[test]
    fn strip_thought_shard_echoes_removes_glued_shards() {
        assert_eq!(strip_thought_shard_echoes("thoughtthought"), "");
        assert_eq!(
            strip_thought_shard_echoes("thoughtthoughtful"),
            "thoughtthoughtful"
        );
        assert_eq!(strip_thought_shard_echoes("thoughtthoughtHello"), "thoughtthoughtHello");
        let core = "thought".repeat(11) + "tho";
        assert_eq!(strip_thought_shard_echoes(&core).trim(), "");
    }

    #[test]
    fn strip_tool_call_suffix_keeps_prefix() {
        let input = "I'll help!<|tool_call>call:f{}<tool_call|>";
        assert_eq!(strip_tool_call_suffix(input), "I'll help!");
    }

    #[test]
    fn finalize_client_content_strips_empty_thinking_and_tool_suffix() {
        let input = "Hi <|channel>thought\n<channel|>there<|tool_call>call:f{}<tool_call|>";
        assert_eq!(finalize_client_content(input).as_deref(), Some("Hi there"));
    }

    #[test]
    fn extract_reasoning_non_streaming_basic() {
        let snap = extract_reasoning_non_streaming(
            "<|channel>thought\nstep one\nstep two<channel|>The answer is 42.",
        );
        assert_eq!(snap.reasoning.as_deref(), Some("step one\nstep two"));
        assert_eq!(snap.content.as_deref(), Some("The answer is 42."));
    }

    #[test]
    fn extract_reasoning_non_streaming_no_channel() {
        let snap = extract_reasoning_non_streaming("just a plain answer");
        assert!(snap.reasoning.is_none());
        assert_eq!(snap.content.as_deref(), Some("just a plain answer"));
    }

    #[test]
    fn extract_reasoning_non_streaming_dangling_end() {
        let snap = extract_reasoning_non_streaming("some thinking<channel|>final answer");
        assert!(snap.reasoning.is_none());
        assert_eq!(
            snap.content.as_deref(),
            Some("some thinking<channel|>final answer")
        );
    }

    #[test]
    fn diff_reasoning_streaming_snapshots_basic() {
        let prev = ReasoningSnapshot {
            reasoning: Some("step".to_string()),
            content: None,
        };
        let curr = ReasoningSnapshot {
            reasoning: Some("step one".to_string()),
            content: None,
        };
        let (dr, dc) = diff_reasoning_streaming_snapshots(&curr, &prev);
        assert_eq!(dr, " one");
        assert_eq!(dc, "");
    }

    #[test]
    fn streaming_content_cleaner_prefix_diff() {
        let mut cleaner = StreamingContentCleaner::new();
        let d1 = cleaner.pre_tool_content_delta("", "<|channel>", "<|channel>");
        assert!(d1.is_none());

        let d2 = cleaner.pre_tool_content_delta(
            "<|channel>",
            "<|channel>thought\n<channel|>Hi.",
            "thought\n<channel|>Hi.",
        );
        assert_eq!(d2.as_deref(), Some("Hi."));
    }
}
