// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod format;
mod parser;

pub(crate) use parser::{TOOL_CALL_END, TOOL_CALL_START};
pub use format::{
    diff_reasoning_streaming_snapshots, extract_reasoning_non_streaming,
    extract_tool_handoff_text, strip_trailing_incomplete_token, ReasoningSnapshot,
    StreamingContentCleaner,
};
pub(crate) use format::{CHANNEL_END, CHANNEL_START};
pub use parser::{
    detect_tool_call_start_gemma4, find_tool_call_end_position_gemma4,
    parse_args_object_partial, split_partial_call_prefix_gemma4,
    trim_safe_json_suffix, try_tool_call_parse_gemma4,
};
