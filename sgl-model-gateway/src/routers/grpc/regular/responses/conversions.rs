//! Conversion utilities for translating between /v1/responses and /v1/chat/completions formats
//!
//! This module implements the conversion approach where:
//! 1. ResponsesRequest → ChatCompletionRequest (for backend processing)
//! 2. ChatCompletionResponse → ResponsesResponse (for client response)
//!
//! This allows the gRPC router to reuse the existing chat pipeline infrastructure
//! without requiring Python backend changes.

use crate::{
    protocols::{
        chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent},
        common::{
            ContentPart, FunctionCallResponse, ImageUrl, JsonSchemaFormat, ResponseFormat,
            StreamOptions, ToolCall, UsageInfo,
        },
        responses::{
            FunctionCallOutputContent, ResponseContentPart, ResponseInput,
            ResponseInputOutputItem, ResponseOutputItem,
            ResponseReasoningContent::ReasoningText, ResponseStatus, ResponsesRequest,
            ResponsesResponse, ResponsesUsage, StringOrContentParts, TextConfig, TextFormat,
        },
        UNKNOWN_MODEL_ID,
    },
    routers::grpc::common::responses::utils::extract_tools_from_response_tools,
};

/// Convert a ResponsesRequest to ChatCompletionRequest for processing through the chat pipeline
///
/// # Conversion Logic
/// - `input` (text/items) → `messages` (chat messages)
/// - `instructions` → system message (prepended)
/// - `max_output_tokens` → `max_completion_tokens`
/// - `tools` → function tools extracted from ResponseTools
/// - `tool_choice` → passed through from request
/// - Response-specific fields (previous_response_id, conversation) are handled by router
pub(crate) fn responses_to_chat(req: &ResponsesRequest) -> Result<ChatCompletionRequest, String> {
    let mut messages = Vec::new();

    // 1. Add system message if instructions provided
    if let Some(instructions) = &req.instructions {
        messages.push(ChatMessage::System {
            content: MessageContent::Text(instructions.clone()),
            name: None,
        });
    }

    // 2. Convert input to chat messages
    match &req.input {
        ResponseInput::Text(text) => {
            // Simple text input → user message
            messages.push(ChatMessage::User {
                content: MessageContent::Text(text.clone()),
                name: None,
            });
        }
        ResponseInput::Items(items) => {
            // Structured items → convert each to appropriate chat message
            for item in items {
                match item {
                    ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
                        // Convert SimpleInputMessage to chat message. Images are
                        // preserved (not just InputText) via `convert_content_parts`.
                        let message_content = match content {
                            StringOrContentParts::String(s) => MessageContent::Text(s.clone()),
                            StringOrContentParts::Array(parts) => convert_content_parts(parts),
                        };

                        messages.push(role_to_chat_message(role.as_str(), message_content));
                    }
                    ResponseInputOutputItem::Message { role, content, .. } => {
                        // Extract content parts, preserving images alongside text.
                        let message_content = convert_content_parts(content);

                        messages.push(role_to_chat_message(role.as_str(), message_content));
                    }
                    ResponseInputOutputItem::FunctionToolCall {
                        id: _,
                        call_id,
                        name,
                        arguments,
                        output,
                        ..
                    } => {
                        // Tool call from history - add as assistant message with tool call
                        // followed by tool response if output exists
                        //
                        // Use `call_id` (always present) rather than the optional
                        // `id` here: `call_id` is the field that correlates this
                        // function_call with a *separate* `function_call_output`
                        // item (see the `ResponseInputOutputItem::FunctionCallOutput`
                        // branch below, which keys `tool_call_id` off `call_id`).
                        // `id` is a distinct, unrelated identifier (e.g. `fc_...`
                        // vs `call_...`) and using it here would silently break
                        // that correlation. This mirrors the Python reference
                        // implementation's `call_id or id` precedence in
                        // `serving_responses.py::_normalize_response_message_for_chat`.
                        let effective_id = call_id.clone();

                        // Add assistant message with tool_calls (the LLM's decision)
                        messages.push(ChatMessage::Assistant {
                            content: None,
                            name: None,
                            tool_calls: Some(vec![ToolCall {
                                id: effective_id.clone(),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name: name.clone(),
                                    arguments: Some(arguments.clone()),
                                },
                            }]),
                            reasoning_content: None,
                        });

                        // Add tool result message if output exists
                        if let Some(output_text) = output {
                            messages.push(ChatMessage::Tool {
                                content: MessageContent::Text(output_text.clone()),
                                tool_call_id: effective_id.clone(),
                            });
                        }
                    }
                    ResponseInputOutputItem::Reasoning { content, .. } => {
                        // Reasoning content - add as assistant message with reasoning_content
                        let reasoning_text = content
                            .iter()
                            .map(|c| match c {
                                ReasoningText { text } => text.as_str(),
                            })
                            .collect::<Vec<_>>()
                            .join("\n");

                        messages.push(ChatMessage::Assistant {
                            content: None,
                            name: None,
                            tool_calls: None,
                            reasoning_content: Some(reasoning_text),
                        });
                    }
                    ResponseInputOutputItem::FunctionCallOutput {
                        call_id, output, ..
                    } => {
                        // Function call output - add as tool message. Images
                        // (e.g. from a `view_image`-style tool) are preserved
                        // via `function_call_output_to_message_content`
                        // instead of being flattened to text and dropped.
                        // Note: The function name is looked up from prev_outputs in Harmony path
                        // For Chat path, we just use the call_id
                        messages.push(ChatMessage::Tool {
                            content: function_call_output_to_message_content(output),
                            tool_call_id: call_id.clone(),
                        });
                    }
                }
            }
        }
    }

    // Ensure we have at least one message
    if messages.is_empty() {
        return Err("Request must contain at least one message".to_string());
    }

    // 3. Extract function tools from ResponseTools
    // Only function tools are extracted here (include_mcp: false).
    // MCP tools are merged later by the tool loop (see tool_loop.rs:prepare_chat_tools_and_choice)
    // before the chat pipeline, where tool_choice constraints are applied to ALL tools combined.
    let function_tools = extract_tools_from_response_tools(req.tools.as_deref(), false);
    let tools = if function_tools.is_empty() {
        None
    } else {
        Some(function_tools)
    };

    // 4. Build ChatCompletionRequest
    let is_streaming = req.stream.unwrap_or(false);

    Ok(ChatCompletionRequest {
        messages,
        model: if req.model.is_empty() {
            UNKNOWN_MODEL_ID.to_string()
        } else {
            req.model.clone()
        },
        temperature: req.temperature,
        max_completion_tokens: req.max_output_tokens,
        stream: is_streaming,
        stream_options: if is_streaming {
            Some(StreamOptions {
                include_usage: Some(true),
            })
        } else {
            None
        },
        parallel_tool_calls: req.parallel_tool_calls,
        top_logprobs: req.top_logprobs,
        top_p: req.top_p,
        skip_special_tokens: true,
        tools,
        tool_choice: req.tool_choice.clone(),
        response_format: map_text_to_response_format(&req.text),
        ..Default::default()
    })
}

/// Extract text content from ResponseContentPart array
fn extract_text_from_content(content: &[ResponseContentPart]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            ResponseContentPart::InputText { text } => Some(text.as_str()),
            ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Convert a single Responses-API content part to a Chat Completions content
/// part. Returns `None` for parts with no Chat Completions equivalent
/// (currently just `Unknown`).
fn response_content_part_to_chat(part: &ResponseContentPart) -> Option<ContentPart> {
    match part {
        ResponseContentPart::InputText { text } | ResponseContentPart::OutputText { text, .. } => {
            Some(ContentPart::Text { text: text.clone() })
        }
        ResponseContentPart::InputImage { image_url, detail } => Some(ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: image_url.url().to_string(),
                detail: image_url
                    .detail()
                    .map(str::to_string)
                    .or_else(|| detail.clone())
                    .or_else(|| Some("auto".to_string())),
            },
        }),
        ResponseContentPart::Unknown => None,
    }
}

/// Convert a Responses-API content-part array to Chat Completions
/// `MessageContent`, preserving images instead of dropping them.
///
/// When the parts are all text (the common case), this collapses to a plain
/// `MessageContent::Text` — identical to the previous text-only behavior —
/// so pure-text conversations are unaffected. `MessageContent::Parts` is
/// only used once at least one image is present.
fn convert_content_parts(parts: &[ResponseContentPart]) -> MessageContent {
    let has_image = parts
        .iter()
        .any(|part| matches!(part, ResponseContentPart::InputImage { .. }));
    if !has_image {
        return MessageContent::Text(extract_text_from_content(parts));
    }
    MessageContent::Parts(
        parts
            .iter()
            .filter_map(response_content_part_to_chat)
            .collect(),
    )
}

/// Convert a `function_call_output.output` value to Chat Completions
/// `MessageContent`. Mirrors `convert_content_parts`: text-only outputs stay
/// a plain string, outputs containing images become `MessageContent::Parts`
/// so tools like `view_image` can round-trip screenshots to the model.
fn function_call_output_to_message_content(output: &FunctionCallOutputContent) -> MessageContent {
    match output {
        FunctionCallOutputContent::Text(s) => MessageContent::Text(s.clone()),
        FunctionCallOutputContent::Parts(parts) => convert_content_parts(parts),
    }
}

/// Convert role and content to ChatMessage
fn role_to_chat_message(role: &str, content: MessageContent) -> ChatMessage {
    match role {
        "user" => ChatMessage::User { content, name: None },
        "assistant" => ChatMessage::Assistant {
            content: Some(content),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        },
        "system" => ChatMessage::System { content, name: None },
        _ => {
            // Unknown role, treat as user message
            ChatMessage::User { content, name: None }
        }
    }
}

/// Map TextConfig from Responses API to ResponseFormat for Chat API
///
/// Converts the structured output configuration from the Responses API format
/// to the Chat API format for non-Harmony models.
fn map_text_to_response_format(text: &Option<TextConfig>) -> Option<ResponseFormat> {
    let text_config = text.as_ref()?;
    let format = text_config.format.as_ref()?;

    match format {
        TextFormat::Text => Some(ResponseFormat::Text),
        TextFormat::JsonObject => Some(ResponseFormat::JsonObject),
        TextFormat::JsonSchema {
            name,
            schema,
            description: _,
            strict,
        } => Some(ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name: name.clone(),
                schema: schema.clone(),
                strict: *strict,
            },
        }),
    }
}

/// Convert a ChatCompletionResponse to ResponsesResponse
///
/// # Conversion Logic
/// - `id` → `response_id_override` if provided, otherwise `chat_resp.id`
/// - `model` → `model` (pass through)
/// - `choices[0].message` → `output` array (convert to ResponseOutputItem::Message)
/// - `choices[0].finish_reason` → determines `status` (stop/length → Completed)
/// - `created` timestamp → `created_at`
pub(crate) fn chat_to_responses(
    chat_resp: &ChatCompletionResponse,
    original_req: &ResponsesRequest,
    response_id_override: Option<String>,
) -> Result<ResponsesResponse, String> {
    // Extract the first choice (responses API doesn't support n>1)
    let choice = chat_resp
        .choices
        .first()
        .ok_or_else(|| "Chat response contains no choices".to_string())?;

    // Convert assistant message to output items
    let mut output: Vec<ResponseOutputItem> = Vec::new();

    // Convert message content to output item
    if let Some(content) = &choice.message.content {
        if !content.is_empty() {
            output.push(ResponseOutputItem::Message {
                id: format!("msg_{}", chat_resp.id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: content.clone(),
                    annotations: vec![],
                    logprobs: choice.logprobs.clone(),
                }],
                status: "completed".to_string(),
            });
        }
    }

    // Convert reasoning content if present (O1-style models)
    if let Some(reasoning) = &choice.message.reasoning_content {
        if !reasoning.is_empty() {
            output.push(ResponseOutputItem::Reasoning {
                id: format!("reasoning_{}", chat_resp.id),
                summary: vec![],
                content: vec![ReasoningText {
                    text: reasoning.clone(),
                }],
                status: Some("completed".to_string()),
            });
        }
    }

    // Convert tool calls if present
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            output.push(ResponseOutputItem::FunctionToolCall {
                id: tool_call.id.clone(),
                call_id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: tool_call.function.arguments.clone().unwrap_or_default(),
                output: None, // Tool hasn't been executed yet
                status: "in_progress".to_string(),
            });
        }
    }

    // Determine response status based on finish_reason
    let status = match choice.finish_reason.as_deref() {
        Some("stop") | Some("length") => ResponseStatus::Completed,
        Some("tool_calls") => ResponseStatus::InProgress, // Waiting for tool execution
        Some("failed") | Some("error") => ResponseStatus::Failed,
        _ => ResponseStatus::Completed, // Default to completed
    };

    // Convert usage from Usage to UsageInfo, then wrap in ResponsesUsage
    let usage = chat_resp.usage.as_ref().map(|u| {
        let usage_info = UsageInfo {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            reasoning_tokens: u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            prompt_tokens_details: None, // Chat response doesn't have this
        };
        ResponsesUsage::Classic(usage_info)
    });

    // Generate response
    let response_id = response_id_override.unwrap_or_else(|| chat_resp.id.clone());
    Ok(ResponsesResponse::builder(&response_id, &chat_resp.model)
        .copy_from_request(original_req)
        .created_at(chat_resp.created as i64)
        .status(status)
        .output(output)
        .maybe_text(original_req.text.clone())
        .maybe_usage(usage)
        .build())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_input_conversion() {
        let req = ResponsesRequest {
            input: ResponseInput::Text("Hello, world!".to_string()),
            instructions: Some("You are a helpful assistant.".to_string()),
            model: "gpt-4".to_string(),
            temperature: Some(0.7),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2); // system + user
        assert_eq!(chat_req.model, "gpt-4");
        assert_eq!(chat_req.temperature, Some(0.7));
    }

    #[test]
    fn test_items_input_conversion() {
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![
                ResponseInputOutputItem::Message {
                    id: "msg_1".to_string(),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText {
                        text: "Hello!".to_string(),
                    }],
                    status: None,
                },
                ResponseInputOutputItem::Message {
                    id: "msg_2".to_string(),
                    role: "assistant".to_string(),
                    content: vec![ResponseContentPart::OutputText {
                        text: "Hi there!".to_string(),
                        annotations: vec![],
                        logprobs: None,
                    }],
                    status: None,
                },
            ]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2); // user + assistant
    }

    #[test]
    fn test_empty_input_error() {
        let req = ResponsesRequest {
            input: ResponseInput::Text("".to_string()),
            ..Default::default()
        };

        // Empty text should still create a user message, so this should succeed
        let result = responses_to_chat(&req);
        assert!(result.is_ok());
    }

    /// Regression test for a `function_call`/`function_call_output` pair whose
    /// `id` and `call_id` differ (the normal case for real Responses API
    /// clients, e.g. `id: "fc_..."` vs `call_id: "call_..."`). The assistant
    /// message's `tool_calls[0].id` must match the following tool message's
    /// `tool_call_id` — both must key off `call_id`, not `id` — otherwise the
    /// backend can't correlate the tool call with its result.
    #[test]
    fn test_function_call_output_correlates_via_call_id_not_id() {
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![
                ResponseInputOutputItem::FunctionToolCall {
                    id: Some("fc1".to_string()),
                    call_id: "c1".to_string(),
                    name: "view_image".to_string(),
                    arguments: "{}".to_string(),
                    output: None,
                    status: Some("in_progress".to_string()),
                },
                ResponseInputOutputItem::FunctionCallOutput {
                    id: None,
                    call_id: "c1".to_string(),
                    output: FunctionCallOutputContent::Text("done".to_string()),
                    status: None,
                },
            ]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2);

        let ChatMessage::Assistant { tool_calls, .. } = &chat_req.messages[0] else {
            panic!("expected assistant message, got {:?}", chat_req.messages[0]);
        };
        let tool_call_id = &tool_calls.as_ref().unwrap()[0].id;

        let ChatMessage::Tool { tool_call_id: output_tool_call_id, .. } = &chat_req.messages[1]
        else {
            panic!("expected tool message, got {:?}", chat_req.messages[1]);
        };

        // Both must key off call_id ("c1"), not id ("fc1").
        assert_eq!(tool_call_id, "c1");
        assert_eq!(output_tool_call_id, "c1");
        assert_eq!(tool_call_id, output_tool_call_id);
    }

    /// A `message` item whose content mixes `input_text` and `input_image`
    /// must preserve the image as a `ContentPart::ImageUrl` instead of
    /// silently dropping it (the pre-fix behavior for the unrecognized
    /// `input_image` variant).
    #[test]
    fn test_message_with_input_image_preserves_image() {
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "user".to_string(),
                content: vec![
                    ResponseContentPart::InputText {
                        text: "describe this image".to_string(),
                    },
                    ResponseContentPart::InputImage {
                        image_url: crate::protocols::responses::ResponseImageUrlValue::Url(
                            "data:image/png;base64,AAAA".to_string(),
                        ),
                        detail: None,
                    },
                ],
                status: None,
            }]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 1);

        let ChatMessage::User { content, .. } = &chat_req.messages[0] else {
            panic!("expected user message, got {:?}", chat_req.messages[0]);
        };
        let MessageContent::Parts(parts) = content else {
            panic!("expected MessageContent::Parts, got {:?}", content);
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(parts[0], ContentPart::Text { .. }));
        match &parts[1] {
            ContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "data:image/png;base64,AAAA");
                assert_eq!(image_url.detail.as_deref(), Some("auto"));
            }
            other => panic!("expected ContentPart::ImageUrl, got {:?}", other),
        }
    }

    /// A `function_call_output` whose `output` is a content-part array
    /// containing an `input_image` (e.g. a `view_image`-style tool result)
    /// must preserve the image instead of being flattened to empty text.
    #[test]
    fn test_function_call_output_with_image_preserves_image() {
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
                id: None,
                call_id: "c1".to_string(),
                output: FunctionCallOutputContent::Parts(vec![ResponseContentPart::InputImage {
                    image_url: crate::protocols::responses::ResponseImageUrlValue::Url(
                        "data:image/png;base64,AAAA".to_string(),
                    ),
                    detail: None,
                }]),
                status: None,
            }]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 1);

        let ChatMessage::Tool {
            content,
            tool_call_id,
        } = &chat_req.messages[0]
        else {
            panic!("expected tool message, got {:?}", chat_req.messages[0]);
        };
        assert_eq!(tool_call_id, "c1");
        let MessageContent::Parts(parts) = content else {
            panic!("expected MessageContent::Parts, got {:?}", content);
        };
        assert_eq!(parts.len(), 1);
        assert!(matches!(parts[0], ContentPart::ImageUrl { .. }));
    }
}
