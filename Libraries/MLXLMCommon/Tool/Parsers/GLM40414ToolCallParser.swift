// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for the GLM-4-0414 tool call format: `name\n{JSON arguments}`.
///
/// The dense GLM-4-0414 family (GLM-4-9B/32B-0414, GLM-Z1-*-0414;
/// `model_type == "glm4"`, `Glm4ForCausalLM`) does **not** use the tagged
/// `<tool_call>…<arg_key>…<arg_value>…` syntax of the GLM-4.5/4.6 MoE models
/// (`glm4_moe`, handled by ``GLM4ToolCallParser``). Its chat template renders an
/// assistant tool call as the function name followed by a newline and the
/// JSON-encoded arguments, with no wrapper tokens:
///
/// ```
/// <|assistant|>get_lab_result
/// {"test_name": "hemoglobin"}
/// ```
///
/// This mirrors the reference `process_response` in the official GLM-4 repo
/// (`zai-org/GLM-4`, `demo/composite_demo/src/client.py`), which splits each
/// `<|assistant|>` turn on the first newline: the leading line is the function
/// name (the message `metadata`) and the remainder is `json.loads`-ed as the
/// arguments.
///
/// Because there is no start tag, this is an inline (tagless) format. The
/// function name precedes the JSON, so ``ToolCallProcessor`` buffers the
/// candidate name until it can confirm a call (see `usesLeadingFunctionName`):
/// the leading token must be a bare function name — and, when tool schemas are
/// available, one of the declared tools — immediately followed by a JSON object.
/// Anything else passes through as regular text.
///
/// The terminating token after a call (`<|observation|>` / `<|user|>` /
/// `<|endoftext|>`) is one of the model's EOS token IDs and is intercepted
/// before detokenization, so a streamed call is confirmed either when its JSON
/// object closes mid-stream or, failing that, via
/// ``ToolCallProcessor/processEOS(returnBufferedText:)`` at generation end.
public struct GLM40414ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = nil
    public let endTag: String? = nil

    /// GLM-4-0414 emits the function name ahead of the JSON arguments, so the
    /// streaming processor must retain the leading name rather than treating the
    /// pre-`{` text as regular output.
    public var usesLeadingFunctionName: Bool { true }

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content

        // A historical/second call may be prefixed with the assistant role token.
        if let range = text.range(of: "<|assistant|>") {
            text = String(text[range.upperBound...])
        }

        // The arguments are the first top-level JSON object; the function name is
        // everything before it. Using the `{` boundary (rather than only the
        // newline) keeps parsing robust to the trailing space the chat template
        // renders after the name.
        guard let braceIndex = text.firstIndex(of: "{") else { return nil }

        let name = String(text[..<braceIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard isBareFunctionName(name), nameMatchesTools(name, tools: tools) else { return nil }

        let jsonString = String(text[braceIndex...]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard let arguments = tryParseJSON(jsonString) as? [String: any Sendable] else {
            return nil
        }

        return ToolCall(function: .init(name: name, arguments: arguments))
    }

    /// Multiple tool calls in a single turn are separated by the `<|assistant|>`
    /// role token (which, unlike the terminating observation/user tokens, is not
    /// an EOS token). Each segment is parsed independently.
    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?)
        -> [ToolCall]
    {
        toolCallBuffer
            .components(separatedBy: "<|assistant|>")
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .compactMap { parse(content: $0, tools: tools) }
    }

    // MARK: - Helpers

    /// Whether `name` is a single bare function-name token (identifier characters
    /// only, no internal whitespace). This is the primary guard that keeps plain
    /// prose — whose leading line is not a lone identifier — from being parsed as
    /// a call.
    private func isBareFunctionName(_ name: String) -> Bool {
        guard let first = name.first, first.isLetter || first == "_" else { return false }
        for ch in name where !(ch.isLetter || ch.isNumber || ch == "_" || ch == "-" || ch == ".") {
            return false
        }
        return true
    }

    /// When tool schemas are available, the name must match a declared tool. With
    /// no schemas the bare-name shape check alone gates detection.
    private func nameMatchesTools(_ name: String, tools: [[String: any Sendable]]?) -> Bool {
        guard let tools else { return true }
        let names = tools.compactMap {
            ($0["function"] as? [String: any Sendable])?["name"] as? String
        }
        return names.isEmpty || names.contains(name)
    }
}
