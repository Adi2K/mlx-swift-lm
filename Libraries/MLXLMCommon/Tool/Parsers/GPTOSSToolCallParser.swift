// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for the GPT-OSS harmony format:
/// `<|channel|>commentary to=functions.name <|constrain|>json<|message|>{JSON}<|call|>`
///
/// GPT-OSS structures its output as a sequence of channel-tagged messages:
/// reasoning on the `analysis` channel, tool calls on the `commentary` channel
/// addressed to a `functions.name` recipient, and user-facing text on the
/// `final` channel. The recipient may appear in the channel header (the form
/// the model generates) or in the role header (the form the chat template uses
/// to render prior turns):
/// `<|start|>assistant to=functions.name<|channel|>commentary json<|message|>{JSON}<|call|>`
///
/// Analysis and final channel messages diverge from `startTag` right after
/// `<|channel|>`, so `ToolCallProcessor` passes them through as regular text
/// with their channel markers intact (mirroring how `<think>` spans pass
/// through untouched for other formats).
///
/// `<|call|>` is one of GPT-OSS's EOS tokens and is intercepted at the token
/// ID level before detokenization, so streamed tool calls usually end without
/// it and are extracted via `ToolCallProcessor.processEOS()` at generation
/// end; the tag is still honored when it does appear as text.
///
/// Reference: https://cookbook.openai.com/articles/openai-harmony
public struct GPTOSSToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|channel|>commentary"
    public let endTag: String? = "<|call|>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        let text = content

        // Find the recipient marker (e.g. "to=functions.get_weather"),
        // whether it appears in the channel header or the role header.
        guard let toRange = text.range(of: "to=") else { return nil }

        // The recipient extends to the next whitespace or special-token marker.
        var nameEnd = toRange.upperBound
        while nameEnd < text.endIndex, !text[nameEnd].isWhitespace, text[nameEnd] != "<" {
            nameEnd = text.index(after: nameEnd)
        }

        var funcName = String(text[toRange.upperBound ..< nameEnd])

        // Strip "functions." prefix if present
        if funcName.hasPrefix("functions.") {
            funcName = String(funcName.dropFirst("functions.".count))
        } else if let dotIdx = funcName.firstIndex(of: ".") {
            // Also handle other namespaces like "tools."
            funcName = String(funcName[funcName.index(after: dotIdx)...])
        }

        guard !funcName.isEmpty else { return nil }

        // Arguments are the <|message|> payload following the recipient,
        // running up to <|call|> — or to the end of the span when <|call|>
        // was consumed as a stop token. <|end|>/<|return|> guard against
        // malformed spans that terminate like a non-tool message.
        guard let msgRange = text.range(of: "<|message|>", range: nameEnd ..< text.endIndex)
        else { return nil }

        var argsStr = String(text[msgRange.upperBound...])
        let terminators = ["<|call|>", "<|end|>", "<|return|>"]
        if let cut = terminators.compactMap({ argsStr.range(of: $0) })
            .min(by: { $0.lowerBound < $1.lowerBound })
        {
            argsStr = String(argsStr[..<cut.lowerBound])
        }
        argsStr = argsStr.trimmingCharacters(in: .whitespacesAndNewlines)

        // Deserialize the JSON arguments
        guard let arguments = tryParseJSON(argsStr) as? [String: any Sendable] else {
            return nil
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }
}
