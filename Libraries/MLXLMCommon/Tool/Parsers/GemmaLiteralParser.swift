// Copyright © 2026 Apple Inc.

import Foundation

/// Reads values written in the Gemma tool-call dialect.
///
/// The dialect is JSON with two changes: object keys may be bare, and strings are usually wrapped
/// in an escape marker (`<|"|>` for Gemma 4, `<escape>` for FunctionGemma) instead of double
/// quotes. Text between markers is verbatim, so it needs no escaping. Whichever of the marker or
/// `"` opens a string decides how it is read, so each may contain the other.
///
/// Scalars and quoted strings keep their JSON meaning. A bare word JSON cannot name, as in
/// `{ok: yes}`, fails the whole value: guessing a type there is worse than keeping the text.
struct GemmaLiteralParser {
    private let source: Substring
    private let marker: String
    private var index: String.Index
    private var depth = 0

    private static let maximumDepth = 32

    /// Characters that end a bare key or scalar, and that a bare key may not contain.
    private static let structural: Set<Character> = [",", "{", "}", "[", "]"]

    private init(_ source: Substring, marker: String) {
        self.source = source
        self.marker = marker
        self.index = source.startIndex
    }

    /// Parses `source` as one value, or returns `nil` if any part of it is malformed.
    static func parse(_ source: Substring, marker: String) -> (any Sendable)? {
        var parser = Self(source, marker: marker)
        guard let value = parser.value() else { return nil }
        parser.skipWhitespace()
        return parser.isAtEnd ? value : nil
    }

    /// Reads a key the caller has delimited: a marker or JSON string, or bare text taken as is.
    static func key(_ source: Substring, marker: String) -> String? {
        let text = source.trimmingWhitespace()
        guard text.hasPrefix(marker) || text.first == "\"" else { return String(text) }
        return parse(text, marker: marker) as? String
    }

    // MARK: - Grammar

    private mutating func value() -> (any Sendable)? {
        guard depth < Self.maximumDepth else { return nil }
        depth += 1
        defer { depth -= 1 }
        skipWhitespace()

        if atMarker { return markedString() }
        switch current {
        case "\"": return quotedString()
        case "{": return object()
        case "[": return array()
        default: return scalar()
        }
    }

    private mutating func object() -> (any Sendable)? {
        advance()
        var members: [String: any Sendable] = [:]
        while true {
            if consume("}") { return members }
            guard let key = key(), consume(":"), let value = value() else { return nil }
            members[key] = value
            guard consume(",") else { return consume("}") ? members : nil }
        }
    }

    private mutating func array() -> (any Sendable)? {
        advance()
        var elements: [any Sendable] = []
        while true {
            if consume("]") { return elements }
            guard let element = value() else { return nil }
            elements.append(element)
            guard consume(",") else { return consume("]") ? elements : nil }
        }
    }

    /// A marker string, a JSON string, or a bare run up to the colon.
    private mutating func key() -> String? {
        if atMarker { return markedString() }
        if current == "\"" { return quotedString() }

        let start = index
        while let character = current, character != ":" {
            guard !Self.structural.contains(character) else { return nil }
            advance()
        }
        let key = source[start ..< index].trimmingWhitespace()
        return key.isEmpty ? nil : String(key)
    }

    private mutating func markedString() -> String? {
        let start = source.index(index, offsetBy: marker.count)
        guard let close = source[start...].range(of: marker) else { return nil }
        index = close.upperBound
        return String(source[start ..< close.lowerBound])
    }

    private mutating func quotedString() -> String? {
        let start = index
        advance()
        while let character = current {
            advance()
            if character == "\\" {
                if !isAtEnd { advance() }
            } else if character == "\"" {
                return Self.json(source[start ..< index]) as? String
            }
        }
        return nil
    }

    /// `true`, `false`, `null` or a number. Anything else is not a value.
    private mutating func scalar() -> (any Sendable)? {
        let start = index
        while let character = current, !Self.structural.contains(character) { advance() }
        return Self.json(source[start ..< index].trimmingWhitespace())
    }

    /// Decodes one JSON token, so escapes and number forms mean what they mean in JSON.
    private static func json(_ token: Substring) -> (any Sendable)? {
        guard
            let object = try? JSONSerialization.jsonObject(
                with: Data(token.utf8), options: .fragmentsAllowed)
        else { return nil }
        return asSendable(object)
    }

    // MARK: - Cursor

    private var isAtEnd: Bool { index >= source.endIndex }

    private var current: Character? { isAtEnd ? nil : source[index] }

    private var atMarker: Bool { source[index...].hasPrefix(marker) }

    private mutating func advance() {
        index = source.index(after: index)
    }

    private mutating func skipWhitespace() {
        while current?.isWhitespace == true { advance() }
    }

    /// Consumes `character` if it comes next, after any whitespace.
    private mutating func consume(_ character: Character) -> Bool {
        skipWhitespace()
        guard current == character else { return false }
        advance()
        return true
    }
}
