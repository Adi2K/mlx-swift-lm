import Foundation
import MLXLMCommon
import Testing

struct ToolTests {
    @Test("Test Weather Tool Schema Generation")
    func testWeatherToolSchemaGeneration() throws {
        struct WeatherInput: Codable {
            let location: String
            let unit: String?
        }

        struct WeatherOutput: Codable {
            let temperature: Double
            let conditions: String
        }

        let tool = Tool<WeatherInput, WeatherOutput>(
            name: "get_current_weather",
            description: "Get the current weather in a given location",
            parameters: [
                .required(
                    "location", type: .string, description: "The city, e.g. Istanbul"
                ),
                .optional(
                    "unit",
                    type: .string,
                    description: "The unit of temperature",
                    extraProperties: [
                        "enum": ["celsius", "fahrenheit"]
                    ]
                ),
            ]
        ) { input in
            WeatherOutput(temperature: 14.0, conditions: "Sunny")
        }

        let actual = tool.schema as NSDictionary

        let expected: NSDictionary = [
            "type": "function",
            "function": [
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "location": [
                            "type": "string",
                            "description": "The city, e.g. Istanbul",
                        ],
                        "unit": [
                            "type": "string",
                            "description": "The unit of temperature",
                            "enum": ["celsius", "fahrenheit"],
                        ],
                    ],
                    "required": ["location"],
                ],
            ],
        ]

        #expect(actual == expected)
    }

    @Test("Test Tool Call Detection in Generated Text - Default JSON Format")
    func testToolCallDetection() throws {
        let processor = ToolCallProcessor()
        let chunks: [String] = [
            "<tool", "_", "call>", "{", "\"", "name", "\"", ":", " ", "\"", "get", "_", "current",
            "_", "weather", "\"", ",", " ", "\"", "arguments", "\"", ":", " ", "{", "\"",
            "location", "\"", ":", " ", "\"", "San", " Francisco", "\"", ",", " ", "\"", "unit",
            "\"", ":", " ", "\"", "celsius", "\"", "}", "}", "</tool", "_", "call>",
        ]

        for chunk in chunks {
            let result = processor.processChunk(chunk)
            #expect(result == nil)
        }

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)

        #expect(toolCall.function.name == "get_current_weather")
        #expect(toolCall.function.arguments["location"] == .string("San Francisco"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    // MARK: - JSON Format Tests

    @Test("Test JSON Tool Call Parser - Default Tags")
    func testJSONParserDefaultTags() throws {
        let parser = JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}</tool_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test("Test JSON Tool Call Parser - Custom Tags")
    func testJSONParserCustomTags() throws {
        let parser = JSONToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>{\"name\": \"search\", \"arguments\": {\"query\": \"swift programming\"}}<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("swift programming"))
    }

    @Test("Test JSON Tool Call Parser - Stringified Arguments")
    func testJSONParserStringifiedArguments() throws {
        let parser = JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            #"<tool_call>{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}</tool_call>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test JSON Tool Call Parser - Stringified Empty Arguments")
    func testJSONParserStringifiedEmptyArguments() throws {
        let parser = JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            #"<tool_call>{"name":"current_time","arguments":"{}"}</tool_call>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "current_time")
        #expect(toolCall.function.arguments.isEmpty)
    }

    @Test("Test JSON Tool Call Parser - Stringified Array Arguments")
    func testJSONParserStringifiedArrayArguments() throws {
        let parser = JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            #"<tool_call>{"name":"search_many","arguments":"{\"queries\":[\"swift\",\"mlx\"],\"limit\":2}"}</tool_call>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search_many")
        #expect(toolCall.function.arguments["limit"] == .int(2))
        #expect(
            toolCall.function.arguments["queries"] == .array([.string("swift"), .string("mlx")]))
    }

    @Test("Test JSON Format via ToolCallProcessor - Bare JSON Fallback")
    func testJSONFormatProcessorBareJSONFallback() throws {
        let processor = ToolCallProcessor(format: .json)
        let chunks: [String] = [
            "{\"name\": \"get_weather\", ",
            "\"arguments\": {\"location\": \"Rome\"}}",
        ]

        var emittedText = ""
        for chunk in chunks {
            if let text = processor.processChunk(chunk) {
                emittedText += text
            }
        }

        if let text = processor.processEOS(returnBufferedText: true) {
            emittedText += text
        }

        #expect(emittedText.isEmpty)
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Rome"))
    }

    @Test("Test JSON Format via ToolCallProcessor - Bare JSON With Leading Text")
    func testJSONFormatProcessorBareJSONWithLeadingText() throws {
        let processor = ToolCallProcessor(format: .json)
        let chunk =
            "Let me check that.\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Milan\"}}"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == "Let me check that.\n")
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Milan"))
    }

    @Test("Test JSON Format via ToolCallProcessor - Tagged JSON In Single Chunk")
    func testJSONFormatProcessorTaggedSingleChunk() throws {
        let processor = ToolCallProcessor(format: .json)
        let chunk =
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Tokyo\"}}</tool_call>"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == nil)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Test JSON Format via ToolCallProcessor - Multiple Tagged Calls Preserve Order")
    func testJSONFormatProcessorMultipleTaggedCallsPreserveOrder() throws {
        let processor = ToolCallProcessor(format: .json)
        let chunk =
            "<tool_call>{\"name\":\"first_call\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"second_call\",\"arguments\":{}}</tool_call>"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == nil)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 2)
        #expect(processor.toolCalls[0].function.name == "first_call")
        #expect(processor.toolCalls[1].function.name == "second_call")
    }

    @Test("Test JSON Format via ToolCallProcessor - Tagged JSON With Leading Text")
    func testJSONFormatProcessorTaggedWithLeadingText() throws {
        let processor = ToolCallProcessor(format: .json)
        let chunk =
            "Let me check that.\n<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Osaka\"}}</tool_call>"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == "Let me check that.\n")
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Osaka"))
    }

    @Test("Test JSON Format via ToolCallProcessor - Invalid Bare JSON Flushes At EOS")
    func testJSONFormatProcessorInvalidBareJSONFlushesAtEOS() {
        let processor = ToolCallProcessor(format: .json)
        let chunk = "{\"name\": \"get_weather\", \"arguments\": "

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == nil)
        #expect(eosOutput == chunk)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Test JSON Format via ToolCallProcessor - Non Tool JSON Stays Text")
    func testJSONFormatProcessorNonToolJSONStaysText() {
        let processor = ToolCallProcessor(format: .json)
        let chunk = "{\"status\": \"ok\", \"data\": {\"value\": 42}}"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == chunk)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Test JSON Format via ToolCallProcessor - Split Non Tool JSON Stays Text")
    func testJSONFormatProcessorSplitNonToolJSONStaysText() {
        let processor = ToolCallProcessor(format: .json)
        let chunks = ["{\"status\": ", "\"ok\", \"data\": {\"value\": 42}}"]

        var emittedText = ""
        for chunk in chunks {
            if let output = processor.processChunk(chunk) {
                emittedText += output
            }
        }

        if let eosOutput = processor.processEOS(returnBufferedText: true) {
            emittedText += eosOutput
        }

        #expect(emittedText == "{\"status\": \"ok\", \"data\": {\"value\": 42}}")
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Test JSON Format via ToolCallProcessor - Missing Arguments Stays Text")
    func testJSONFormatProcessorMissingArgumentsStaysText() {
        let processor = ToolCallProcessor(format: .json)
        let chunk = "{\"name\": \"not_a_tool_call_payload\"}"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == chunk)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Test JSON Format via ToolCallProcessor - Brace Text Is Not Treated As JSON Tool Call")
    func testJSONFormatProcessorBraceTextNotToolCall() {
        let processor = ToolCallProcessor(format: .json)

        let first = processor.processChunk("Use {")
        let second = processor.processChunk("x} notation")
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(first == "Use ")
        #expect(second == "{x} notation")
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test(
        "Test JSON Format via ToolCallProcessor - Unknown Tool Name Stays Text When Tools Are Provided"
    )
    func testJSONFormatProcessorUnknownToolNameStaysTextWithTools() {
        struct EmptyInput: Codable {}
        struct EmptyOutput: Codable { let ok: Bool }

        let tool = Tool<EmptyInput, EmptyOutput>(
            name: "get_weather",
            description: "Gets weather",
            parameters: []
        ) { _ in
            EmptyOutput(ok: true)
        }

        let processor = ToolCallProcessor(format: .json, tools: [tool.schema])
        let chunk = "{\"name\": \"not_declared\", \"arguments\": {}}"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == chunk)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Test JSON Format via ToolCallProcessor - Recovers Tagged Tool Call After Brace Text")
    func testJSONFormatProcessorRecoversTaggedToolCallAfterBraceText() throws {
        let processor = ToolCallProcessor(format: .json)
        var emittedText = ""

        if let output = processor.processChunk("note {x") {
            emittedText += output
        }
        if let output = processor.processChunk(
            "} <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}</tool_call>"
        ) {
            emittedText += output
        }
        if let eosOutput = processor.processEOS(returnBufferedText: true) {
            emittedText += eosOutput
        }

        #expect(emittedText == "note {x} ")
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test(
        "Test JSON Format via ToolCallProcessor - Unknown Tagged Tool Preserved And Continues Parsing"
    )
    func testJSONFormatProcessorUnknownTaggedToolPreservedAndContinuesParsing() throws {
        struct EmptyInput: Codable {}
        struct EmptyOutput: Codable { let ok: Bool }

        let tool = Tool<EmptyInput, EmptyOutput>(
            name: "get_weather",
            description: "Gets weather",
            parameters: []
        ) { _ in
            EmptyOutput(ok: true)
        }

        let processor = ToolCallProcessor(format: .json, tools: [tool.schema])
        let chunk =
            "<tool_call>{\"name\":\"not_declared\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == "<tool_call>{\"name\":\"not_declared\",\"arguments\":{}}</tool_call>")
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
    }

    @Test(
        "Test JSON Format via ToolCallProcessor - Unknown Tagged Tool With Leading Text Preserved"
    )
    func testJSONFormatProcessorUnknownTaggedToolWithLeadingTextPreserved() throws {
        struct EmptyInput: Codable {}
        struct EmptyOutput: Codable { let ok: Bool }

        let tool = Tool<EmptyInput, EmptyOutput>(
            name: "get_weather",
            description: "Gets weather",
            parameters: []
        ) { _ in
            EmptyOutput(ok: true)
        }

        let processor = ToolCallProcessor(format: .json, tools: [tool.schema])
        let chunk =
            "Preface <tool_call>{\"name\":\"not_declared\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(
            output == "Preface <tool_call>{\"name\":\"not_declared\",\"arguments\":{}}</tool_call>")
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
    }

    @Test(
        "Test JSON Format via ToolCallProcessor - Declared Tool Name Parses When Tools Are Provided"
    )
    func testJSONFormatProcessorDeclaredToolNameParsesWithTools() throws {
        struct EmptyInput: Codable {}
        struct EmptyOutput: Codable { let ok: Bool }

        let tool = Tool<EmptyInput, EmptyOutput>(
            name: "get_weather",
            description: "Gets weather",
            parameters: []
        ) { _ in
            EmptyOutput(ok: true)
        }

        let processor = ToolCallProcessor(format: .json, tools: [tool.schema])
        let chunk = "{\"name\": \"get_weather\", \"arguments\": {}}"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == nil)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
    }

    // MARK: - Pythonic Format Tests (LFM2/LFM2.5)

    @Test("Test Pythonic Tool Call Parser - Basic")
    func testPythonicParserBasic() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>[get_weather(location='Paris', unit='celsius')]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test Pythonic Tool Call Parser - Double Quotes")
    func testPythonicParserDoubleQuotes() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>[search(query=\"swift programming\")]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("swift programming"))
    }

    @Test("Test Pythonic Tool Call Parser - Without Brackets")
    func testPythonicParserWithoutBrackets() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>current_time(timezone='UTC')<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "current_time")
        #expect(toolCall.function.arguments["timezone"] == .string("UTC"))
    }

    @Test("Test Pythonic Tool Call Parser - Nested Parentheses in Argument Value")
    func testPythonicParserNestedParentheses() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>[run_script(code=\"response = requests.get('https://api.example.com/data')\")] <|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "run_script")
        #expect(
            toolCall.function.arguments["code"]
                == .string("response = requests.get('https://api.example.com/data')"))
    }

    @Test("Test Pythonic Tool Call Parser - Nested Parentheses Without Brackets")
    func testPythonicParserNestedParenthesesNoBrackets() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>run_script(code=\"print('hello')\")<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "run_script")
        #expect(toolCall.function.arguments["code"] == .string("print('hello')"))
    }

    @Test("Test Pythonic Tool Call Parser - No Arguments")
    func testPythonicParserNoArguments() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>[current_time()]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "current_time")
        #expect(toolCall.function.arguments.isEmpty)
    }

    @Test("Test Pythonic Tool Call Parser - Multiple Tools via parseEOS")
    func testPythonicParserMultipleToolsEOS() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")

        let content1 =
            "<|tool_call_start|>[get_weather(location='Paris'), current_time(timezone=\"UTC\")]<|tool_call_end|>"
        let toolCalls1 = parser.parseEOS(content1, tools: nil)

        #expect(toolCalls1.count == 2)
        #expect(toolCalls1[0].function.name == "get_weather")
        #expect(toolCalls1[0].function.arguments["location"] == .string("Paris"))
        #expect(toolCalls1[1].function.name == "current_time")
        #expect(toolCalls1[1].function.arguments["timezone"] == .string("UTC"))

        // Multiple distinct tool call blocks
        let content2 =
            "<|tool_call_start|>[get_weather(location='London')]<|tool_call_end|> <text> <|tool_call_start|>[current_time(timezone='UTC')]<|tool_call_end|>"
        let toolCalls2 = parser.parseEOS(content2, tools: nil)

        #expect(toolCalls2.count == 2)
        #expect(toolCalls2[0].function.name == "get_weather")
        #expect(toolCalls2[0].function.arguments["location"] == .string("London"))
        #expect(toolCalls2[1].function.name == "current_time")
        #expect(toolCalls2[1].function.arguments["timezone"] == .string("UTC"))
    }

    @Test("Test Pythonic Tool Call Parser - Type Conversion")
    func testPythonicParserTypeConversion() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "set_temperature",
                    "parameters": [
                        "properties": [
                            "value": ["type": "integer"],
                            "enabled": ["type": "boolean"],
                        ]
                    ],
                ] as [String: any Sendable]
            ]
        ]
        let content =
            "<|tool_call_start|>[set_temperature(value='25', enabled='true')]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: tools))

        #expect(toolCall.function.name == "set_temperature")
        #expect(toolCall.function.arguments["value"] == .int(25))
        #expect(toolCall.function.arguments["enabled"] == .bool(true))
    }

    @Test("Test LFM2 Format via ToolCallProcessor - Pythonic")
    func testLFM2FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .lfm2)
        let content =
            "<|tool_call_start|>[calculator(expression='2+2')]<|tool_call_end|>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "calculator")
        #expect(toolCall.function.arguments["expression"] == .string("2+2"))
    }

    // MARK: - XML Function Format Tests (Qwen3 Coder)

    @Test("Test XML Function Parser - Qwen3 Coder Format")
    func testXMLFunctionParser() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            "<function=get_weather><parameter=location>Tokyo</parameter><parameter=unit>celsius</parameter></function>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test XML Function Parser - With Type Conversion")
    func testXMLFunctionParserTypeConversion() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "set_temperature",
                    "parameters": [
                        "properties": [
                            "value": ["type": "integer"],
                            "enabled": ["type": "boolean"],
                        ]
                    ],
                ] as [String: any Sendable]
            ]
        ]
        let content =
            "<function=set_temperature><parameter=value>25</parameter><parameter=enabled>true</parameter></function>"

        let toolCall = try #require(parser.parse(content: content, tools: tools))

        #expect(toolCall.function.name == "set_temperature")
        #expect(toolCall.function.arguments["value"] == .int(25))
        #expect(toolCall.function.arguments["enabled"] == .bool(true))
    }

    @Test("Test XML Function Parser - Multiline Content (Qwen3.5 style)")
    func testXMLFunctionParserMultiline() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        // Qwen3.5 models generate newlines between the XML tags
        let content = """
            <tool_call>
            <function=get_current_datetime>
            </function>
            </tool_call>
            """

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_current_datetime")
        #expect(toolCall.function.arguments.isEmpty)
    }

    @Test("Test XML Function Parser - Multiline Parameters")
    func testXMLFunctionParserMultilineParams() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content = """
            <function=get_weather>
            <parameter=location>
            Tokyo
            </parameter>
            </function>
            """

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    // MARK: - Qwen3.5 Format Tests (XML Function with tool_call wrapper)

    @Test("Test Qwen3.5 XML Function Parser - With tool_call Tags")
    func testQwen35Parser() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content = """
            <tool_call>
            <function=get_weather>
            <parameter=location>
            San Francisco
            </parameter>
            <parameter=unit>
            celsius
            </parameter>
            </function>
            </tool_call>
            """

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("San Francisco"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test Qwen3.5 Format via ToolCallProcessor")
    func testQwen35FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .xmlFunction)
        let chunks: [String] = [
            "<tool", "_call>", "\n<function=get_weather>\n",
            "<parameter=location>\nTokyo\n</parameter>",
            "\n</function>\n</tool_call>",
        ]

        for chunk in chunks {
            _ = processor.processChunk(chunk)
        }

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Test Qwen3.5 Format - No Arguments")
    func testQwen35FormatNoArgs() throws {
        let processor = ToolCallProcessor(format: .xmlFunction)
        let content = "<tool_call>\n<function=get_current_datetime>\n</function>\n</tool_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_current_datetime")
        #expect(toolCall.function.arguments.isEmpty)
    }

    // MARK: - GLM4 Format Tests

    @Test("Test GLM4 Tool Call Parser")
    func testGLM4Parser() throws {
        let parser = GLM4ToolCallParser()
        let content =
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Berlin</arg_value><arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Berlin"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test GLM4 Format via ToolCallProcessor")
    func testGLM4FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .glm4)
        let content =
            "<tool_call>search<arg_key>query</arg_key><arg_value>machine learning</arg_value></tool_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("machine learning"))
    }

    // MARK: - Gemma Format Tests

    @Test("Test Gemma Function Parser")
    func testGemmaParser() throws {
        let parser = GemmaFunctionParser(
            startTag: "<start_function_call>", endTag: "<end_function_call>",
            escapeMarker: "<escape>")
        let content =
            "<start_function_call>call:get_weather{location:Paris,unit:celsius}<end_function_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test Gemma Function Parser - Escaped Strings")
    func testGemmaParserEscapedStrings() throws {
        let parser = GemmaFunctionParser(
            startTag: "<start_function_call>", endTag: "<end_function_call>",
            escapeMarker: "<escape>")
        // Note: Gemma uses <escape> for both start and end markers (not </escape>)
        let content =
            "<start_function_call>call:search{query:<escape>hello, world!<escape>}<end_function_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("hello, world!"))
    }

    @Test("Test Gemma 4 Function Parser - Type Conversion")
    func testGemma4ParserTypeConversion() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: #"<|"|>"#)
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "mail_read",
                    "parameters": [
                        "properties": [
                            "account": ["type": "string"],
                            "mailbox": ["type": "string"],
                            "id": ["type": "integer"],
                        ]
                    ],
                ] as [String: any Sendable]
            ]
        ]
        let content =
            #"<|tool_call>call:mail_read{account:<|"|>me@example.com<|"|>,mailbox:<|"|>INBOX<|"|>,id:<|"|>158348<|"|>}<tool_call|>"#

        let toolCall = try #require(parser.parse(content: content, tools: tools))

        #expect(toolCall.function.name == "mail_read")
        #expect(toolCall.function.arguments["account"] == .string("me@example.com"))
        #expect(toolCall.function.arguments["mailbox"] == .string("INBOX"))
        #expect(toolCall.function.arguments["id"] == .int(158_348))
        #expect(toolCall.function.arguments["id"] != .string("158348"))
    }

    @Test("Test Gemma Format via ToolCallProcessor")
    func testGemmaFormatProcessor() throws {
        let processor = ToolCallProcessor(format: .gemma)
        let content = "<start_function_call>call:calculator{expression:2+2}<end_function_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "calculator")
        #expect(toolCall.function.arguments["expression"] == .string("2+2"))
    }

    // MARK: - GPT-OSS Harmony Format Tests

    @Test("Test GPT-OSS Tool Call Parser")
    func testGPTOSSParser() throws {
        let parser = GPTOSSToolCallParser()
        let content =
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\": \"Tokyo\"}<|call|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Test GPT-OSS Tool Call Parser - Recipient in Role Header")
    func testGPTOSSParserRoleRecipient() throws {
        // The chat template renders historical tool calls with the recipient in
        // the role header instead of the channel header.
        let parser = GPTOSSToolCallParser()
        let content =
            "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"location\": \"Paris\", \"days\": 3}<|call|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
        #expect(toolCall.function.arguments["days"] == .int(3))
    }

    @Test("Test GPT-OSS Tool Call Parser - Without <|call|> Token")
    func testGPTOSSParserWithoutCallToken() throws {
        // <|call|> is one of GPT-OSS's EOS tokens and is intercepted at the
        // token ID level, so a streamed tool call span usually ends without it.
        let parser = GPTOSSToolCallParser()
        let content =
            "<|channel|>commentary to=functions.search <|constrain|>json<|message|>{\"query\": \"swift\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("swift"))
    }

    @Test("Test GPT-OSS Tool Call Parser - Multiple Calls via parseEOS")
    func testGPTOSSParserEOSMultipleToolCalls() throws {
        let parser = GPTOSSToolCallParser()
        let content =
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\": \"Paris\"}<|call|>"
            + "<|start|>assistant<|channel|>commentary to=functions.get_time <|constrain|>json<|message|>{\"timezone\": \"UTC\"}"

        let toolCalls = parser.parseEOS(content, tools: nil)

        #expect(toolCalls.count == 2)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("Paris"))
        #expect(toolCalls[1].function.name == "get_time")
        #expect(toolCalls[1].function.arguments["timezone"] == .string("UTC"))
    }

    @Test("Test GPT-OSS Tool Call Parser - Malformed Input")
    func testGPTOSSParserMalformed() throws {
        let parser = GPTOSSToolCallParser()

        // No recipient
        #expect(
            parser.parse(
                content: "<|channel|>commentary <|constrain|>json<|message|>{\"a\": 1}<|call|>",
                tools: nil) == nil)

        // No <|message|> marker
        #expect(
            parser.parse(
                content: "<|channel|>commentary to=functions.get_weather json{\"a\": 1}",
                tools: nil) == nil)

        // Empty function name after namespace prefix
        #expect(
            parser.parse(
                content: "<|channel|>commentary to=functions.<|message|>{}",
                tools: nil) == nil)

        // Truncated JSON arguments
        #expect(
            parser.parse(
                content:
                    "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\": ",
                tools: nil) == nil)

        // Non-object JSON arguments
        #expect(
            parser.parse(
                content:
                    "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>[1, 2]<|call|>",
                tools: nil) == nil)
    }

    @Test("Test GPT-OSS Format via ToolCallProcessor - Analysis Preamble")
    func testGPTOSSFormatProcessor() throws {
        let processor = ToolCallProcessor(format: .gptOss)
        let chunks: [String] = [
            "<|channel|>", "analysis", "<|message|>", "Need", " weather", " data.", "<|end|>",
            "<|start|>", "assistant", "<|channel|>", "commentary", " to=functions.get_weather",
            " <|constrain|>json", "<|message|>", "{\"location\":", " \"Tokyo\"}",
        ]

        var text = ""
        for chunk in chunks {
            if let result = processor.processChunk(chunk) {
                text += result
            }
        }

        // Analysis channel content passes through as regular text, markers
        // intact (reasoning extraction is a separate concern, like <think>
        // tags in other formats).
        #expect(text == "<|channel|>analysis<|message|>Need weather data.<|end|><|start|>assistant")

        // <|call|> never arrives in text (EOS token), so the tool call stays
        // buffered until processEOS
        #expect(processor.toolCalls.count == 0)
        processor.processEOS()

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Test GPT-OSS Format via ToolCallProcessor - No Preamble")
    func testGPTOSSFormatProcessorNoPreamble() throws {
        let processor = ToolCallProcessor(format: .gptOss)
        let content =
            "<|channel|>commentary to=functions.search <|constrain|>json<|message|>{\"query\": \"AI news\"}"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 0)
        processor.processEOS()

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("AI news"))
    }

    @Test("Test GPT-OSS Format via ToolCallProcessor - Final Channel Response")
    func testGPTOSSFormatProcessorFinalChannel() throws {
        let processor = ToolCallProcessor(format: .gptOss)
        let chunks: [String] = [
            "<|channel|>", "final", "<|message|>", "The", " sky", " is", " blue.",
        ]

        var text = ""
        for chunk in chunks {
            if let result = processor.processChunk(chunk) {
                text += result
            }
        }
        if let residual = processor.processEOS(returnBufferedText: true) {
            text += residual
        }

        // A final-channel response contains no tool call and passes through
        // as regular text.
        #expect(processor.toolCalls.isEmpty)
        #expect(text == "<|channel|>final<|message|>The sky is blue.")
    }

    @Test("Test GPT-OSS Format via ToolCallProcessor - Multiple Tool Calls")
    func testGPTOSSFormatProcessorMultipleToolCalls() throws {
        let processor = ToolCallProcessor(format: .gptOss)
        let chunks: [String] = [
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\": \"Paris\"}",
            "<|call|>",
            "<|start|>assistant",
            "<|channel|>commentary to=functions.get_time <|constrain|>json<|message|>{\"timezone\": \"UTC\"}",
            "<|call|>",
        ]

        for chunk in chunks {
            _ = processor.processChunk(chunk)
        }
        processor.processEOS()

        #expect(processor.toolCalls.count == 2)

        let first = try #require(processor.toolCalls.first)
        #expect(first.function.name == "get_weather")
        #expect(first.function.arguments["location"] == .string("Paris"))

        let second = processor.toolCalls[1]
        #expect(second.function.name == "get_time")
        #expect(second.function.arguments["timezone"] == .string("UTC"))
    }

    @Test("Test GPT-OSS Format via ToolCallProcessor - Malformed Call Degrades to Text")
    func testGPTOSSFormatProcessorMalformedDegradesToText() throws {
        let processor = ToolCallProcessor(format: .gptOss)
        let content =
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\": "

        _ = processor.processChunk(content)
        let residual = processor.processEOS(returnBufferedText: true)

        #expect(processor.toolCalls.isEmpty)
        #expect(residual == content)
    }

    // MARK: - GLM-4-0414 Format Tests

    /// The `get_lab_result(test_name)` schema used to gate leading-name detection.
    private static func glm40414Tools() -> [[String: any Sendable]] {
        [
            [
                "type": "function",
                "function": [
                    "name": "get_lab_result",
                    "description": "Get the latest lab result for a patient.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "test_name": [
                                "type": "string",
                                "description": "Name of the lab test",
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["test_name"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]
    }

    @Test("Test GLM-4-0414 Tool Call Parser")
    func testGLM40414Parser() throws {
        let parser = GLM40414ToolCallParser()
        let content = "get_lab_result\n{\"test_name\": \"hemoglobin\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_lab_result")
        #expect(toolCall.function.arguments["test_name"] == .string("hemoglobin"))
    }

    @Test("Test GLM-4-0414 Tool Call Parser - No Arguments")
    func testGLM40414ParserNoArguments() throws {
        let parser = GLM40414ToolCallParser()

        let toolCall = try #require(parser.parse(content: "list_patients\n{}", tools: nil))

        #expect(toolCall.function.name == "list_patients")
        #expect(toolCall.function.arguments.isEmpty)
    }

    @Test("Test GLM-4-0414 Tool Call Parser - Name Must Match Declared Tool")
    func testGLM40414ParserNameMatchesTools() throws {
        let parser = GLM40414ToolCallParser()
        let tools = Self.glm40414Tools()

        // Declared tool parses.
        let toolCall = try #require(
            parser.parse(content: "get_lab_result\n{\"test_name\": \"sodium\"}", tools: tools))
        #expect(toolCall.function.name == "get_lab_result")
        #expect(toolCall.function.arguments["test_name"] == .string("sodium"))

        // Undeclared name is rejected when schemas are available.
        #expect(
            parser.parse(content: "drop_table\n{\"name\": \"patients\"}", tools: tools) == nil)
    }

    @Test("Test GLM-4-0414 Tool Call Parser - Multiple Calls via parseEOS")
    func testGLM40414ParserMultipleCallsEOS() throws {
        let parser = GLM40414ToolCallParser()
        let content =
            "get_weather\n{\"location\": \"Paris\"}"
            + "<|assistant|>get_time\n{\"timezone\": \"UTC\"}"

        let toolCalls = parser.parseEOS(content, tools: nil)

        #expect(toolCalls.count == 2)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("Paris"))
        #expect(toolCalls[1].function.name == "get_time")
        #expect(toolCalls[1].function.arguments["timezone"] == .string("UTC"))
    }

    @Test("Test GLM-4-0414 Tool Call Parser - Malformed Input")
    func testGLM40414ParserMalformed() throws {
        let parser = GLM40414ToolCallParser()

        // Plain text, no JSON object.
        #expect(parser.parse(content: "The patient looks stable.", tools: nil) == nil)

        // Leading line is not a bare identifier (multi-word prose before the JSON).
        #expect(parser.parse(content: "please call {\"a\": 1}", tools: nil) == nil)

        // Truncated JSON arguments.
        #expect(parser.parse(content: "get_lab_result\n{\"test_name\": ", tools: nil) == nil)

        // Non-object JSON arguments.
        #expect(parser.parse(content: "get_lab_result\n[1, 2]", tools: nil) == nil)
    }

    @Test("Test GLM-4-0414 Format via ToolCallProcessor - Streamed Call")
    func testGLM40414FormatProcessorStreamed() throws {
        let processor = ToolCallProcessor(format: .glm40414, tools: Self.glm40414Tools())
        let chunks: [String] = [
            "get_lab_result", "\n", "{\"test_name\":", " \"hemoglobin\"", "}",
        ]

        var text = ""
        for chunk in chunks {
            if let result = processor.processChunk(chunk) {
                text += result
            }
        }
        if let residual = processor.processEOS(returnBufferedText: true) {
            text += residual
        }

        // The function name is consumed by the call, not leaked as text.
        #expect(text.isEmpty)
        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_lab_result")
        #expect(toolCall.function.arguments["test_name"] == .string("hemoglobin"))
    }

    @Test("Test GLM-4-0414 Format via ToolCallProcessor - Single Chunk")
    func testGLM40414FormatProcessorSingleChunk() throws {
        let processor = ToolCallProcessor(format: .glm40414, tools: Self.glm40414Tools())

        let residual = processor.processChunk("get_lab_result\n{\"test_name\": \"glucose\"}")
        processor.processEOS()

        #expect(residual == nil)
        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_lab_result")
        #expect(toolCall.function.arguments["test_name"] == .string("glucose"))
    }

    @Test("Test GLM-4-0414 Format via ToolCallProcessor - No Arguments")
    func testGLM40414FormatProcessorNoArguments() throws {
        let processor = ToolCallProcessor(format: .glm40414)

        _ = processor.processChunk("list_patients\n{}")
        processor.processEOS()

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "list_patients")
        #expect(toolCall.function.arguments.isEmpty)
    }

    @Test("Test GLM-4-0414 Format via ToolCallProcessor - Plain Text Passes Through")
    func testGLM40414FormatProcessorPlainText() throws {
        let processor = ToolCallProcessor(format: .glm40414, tools: Self.glm40414Tools())
        let chunks: [String] = [
            "Check", " renal", " function", " before", " prescribing", " metformin.",
        ]

        var text = ""
        for chunk in chunks {
            if let result = processor.processChunk(chunk) {
                text += result
            }
        }
        if let residual = processor.processEOS(returnBufferedText: true) {
            text += residual
        }

        #expect(processor.toolCalls.isEmpty)
        #expect(text == "Check renal function before prescribing metformin.")
    }

    @Test("Test GLM-4-0414 Format via ToolCallProcessor - Malformed Call Degrades to Text")
    func testGLM40414FormatProcessorMalformedDegradesToText() throws {
        let processor = ToolCallProcessor(format: .glm40414, tools: Self.glm40414Tools())
        // Name confirmed but the JSON never closes (generation cut off).
        let content = "get_lab_result\n{\"test_name\": "

        _ = processor.processChunk(content)
        let residual = processor.processEOS(returnBufferedText: true)

        #expect(processor.toolCalls.isEmpty)
        #expect(residual == content)
    }

    @Test("Test GLM-4-0414 Format via ToolCallProcessor - Multiple Streamed Calls")
    func testGLM40414FormatProcessorMultipleCalls() throws {
        let processor = ToolCallProcessor(format: .glm40414)
        let chunks: [String] = [
            "get_weather\n{\"location\": \"Paris\"}",
            "<|assistant|>get_time\n{\"timezone\": \"UTC\"}",
        ]

        for chunk in chunks {
            _ = processor.processChunk(chunk)
        }
        processor.processEOS()

        #expect(processor.toolCalls.count == 2)
        let first = try #require(processor.toolCalls.first)
        #expect(first.function.name == "get_weather")
        #expect(first.function.arguments["location"] == .string("Paris"))
        let second = processor.toolCalls[1]
        #expect(second.function.name == "get_time")
        #expect(second.function.arguments["timezone"] == .string("UTC"))
    }

    // MARK: - Kimi K2 Format Tests

    @Test("Test Kimi K2 Tool Call Parser")
    func testKimiK2Parser() throws {
        let parser = KimiK2ToolCallParser()
        let content =
            "<|tool_calls_section_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"London\"}<|tool_calls_section_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("London"))
    }

    @Test("Test Kimi K2 Format via ToolCallProcessor")
    func testKimiK2FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .kimiK2)
        let content =
            "<|tool_calls_section_begin|>functions.search:0<|tool_call_argument_begin|>{\"query\": \"swift\"}<|tool_calls_section_end|>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("swift"))
    }

    // MARK: - MiniMax M2 Format Tests

    @Test("Test MiniMax M2 Tool Call Parser")
    func testMiniMaxM2Parser() throws {
        let parser = MiniMaxM2ToolCallParser()
        let content =
            "<minimax:tool_call><invoke name=\"get_weather\"><parameter name=\"location\">Sydney</parameter></invoke></minimax:tool_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Sydney"))
    }

    @Test("Test MiniMax M2 Format via ToolCallProcessor")
    func testMiniMaxM2FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .minimaxM2)
        let content =
            "<minimax:tool_call><invoke name=\"search\"><parameter name=\"query\">AI news</parameter></invoke></minimax:tool_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("AI news"))
    }

    // MARK: - Llama 3 Format Tests

    @Test("Test Llama 3 Tool Call Parser")
    func testLlama3Parser() throws {
        let parser = Llama3ToolCallParser()

        let content1 = """
            <|python_tag|>{"name": "knowledge_search", "parameters": {"query": "example"}}
            """

        let toolCall1 = try #require(parser.parse(content: content1, tools: nil))
        #expect(toolCall1.function.name == "knowledge_search")
        #expect(toolCall1.function.arguments["query"] == .string("example"))

        let content2 = """
            {"name": "get_weather", "arguments": {"location": "Tokyo"}}
            """

        let toolCall2 = try #require(parser.parse(content: content2, tools: nil))
        #expect(toolCall2.function.name == "get_weather")
        #expect(toolCall2.function.arguments["location"] == .string("Tokyo"))

        // Pythonic format
        let content3 = """
            <|python_tag|>get_weather(location="San Francisco, CA")
            """

        let toolCall3 = try #require(parser.parse(content: content3, tools: nil))
        #expect(toolCall3.function.name == "get_weather")
        #expect(toolCall3.function.arguments["location"] == .string("San Francisco, CA"))

        // Multiple arguments Pythonic
        let content4 = """
            <|python_tag|>calculate(expression="2 + 2", precision=4)
            """

        let toolCall4 = try #require(parser.parse(content: content4, tools: nil))
        #expect(toolCall4.function.name == "calculate")
        #expect(toolCall4.function.arguments["expression"] == .string("2 + 2"))
        #expect(toolCall4.function.arguments["precision"] == .string("4"))

        // Multiple JSON list format via parseEOS
        let content5 = """
            <|python_tag|>[
              {"name": "get_weather", "parameters": {"location": "New York"}},
              {"name": "get_time", "parameters": {"location": "London"}}
            ]
            """
        let toolCalls5 = parser.parseEOS(content5, tools: nil)
        #expect(toolCalls5.count == 2)
        #expect(toolCalls5[0].function.name == "get_weather")
        #expect(toolCalls5[0].function.arguments["location"] == .string("New York"))
        #expect(toolCalls5[1].function.name == "get_time")
        #expect(toolCalls5[1].function.arguments["location"] == .string("London"))

        // Multiple pythonic format via parseEOS
        let content6 = """
            <|python_tag|>[get_weather(location="New York"), get_time(location="London")]
            """
        let toolCalls6 = parser.parseEOS(content6, tools: nil)
        #expect(toolCalls6.count == 2)
        #expect(toolCalls6[0].function.name == "get_weather")
        #expect(toolCalls6[0].function.arguments["location"] == .string("New York"))
        #expect(toolCalls6[1].function.name == "get_time")
        #expect(toolCalls6[1].function.arguments["location"] == .string("London"))
    }

    // MARK: - ToolCallFormat Serialization Tests

    @Test("Test ToolCallFormat Raw Values for Serialization")
    func testToolCallFormatRawValues() throws {
        // Test that raw values are suitable for JSON/CLI serialization
        #expect(ToolCallFormat.json.rawValue == "json")
        #expect(ToolCallFormat.lfm2.rawValue == "lfm2")
        #expect(ToolCallFormat.xmlFunction.rawValue == "xml_function")
        #expect(ToolCallFormat.glm4.rawValue == "glm4")
        #expect(ToolCallFormat.glm40414.rawValue == "glm4_0414")
        #expect(ToolCallFormat.gemma.rawValue == "gemma")
        #expect(ToolCallFormat.gptOss.rawValue == "gpt_oss")
        #expect(ToolCallFormat.kimiK2.rawValue == "kimi_k2")
        #expect(ToolCallFormat.minimaxM2.rawValue == "minimax_m2")
        #expect(ToolCallFormat.mistral.rawValue == "mistral")

        // Test round-trip via raw value
        for format in ToolCallFormat.allCases {
            #expect(ToolCallFormat(rawValue: format.rawValue) == format)
        }
    }

    // MARK: - Format Inference Tests

    @Test("Test ToolCallFormat Inference from Model Type")
    func testToolCallFormatInference() throws {
        // LFM2 models (prefix matching)
        #expect(ToolCallFormat.infer(from: "lfm2") == .lfm2)
        #expect(ToolCallFormat.infer(from: "LFM2") == .lfm2)
        #expect(ToolCallFormat.infer(from: "lfm2_moe") == .lfm2)
        #expect(ToolCallFormat.infer(from: "lfm2_5") == .lfm2)
        #expect(ToolCallFormat.infer(from: "LFM2_5") == .lfm2)
        #expect(ToolCallFormat.infer(from: "lfm25") == .lfm2)

        // GLM-4-0414 dense models: model_type is exactly "glm4"
        #expect(ToolCallFormat.infer(from: "glm4") == .glm40414)
        #expect(ToolCallFormat.infer(from: "GLM4") == .glm40414)
        // GLM-4.5/4.6 MoE and other glm4* variants keep the tagged .glm4 format
        #expect(ToolCallFormat.infer(from: "glm4_moe") == .glm4)
        #expect(ToolCallFormat.infer(from: "glm4_moe_lite") == .glm4)
        #expect(ToolCallFormat.infer(from: "glm4_5") == .glm4)
        #expect(ToolCallFormat.infer(from: "GLM4_5") == .glm4)

        // Gemma models
        #expect(ToolCallFormat.infer(from: "gemma") == .gemma)
        #expect(ToolCallFormat.infer(from: "GEMMA") == .gemma)

        // GPT-OSS models (prefix matching)
        #expect(ToolCallFormat.infer(from: "gpt_oss") == .gptOss)
        #expect(ToolCallFormat.infer(from: "GPT_OSS") == .gptOss)
        #expect(ToolCallFormat.infer(from: "gpt_oss_moe") == .gptOss)

        // Nemotron models (prefix matching)
        #expect(ToolCallFormat.infer(from: "nemotron_h") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "NEMOTRON_H") == .xmlFunction)

        // Qwen3.5 models (prefix matching)
        #expect(ToolCallFormat.infer(from: "qwen3_5") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "qwen3_5_moe") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "QWEN3_5") == .xmlFunction)

        // Qwen3-Next models (prefix matching)
        #expect(ToolCallFormat.infer(from: "qwen3_next") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "qwen3_next_moe") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "QWEN3_NEXT") == .xmlFunction)

        // Mistral3 models (prefix matching)
        #expect(ToolCallFormat.infer(from: "mistral3") == .mistral)
        #expect(ToolCallFormat.infer(from: "Mistral3") == .mistral)
        #expect(ToolCallFormat.infer(from: "mistral3_text") == .mistral)

        // Llama models - require secondary signals from configData
        #expect(ToolCallFormat.infer(from: "llama") == nil)  // Should be nil without configData

        let llama3RopeConfig = """
            {
                "model_type": "llama",
                "rope_scaling": {
                    "rope_type": "llama3"
                }
            }
            """.data(using: .utf8)!
        #expect(ToolCallFormat.infer(from: "llama", configData: llama3RopeConfig) == .llama3)

        let llama3VocabConfig = """
            {
                "model_type": "llama",
                "vocab_size": 128256
            }
            """.data(using: .utf8)!
        #expect(ToolCallFormat.infer(from: "LLAMA", configData: llama3VocabConfig) == .llama3)

        let llama2Config = """
            {
                "model_type": "llama",
                "vocab_size": 32000
            }
            """.data(using: .utf8)!
        #expect(ToolCallFormat.infer(from: "llama", configData: llama2Config) == nil)

        // Unknown models should return nil (use default JSON format)
        #expect(ToolCallFormat.infer(from: "qwen2") == nil)
        #expect(ToolCallFormat.infer(from: "mistral") == nil)
    }

    // MARK: - Mistral Format Tests

    @Test("Test Mistral Tool Call Parser")
    func testMistralParser() throws {
        let parser = MistralToolCallParser()
        let content = "[TOOL_CALLS]get_weather [ARGS]{\"location\": \"Paris\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test("Test Mistral Tool Call Parser - With Call ID")
    func testMistralParserWithCallId() throws {
        let parser = MistralToolCallParser()
        let content = "[TOOL_CALLS]get_weather[CALL_ID]abc123xyz[ARGS]{\"location\": \"Paris\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.id == "abc123xyz")
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test("Test Mistral Tool Call Parser - Preserves [TOOL_CALLS] in Arguments")
    func testMistralParserPreservesStartTagInArguments() throws {
        let parser = MistralToolCallParser()
        let content = "get_note[ARGS]{\"text\": \"literal [TOOL_CALLS] marker\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_note")
        #expect(toolCall.function.arguments["text"] == .string("literal [TOOL_CALLS] marker"))
    }

    @Test("Test Mistral Tool Call Parser - Preserves </s> in Arguments")
    func testMistralParserPreservesEndTagInArguments() throws {
        let parser = MistralToolCallParser()
        let content = "get_note[ARGS]{\"text\": \"literal </s> marker\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_note")
        #expect(toolCall.function.arguments["text"] == .string("literal </s> marker"))
    }

    @Test("Test Mistral Format via ToolCallProcessor")
    func testMistralFormatProcessor() throws {
        let processor = ToolCallProcessor(format: .mistral)
        let chunks: [String] = [
            "[TOOL", "_CALLS]", "get_weather", " [ARGS]",
            "{\"location\":", " \"Tokyo\"}",
        ]

        for chunk in chunks {
            _ = processor.processChunk(chunk)
        }

        // End tag never arrives in text, so tool call stays buffered until processEOS
        #expect(processor.toolCalls.count == 0)
        processor.processEOS()

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Test Mistral Format Processor EOS")
    func testMistralFormatProcessorEOS() throws {
        let processor = ToolCallProcessor(format: .mistral)
        let content = "[TOOL_CALLS]get_weather [ARGS]{\"location\": \"Berlin\"}"

        _ = processor.processChunk(content)

        // Before processEOS, no tool calls extracted (end tag never arrives)
        #expect(processor.toolCalls.count == 0)

        // processEOS extracts the buffered tool call
        processor.processEOS()

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Berlin"))
    }

    @Test("Test Mistral Format Processor Multiple Tool Calls")
    func testMistralFormatProcessorMultipleToolCalls() throws {
        let processor = ToolCallProcessor(format: .mistral)
        let chunks: [String] = [
            "[TOOL_CALLS]get_weather[ARGS]",
            "{\"location\": \"Paris\"}",
            "[TOOL_CALLS]get_time",
            "[ARGS]{\"timezone\": \"UTC\"}",
        ]

        for chunk in chunks {
            let result = processor.processChunk(chunk)
            // All chunks should be buffered (nil) after the start tag
            if chunk == chunks.first {
                #expect(result == nil)
            }
        }

        // No tool calls before processEOS
        #expect(processor.toolCalls.count == 0)
        processor.processEOS()

        // Both tool calls should be extracted
        #expect(processor.toolCalls.count == 2)

        let first = try #require(processor.toolCalls.first)
        #expect(first.function.name == "get_weather")
        #expect(first.function.arguments["location"] == .string("Paris"))

        let second = processor.toolCalls[1]
        #expect(second.function.name == "get_time")
        #expect(second.function.arguments["timezone"] == .string("UTC"))
    }
}
