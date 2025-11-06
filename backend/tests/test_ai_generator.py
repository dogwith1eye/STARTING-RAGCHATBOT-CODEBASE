"""
Unit tests for ai_generator.py

Tests the AIGenerator to verify:
1. Correct integration with Claude API
2. Proper tool calling flow
3. Tool execution handling
4. Conversation history management
5. Response generation with and without tools
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from ai_generator import AIGenerator


@pytest.fixture
def mock_tool_manager():
    """Fixture providing a mock tool manager"""
    mock = Mock()
    mock.execute_tool.return_value = "Tool execution result"
    return mock


class TestAIGeneratorInit:
    """Tests for AIGenerator initialization"""

    def test_init_with_api_key(self):
        """Test that AIGenerator initializes with API key"""
        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        assert generator.model == "claude-sonnet-4-20250514"
        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch("ai_generator.anthropic.Anthropic")
    def test_init_creates_anthropic_client(self, mock_anthropic_class):
        """Test that AIGenerator creates Anthropic client"""
        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        mock_anthropic_class.assert_called_once_with(api_key="test_key")


class TestGenerateResponseWithoutTools:
    """Tests for generate_response() without tool use"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_simple_query(self, mock_anthropic_class):
        """Test basic response generation without tools"""
        # Set up mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "This is a test response."
        mock_response.content = [mock_content]

        mock_client.messages.create.return_value = mock_response

        # Test
        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(query="What is 2+2?")

        # Verify
        assert result == "This is a test response."
        mock_client.messages.create.assert_called_once()

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_includes_system_prompt(self, mock_anthropic_class):
        """Test that system prompt is included in API call"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]

        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")
        generator.generate_response(query="Test query")

        # Get the call arguments
        call_args = mock_client.messages.create.call_args

        # Verify system prompt is included
        assert "system" in call_args.kwargs
        assert "educational content" in call_args.kwargs["system"].lower()

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test that conversation history is included in system prompt"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]

        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")
        history = "User: Previous question\nAssistant: Previous answer"
        generator.generate_response(query="New query", conversation_history=history)

        # Verify history is in system prompt
        call_args = mock_client.messages.create.call_args
        assert "Previous question" in call_args.kwargs["system"]


class TestGenerateResponseWithTools:
    """Tests for generate_response() with tool calling"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_includes_tools(
        self, mock_anthropic_class, mock_tool_manager
    ):
        """Test that tools are passed to Claude API when tool_manager is provided"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]

        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        tool_definitions = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }
        ]

        generator.generate_response(
            query="What is MCP?", tools=tool_definitions, tool_manager=mock_tool_manager
        )

        # Verify tools are in the API call
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tools"] == tool_definitions
        assert "tool_choice" in call_args.kwargs
        assert call_args.kwargs["tool_choice"]["type"] == "auto"

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_handles_tool_use(
        self, mock_anthropic_class, mock_tool_manager
    ):
        """
        CRITICAL TEST: Verify that AIGenerator correctly handles tool_use stop_reason
        and executes tools via ToolManager
        """
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: Claude wants to use a tool
        mock_tool_use_response = Mock()
        mock_tool_use_response.stop_reason = "tool_use"

        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "what is MCP"}

        mock_tool_use_response.content = [mock_tool_block]

        # Second response: Claude's final answer
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_content = Mock()
        mock_final_content.text = "MCP is a protocol for AI communication."
        mock_final_response.content = [mock_final_content]

        # Set up the mock to return tool_use first, then final response
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response,
        ]

        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        tool_definitions = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            }
        ]

        result = generator.generate_response(
            query="What is MCP?", tools=tool_definitions, tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="what is MCP"
        )

        # Verify final result is returned
        assert result == "MCP is a protocol for AI communication."

        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_without_tool_manager_ignores_tool_use(
        self, mock_anthropic_class
    ):
        """Test that tool_use is handled gracefully if no tool_manager provided"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.text = "Tool block text"  # Add text attribute
        mock_response.content = [mock_tool_block]

        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        # Call without tool_manager - should handle gracefully
        result = generator.generate_response(
            query="What is MCP?", tools=[{"name": "test"}]
        )

        # Should return text from response (not crash)
        assert result == "Tool block text"


class TestMultiRoundToolExecution:
    """Tests for multi-round tool execution with new loop-based implementation"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_single_tool_round_preserves_tools_parameter(
        self, mock_anthropic_class, mock_tool_manager
    ):
        """Test that tools parameter is included in first round API call"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response with tool use
        mock_tool_use_response = Mock()
        mock_tool_use_response.stop_reason = "tool_use"

        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_abc"
        mock_tool_block.input = {"query": "test query"}

        mock_tool_use_response.content = [mock_tool_block]

        # Second response (after tool execution)
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_content = Mock()
        mock_final_content.text = "Final answer"
        mock_final_response.content = [mock_final_content]

        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response,
        ]

        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Verify first API call includes tools
        first_call_args = mock_client.messages.create.call_args_list[0]
        assert "tools" in first_call_args.kwargs
        assert first_call_args.kwargs["tools"] == [{"name": "search_course_content"}]

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once()

        # Verify final result
        assert result == "Final answer"

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_sequential_tool_rounds(self, mock_anthropic_class, mock_tool_manager):
        """Test that Claude can make two sequential tool calls"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        mock_tool_use_1 = Mock()
        mock_tool_use_1.stop_reason = "tool_use"
        mock_tool_block_1 = Mock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.name = "search_course_content"
        mock_tool_block_1.id = "tool_1"
        mock_tool_block_1.input = {"query": "MCP servers", "course_name": "MCP"}
        mock_tool_use_1.content = [mock_tool_block_1]

        # Second response: another tool use
        mock_tool_use_2 = Mock()
        mock_tool_use_2.stop_reason = "tool_use"
        mock_tool_block_2 = Mock()
        mock_tool_block_2.type = "tool_use"
        mock_tool_block_2.name = "get_course_outline"
        mock_tool_block_2.id = "tool_2"
        mock_tool_block_2.input = {"course_name": "MCP"}
        mock_tool_use_2.content = [mock_tool_block_2]

        # Third response: final answer (forced by MAX_ROUNDS)
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_content = Mock()
        mock_final_content.text = "MCP servers are covered in Lesson 2"
        mock_final_response.content = [mock_final_content]

        mock_client.messages.create.side_effect = [
            mock_tool_use_1,
            mock_tool_use_2,
            mock_final_response,
        ]

        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(
            query="What are MCP servers and where are they in the course?",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager,
        )

        # Verify 3 API calls were made
        assert mock_client.messages.create.call_count == 3

        # Verify tools were executed twice
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify both API calls include tools (first two calls)
        first_call = mock_client.messages.create.call_args_list[0]
        second_call = mock_client.messages.create.call_args_list[1]
        assert "tools" in first_call.kwargs
        assert "tools" in second_call.kwargs

        # Verify third call (final) does NOT include tools (forced synthesis)
        third_call = mock_client.messages.create.call_args_list[2]
        assert "tools" not in third_call.kwargs

        # Verify result
        assert result == "MCP servers are covered in Lesson 2"

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_rounds_forces_synthesis(self, mock_anthropic_class, mock_tool_manager):
        """Test that reaching MAX_ROUNDS forces final answer without tools"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Both responses request tool use
        mock_tool_use = Mock()
        mock_tool_use.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_x"
        mock_tool_block.input = {"query": "test"}
        mock_tool_use.content = [mock_tool_block]

        # Final response
        mock_final = Mock()
        mock_final.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Based on available information..."
        mock_final.content = [mock_content]

        # Return tool_use twice, then final answer
        mock_client.messages.create.side_effect = [
            mock_tool_use,
            mock_tool_use,
            mock_final,
        ]

        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(
            query="Complex query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Should make 3 API calls (2 tool rounds + 1 forced final)
        assert mock_client.messages.create.call_count == 3

        # Should execute tools exactly 2 times (MAX_ROUNDS)
        assert mock_tool_manager.execute_tool.call_count == 2

        # Final call should NOT have tools parameter
        final_call = mock_client.messages.create.call_args_list[2]
        assert "tools" not in final_call.kwargs

        assert result == "Based on available information..."

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_error_handling(
        self, mock_anthropic_class, mock_tool_manager
    ):
        """Test that tool execution errors are handled gracefully"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        mock_tool_use = Mock()
        mock_tool_use.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_err"
        mock_tool_block.input = {"query": "test"}
        mock_tool_use.content = [mock_tool_block]

        # Second response after error
        mock_final = Mock()
        mock_final.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "I encountered an error searching"
        mock_final.content = [mock_content]

        mock_client.messages.create.side_effect = [mock_tool_use, mock_final]

        # Make tool execution raise an exception
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")

        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(
            query="Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Should still return a result (not crash)
        assert result == "I encountered an error searching"

        # Should have made 2 API calls
        assert mock_client.messages.create.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_message_accumulation_across_rounds(
        self, mock_anthropic_class, mock_tool_manager
    ):
        """Test that messages accumulate correctly across multiple rounds"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First tool use
        mock_tool_use_1 = Mock()
        mock_tool_use_1.stop_reason = "tool_use"
        mock_block_1 = Mock()
        mock_block_1.type = "tool_use"
        mock_block_1.name = "tool1"
        mock_block_1.id = "id1"
        mock_block_1.input = {"param": "value1"}
        mock_tool_use_1.content = [mock_block_1]

        # Second tool use
        mock_tool_use_2 = Mock()
        mock_tool_use_2.stop_reason = "tool_use"
        mock_block_2 = Mock()
        mock_block_2.type = "tool_use"
        mock_block_2.name = "tool2"
        mock_block_2.id = "id2"
        mock_block_2.input = {"param": "value2"}
        mock_tool_use_2.content = [mock_block_2]

        # Final response
        mock_final = Mock()
        mock_final.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Final"
        mock_final.content = [mock_content]

        mock_client.messages.create.side_effect = [
            mock_tool_use_1,
            mock_tool_use_2,
            mock_final,
        ]

        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(
            query="Test",
            tools=[{"name": "tool1"}, {"name": "tool2"}],
            tool_manager=mock_tool_manager,
        )

        # Verify 3 API calls were made (2 tool rounds + 1 final)
        assert mock_client.messages.create.call_count == 3

        # Verify tools were executed twice
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify result is returned correctly
        assert result == "Final"

        # Verify the pattern of API calls includes tools in first two calls
        first_call = mock_client.messages.create.call_args_list[0]
        second_call = mock_client.messages.create.call_args_list[1]
        third_call = mock_client.messages.create.call_args_list[2]

        # First two calls should have tools
        assert "tools" in first_call.kwargs
        assert "tools" in second_call.kwargs

        # Third call (forced final) should NOT have tools
        assert "tools" not in third_call.kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_natural_termination_before_max_rounds(
        self, mock_anthropic_class, mock_tool_manager
    ):
        """Test that Claude can naturally stop before reaching MAX_ROUNDS"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        mock_tool_use = Mock()
        mock_tool_use.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search"
        mock_tool_block.id = "tool1"
        mock_tool_block.input = {"query": "test"}
        mock_tool_use.content = [mock_tool_block]

        # Second response: natural end (no more tools needed)
        mock_final = Mock()
        mock_final.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Here's the answer"
        mock_final.content = [mock_content]

        mock_client.messages.create.side_effect = [mock_tool_use, mock_final]

        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(
            query="Simple query",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager,
        )

        # Should only make 2 API calls (not 3) - natural termination
        assert mock_client.messages.create.call_count == 2

        # Should only execute 1 tool
        assert mock_tool_manager.execute_tool.call_count == 1

        assert result == "Here's the answer"


class TestSystemPrompt:
    """Tests for SYSTEM_PROMPT content"""

    def test_system_prompt_mentions_tools(self):
        """Test that system prompt mentions tool usage"""
        assert "tool" in AIGenerator.SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_course_search(self):
        """Test that system prompt references course search"""
        assert "course" in AIGenerator.SYSTEM_PROMPT.lower()

    def test_system_prompt_is_static(self):
        """Test that SYSTEM_PROMPT is a class variable"""
        gen1 = AIGenerator(api_key="key1", model="model1")
        gen2 = AIGenerator(api_key="key2", model="model2")

        # Should be the same object (not copied)
        assert gen1.SYSTEM_PROMPT is gen2.SYSTEM_PROMPT
