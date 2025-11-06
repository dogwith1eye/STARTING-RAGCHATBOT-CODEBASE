"""
Unit tests for search_tools.py

Tests the CourseSearchTool and ToolManager to verify:
1. Correct execution of the search tool with various parameters
2. Proper handling of MAX_RESULTS configuration (including the bug where MAX_RESULTS=0)
3. Error handling for missing courses
4. Source tracking functionality
5. Result formatting
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method"""

    def test_execute_with_results(self, mock_vector_store, sample_search_results):
        """Test that execute returns formatted results when search succeeds"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="what is MCP")

        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="what is MCP", course_name=None, lesson_number=None
        )

        # Verify result is formatted correctly
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Introduction to MCP" in result
        assert "MCP stands for Model Context Protocol" in result

    def test_execute_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test that execute passes course_name filter correctly"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="what is MCP", course_name="Introduction to MCP")

        # Verify search was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="what is MCP", course_name="Introduction to MCP", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test that execute passes lesson_number filter correctly"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="what is MCP", lesson_number=1)

        # Verify search was called with lesson filter
        mock_vector_store.search.assert_called_once_with(
            query="what is MCP", course_name=None, lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test that execute passes both course and lesson filters"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(
            query="what is MCP", course_name="Introduction to MCP", lesson_number=1
        )

        # Verify search was called with both filters
        mock_vector_store.search.assert_called_once_with(
            query="what is MCP", course_name="Introduction to MCP", lesson_number=1
        )

    def test_execute_with_empty_results(self, mock_vector_store, empty_search_results):
        """Test that execute handles empty results gracefully"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="nonexistent topic")

        # Should return helpful message
        assert "No relevant content found" in result

    def test_execute_with_empty_results_and_filters(
        self, mock_vector_store, empty_search_results
    ):
        """Test that execute includes filter info in empty results message"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(
            query="topic", course_name="Introduction to MCP", lesson_number=2
        )

        # Should mention the filters in the message
        assert "No relevant content found" in result
        assert "Introduction to MCP" in result
        assert "lesson 2" in result

    def test_execute_with_error(self, mock_vector_store, error_search_results):
        """Test that execute returns error message when search fails"""
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="topic", course_name="NonExistent Course")

        # Should return the error message
        assert "No course found matching 'NonExistent Course'" in result

    def test_execute_tracks_sources(self, mock_vector_store, sample_search_results):
        """Test that execute properly tracks sources for UI display"""
        tool = CourseSearchTool(mock_vector_store)

        # Initially no sources
        assert tool.last_sources == []

        result = tool.execute(query="what is MCP")

        # After execution, sources should be populated
        assert len(tool.last_sources) > 0
        assert tool.last_sources[0]["course_title"] == "Introduction to MCP"
        assert "lesson_number" in tool.last_sources[0]

    def test_execute_fetches_links(self, mock_vector_store, sample_search_results):
        """Test that execute fetches and includes course/lesson links in sources"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="what is MCP")

        # Verify link methods were called
        mock_vector_store.get_course_link.assert_called()
        mock_vector_store.get_lesson_link.assert_called()

        # Verify links are in sources
        assert any("course_link" in source for source in tool.last_sources)

    def test_execute_formats_results_correctly(self, mock_vector_store):
        """Test that execute formats results with proper headers"""
        # Create specific search results
        test_results = SearchResults(
            documents=["Content from lesson 1"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0}
            ],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = test_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")

        # Should have course title and lesson number in header
        assert "[Test Course - Lesson 1]" in result
        assert "Content from lesson 1" in result

    def test_execute_with_zero_max_results_bug(
        self, mock_vector_store_with_zero_max_results
    ):
        """
        CRITICAL TEST: Reproduces the MAX_RESULTS=0 bug

        When MAX_RESULTS=0, ChromaDB returns 0 results, causing
        the tool to return "No relevant content found" for all queries.
        """
        tool = CourseSearchTool(mock_vector_store_with_zero_max_results)

        # This should return results, but with max_results=0 it won't
        result = tool.execute(query="what is MCP")

        # With the bug, this will be an empty result message
        assert "No relevant content found" in result

        # Verify the vector store's max_results is indeed 0
        assert mock_vector_store_with_zero_max_results.max_results == 0


class TestCourseOutlineTool:
    """Tests for CourseOutlineTool"""

    def test_get_tool_definition(self):
        """Test that outline tool returns correct definition"""
        mock_store = Mock()
        tool = CourseOutlineTool(mock_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "course_name" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_name"]

    def test_execute_with_valid_course(self, mock_vector_store):
        """Test outline tool with valid course"""
        import json

        # Mock the catalog response
        mock_vector_store.course_catalog.get.return_value = {
            "metadatas": [
                {
                    "title": "Introduction to MCP",
                    "course_link": "https://example.com/mcp",
                    "instructor": "Test Instructor",
                    "lessons_json": json.dumps(
                        [
                            {"lesson_number": 0, "lesson_title": "Getting Started"},
                            {"lesson_number": 1, "lesson_title": "Core Concepts"},
                        ]
                    ),
                }
            ]
        }

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="MCP")

        # Verify output format
        assert "Introduction to MCP" in result
        assert "Test Instructor" in result
        assert "Lesson 0: Getting Started" in result
        assert "Lesson 1: Core Concepts" in result

    def test_execute_with_invalid_course(self, mock_vector_store):
        """Test outline tool with course that doesn't exist"""
        mock_vector_store._resolve_course_name.return_value = None

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="Nonexistent Course")

        assert "No course found matching" in result


class TestToolManager:
    """Tests for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test that tools can be registered"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        # Verify tool is registered
        assert "search_course_content" in manager.tools

    def test_register_multiple_tools(self, mock_vector_store):
        """Test that multiple tools can be registered"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        # Verify both tools are registered
        assert "search_course_content" in manager.tools
        assert "get_course_outline" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test that tool definitions are returned correctly"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
        assert "input_schema" in definitions[0]

    def test_execute_tool(self, mock_vector_store, sample_search_results):
        """Test that tools can be executed by name"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="what is MCP")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_nonexistent_tool(self):
        """Test that executing nonexistent tool returns error"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test that last sources can be retrieved from tools"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute a search to populate sources
        manager.execute_tool("search_course_content", query="what is MCP")

        sources = manager.get_last_sources()

        assert len(sources) > 0
        assert "course_title" in sources[0]

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test that sources can be reset"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute a search to populate sources
        manager.execute_tool("search_course_content", query="what is MCP")
        assert len(manager.get_last_sources()) > 0

        # Reset sources
        manager.reset_sources()

        # Verify sources are cleared
        assert len(manager.get_last_sources()) == 0


class TestCourseSearchToolDefinition:
    """Tests for CourseSearchTool.get_tool_definition()"""

    def test_get_tool_definition_structure(self, mock_vector_store):
        """Test that tool definition has correct structure"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        # Verify schema structure
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_get_tool_definition_parameters(self, mock_vector_store):
        """Test that tool definition has correct parameters"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        properties = definition["input_schema"]["properties"]

        # Verify all expected parameters
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties

        # Verify only query is required
        assert definition["input_schema"]["required"] == ["query"]

    def test_get_tool_definition_parameter_types(self, mock_vector_store):
        """Test that parameters have correct types"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        properties = definition["input_schema"]["properties"]

        assert properties["query"]["type"] == "string"
        assert properties["course_name"]["type"] == "string"
        assert properties["lesson_number"]["type"] == "integer"
