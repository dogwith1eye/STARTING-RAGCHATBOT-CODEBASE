"""
Integration tests for rag_system.py

Tests the complete RAG system flow:
1. Query processing through the complete pipeline
2. Tool integration and execution
3. Session management
4. Content queries vs general queries
5. CRITICAL: Testing with MAX_RESULTS=0 bug
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from rag_system import RAGSystem


class TestRAGSystemInit:
    """Tests for RAGSystem initialization"""

    def test_init_with_correct_config(self, test_config):
        """Test that RAGSystem initializes all components correctly"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            rag = RAGSystem(test_config)

            # Verify vector store was initialized with correct max_results
            mock_vector_store.assert_called_once_with(
                test_config.CHROMA_PATH,
                test_config.EMBEDDING_MODEL,
                test_config.MAX_RESULTS,  # Should be 5
            )

    def test_init_with_zero_max_results_bug(self, test_config_zero_max_results):
        """
        CRITICAL TEST: RAGSystem initialized with MAX_RESULTS=0

        This reproduces the bug where MAX_RESULTS=0 is passed to VectorStore,
        causing all searches to return empty results.
        """
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            rag = RAGSystem(test_config_zero_max_results)

            # Verify the bug: vector store was initialized with max_results=0
            call_args = mock_vector_store.call_args
            assert call_args[0][2] == 0  # Third positional arg is max_results

    def test_init_registers_tools(self, test_config):
        """Test that search tools are registered during initialization"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            rag = RAGSystem(test_config)

            # Verify tools are registered
            assert "search_course_content" in rag.tool_manager.tools
            assert "get_course_outline" in rag.tool_manager.tools


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method"""

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_query_calls_ai_generator(
        self, mock_session, mock_ai_gen_class, mock_vector, mock_doc, test_config
    ):
        """Test that query calls AI generator with correct parameters"""
        # Set up mocks
        mock_ai_instance = Mock()
        mock_ai_gen_class.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "Test response"

        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_conversation_history.return_value = None

        rag = RAGSystem(test_config)

        # Execute query
        response, sources = rag.query("What is MCP?")

        # Verify AI generator was called
        mock_ai_instance.generate_response.assert_called_once()

        # Verify tools were passed
        call_args = mock_ai_instance.generate_response.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tool_manager"] == rag.tool_manager

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_query_with_session_includes_history(
        self, mock_session, mock_ai_gen_class, mock_vector, mock_doc, test_config
    ):
        """Test that query includes conversation history when session_id provided"""
        mock_ai_instance = Mock()
        mock_ai_gen_class.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "Response"

        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_conversation_history.return_value = (
            "User: Previous\nAssistant: Previous response"
        )

        rag = RAGSystem(test_config)

        response, sources = rag.query("Follow-up question", session_id="test_session")

        # Verify history was retrieved
        mock_session_instance.get_conversation_history.assert_called_once_with(
            "test_session"
        )

        # Verify history was passed to AI generator
        call_args = mock_ai_instance.generate_response.call_args
        assert "conversation_history" in call_args.kwargs
        assert "Previous" in call_args.kwargs["conversation_history"]

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_query_updates_session_history(
        self, mock_session, mock_ai_gen_class, mock_vector, mock_doc, test_config
    ):
        """Test that query updates session history after getting response"""
        mock_ai_instance = Mock()
        mock_ai_gen_class.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "AI response"

        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_conversation_history.return_value = None

        rag = RAGSystem(test_config)

        response, sources = rag.query("Test question", session_id="test_session")

        # Verify session was updated
        mock_session_instance.add_exchange.assert_called_once_with(
            "test_session", "Test question", "AI response"
        )

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_query_retrieves_sources_from_tool_manager(
        self, mock_session, mock_ai_gen_class, mock_vector, mock_doc, test_config
    ):
        """Test that query retrieves sources from tool manager after execution"""
        mock_ai_instance = Mock()
        mock_ai_gen_class.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "Response"

        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        rag = RAGSystem(test_config)

        # Mock the tool manager to return sources
        rag.tool_manager.get_last_sources = Mock(
            return_value=[{"course_title": "MCP", "lesson_number": 0}]
        )
        rag.tool_manager.reset_sources = Mock()

        response, sources = rag.query("What is MCP?")

        # Verify sources were retrieved
        rag.tool_manager.get_last_sources.assert_called_once()

        # Verify sources were returned
        assert len(sources) == 1
        assert sources[0]["course_title"] == "MCP"

        # Verify sources were reset
        rag.tool_manager.reset_sources.assert_called_once()


class TestRAGSystemQueryWithMaxResultsZero:
    """Tests specifically for the MAX_RESULTS=0 bug"""

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_query_with_max_results_zero_returns_no_results(
        self,
        mock_session,
        mock_ai_gen_class,
        mock_vector_class,
        mock_doc,
        test_config_zero_max_results,
        mock_vector_store_with_zero_max_results,
    ):
        """
        CRITICAL INTEGRATION TEST: Query with MAX_RESULTS=0

        This tests the complete flow when MAX_RESULTS=0:
        1. RAGSystem initializes with MAX_RESULTS=0
        2. VectorStore is created with max_results=0
        3. When Claude uses the search tool, it returns empty results
        4. User sees "query failed" or "no content found"
        """
        # Mock VectorStore to return our mock with max_results=0
        mock_vector_class.return_value = mock_vector_store_with_zero_max_results

        # Mock AI generator to simulate tool use
        mock_ai_instance = Mock()
        mock_ai_gen_class.return_value = mock_ai_instance

        # Simulate Claude calling the tool and getting empty results
        def mock_generate(
            query, conversation_history=None, tools=None, tool_manager=None
        ):
            if tool_manager:
                # Simulate Claude deciding to use the search tool
                # With max_results=0, this will return "No relevant content found"
                tool_result = tool_manager.execute_tool(
                    "search_course_content", query="what is MCP"
                )
                # Claude would then synthesize this into a response
                if "No relevant content found" in tool_result:
                    return "I couldn't find any information about that in the course materials."
            return "Test response"

        mock_ai_instance.generate_response.side_effect = mock_generate

        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        # Initialize RAG system with the buggy config
        rag = RAGSystem(test_config_zero_max_results)

        # Execute query that should trigger search
        response, sources = rag.query("What is MCP?")

        # With the bug, no sources will be returned
        assert len(sources) == 0

        # Response will indicate no content was found
        assert "couldn't find" in response.lower() or "no" in response.lower()


class TestRAGSystemQueryContentVsGeneral:
    """Tests to verify RAG system handles content queries vs general queries correctly"""

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_content_query_uses_search_tool(
        self, mock_session, mock_ai_gen_class, mock_vector, mock_doc, test_config
    ):
        """Test that content-specific queries trigger tool usage"""
        mock_ai_instance = Mock()
        mock_ai_gen_class.return_value = mock_ai_instance

        # Simulate Claude deciding to use search tool for content query
        def mock_generate(
            query, conversation_history=None, tools=None, tool_manager=None
        ):
            if "course materials" in query and tools and tool_manager:
                # Claude should use the search tool for course content
                tool_result = tool_manager.execute_tool(
                    "search_course_content", query="what is MCP"
                )
                return f"Based on the course materials: {tool_result}"
            return "General response"

        mock_ai_instance.generate_response.side_effect = mock_generate

        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        rag = RAGSystem(test_config)

        response, sources = rag.query("What is MCP according to the course?")

        # Verify tools were available
        call_args = mock_ai_instance.generate_response.call_args
        assert call_args.kwargs["tools"] is not None

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_general_query_may_not_use_tool(
        self, mock_session, mock_ai_gen_class, mock_vector, mock_doc, test_config
    ):
        """Test that general knowledge queries may not trigger tool usage"""
        mock_ai_instance = Mock()
        mock_ai_gen_class.return_value = mock_ai_instance

        # Simulate Claude answering without tools for general query
        def mock_generate(
            query, conversation_history=None, tools=None, tool_manager=None
        ):
            if "what is" in query.lower() and "course" not in query.lower():
                # General knowledge - Claude might not use tools
                return "General knowledge answer without searching"
            return "Response"

        mock_ai_instance.generate_response.side_effect = mock_generate

        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        rag = RAGSystem(test_config)

        response, sources = rag.query("What is the capital of France?")

        # General queries might not produce sources
        # (This depends on Claude's decision)
        assert isinstance(sources, list)


class TestRAGSystemCourseAnalytics:
    """Tests for get_course_analytics() method"""

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_get_course_analytics(
        self, mock_session, mock_ai_gen, mock_vector_class, mock_doc, test_config
    ):
        """Test that course analytics are retrieved correctly"""
        mock_vector_instance = Mock()
        mock_vector_class.return_value = mock_vector_instance

        mock_vector_instance.get_course_count.return_value = 3
        mock_vector_instance.get_existing_course_titles.return_value = [
            "Course 1",
            "Course 2",
            "Course 3",
        ]

        rag = RAGSystem(test_config)

        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3
        assert "Course 1" in analytics["course_titles"]
