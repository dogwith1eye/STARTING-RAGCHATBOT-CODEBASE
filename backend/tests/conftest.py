"""
Pytest configuration and shared fixtures for the RAG chatbot test suite.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend directory to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_course() -> Course:
    """Sample course data for testing"""
    return Course(
        title="Introduction to MCP",
        course_link="https://example.com/mcp",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=0, title="Getting Started", lesson_link="https://example.com/mcp/lesson0"),
            Lesson(lesson_number=1, title="Core Concepts", lesson_link="https://example.com/mcp/lesson1"),
            Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/mcp/lesson2"),
        ]
    )


@pytest.fixture
def sample_chunks() -> List[CourseChunk]:
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="MCP stands for Model Context Protocol. It enables communication between AI assistants and external data sources.",
            course_title="Introduction to MCP",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="The protocol uses a client-server architecture where the AI assistant acts as a client.",
            course_title="Introduction to MCP",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Advanced features include streaming responses and context management.",
            course_title="Introduction to MCP",
            lesson_number=2,
            chunk_index=2
        ),
    ]


@pytest.fixture
def sample_search_results() -> SearchResults:
    """Sample search results with content"""
    return SearchResults(
        documents=[
            "MCP stands for Model Context Protocol. It enables communication between AI assistants and external data sources.",
            "The protocol uses a client-server architecture where the AI assistant acts as a client."
        ],
        metadata=[
            {"course_title": "Introduction to MCP", "lesson_number": 0, "chunk_index": 0},
            {"course_title": "Introduction to MCP", "lesson_number": 1, "chunk_index": 1}
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results() -> SearchResults:
    """Empty search results"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results() -> SearchResults:
    """Search results with error"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="No course found matching 'NonExistent Course'"
    )


# ============================================================================
# Mock VectorStore Fixtures
# ============================================================================

@pytest.fixture
def mock_vector_store(sample_search_results):
    """Mock VectorStore that returns sample results by default"""
    mock = Mock()
    mock.max_results = 5
    mock.search.return_value = sample_search_results
    mock.get_course_link.return_value = "https://example.com/mcp"
    mock.get_lesson_link.return_value = "https://example.com/mcp/lesson0"
    mock.get_course_count.return_value = 1
    mock._resolve_course_name.return_value = "Introduction to MCP"
    return mock


@pytest.fixture
def mock_vector_store_with_zero_max_results(empty_search_results):
    """Mock VectorStore with MAX_RESULTS=0 (reproduces the bug)"""
    mock = Mock()
    mock.max_results = 0
    mock.search.return_value = empty_search_results  # Returns empty when max_results=0
    mock.get_course_link.return_value = "https://example.com/mcp"
    mock.get_lesson_link.return_value = None
    mock.get_course_count.return_value = 1
    mock._resolve_course_name.return_value = "Introduction to MCP"
    return mock


@pytest.fixture
def mock_vector_store_empty(empty_search_results):
    """Mock VectorStore with no courses loaded"""
    mock = Mock()
    mock.max_results = 5
    mock.search.return_value = empty_search_results
    mock.get_course_link.return_value = None
    mock.get_lesson_link.return_value = None
    mock.get_course_count.return_value = 0
    mock._resolve_course_name.return_value = None
    return mock


# ============================================================================
# Mock Anthropic API Fixtures
# ============================================================================

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing AI generation"""
    mock_client = Mock()

    # Create mock response object
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"

    # Create mock content
    mock_content = Mock()
    mock_content.text = "MCP is a protocol for AI communication with external data sources."
    mock_response.content = [mock_content]

    # Set up the messages.create method
    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_anthropic_client_with_tool_use(sample_search_results):
    """Mock Anthropic client that requests tool use"""
    mock_client = Mock()

    # First response: Claude requests to use search tool
    mock_tool_use_response = Mock()
    mock_tool_use_response.stop_reason = "tool_use"

    # Create mock tool use content block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {"query": "what is MCP"}

    mock_tool_use_response.content = [mock_tool_block]

    # Second response: Claude's final answer after tool execution
    mock_final_response = Mock()
    mock_final_response.stop_reason = "end_turn"

    mock_final_content = Mock()
    mock_final_content.text = "MCP is a protocol that enables AI assistants to communicate with external data sources."
    mock_final_response.content = [mock_final_content]

    # Set up messages.create to return first tool_use, then final response
    mock_client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]

    return mock_client


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Test configuration with proper settings"""
    from config import Config
    return Config(
        ANTHROPIC_API_KEY="test_api_key",
        ANTHROPIC_MODEL="claude-sonnet-4-20250514",
        CHROMA_PATH="./test_chroma_db",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=5,  # Proper value
        MAX_HISTORY=2
    )


@pytest.fixture
def test_config_zero_max_results():
    """Test configuration with MAX_RESULTS=0 (reproduces bug)"""
    from config import Config
    return Config(
        ANTHROPIC_API_KEY="test_api_key",
        ANTHROPIC_MODEL="claude-sonnet-4-20250514",
        CHROMA_PATH="./test_chroma_db",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=0,  # BUG: This causes searches to return 0 results
        MAX_HISTORY=2
    )


# ============================================================================
# Tool-related Fixtures
# ============================================================================

@pytest.fixture
def mock_tool_manager(sample_search_results):
    """Mock ToolManager for testing"""
    mock = Mock()
    mock.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "course_name": {"type": "string", "description": "Course title"},
                    "lesson_number": {"type": "integer", "description": "Lesson number"}
                },
                "required": ["query"]
            }
        }
    ]

    # Format results like CourseSearchTool would
    formatted_result = "[Introduction to MCP - Lesson 0]\n" + sample_search_results.documents[0]
    mock.execute_tool.return_value = formatted_result

    mock.get_last_sources.return_value = [
        {
            "course_title": "Introduction to MCP",
            "lesson_number": 0,
            "lesson_link": "https://example.com/mcp/lesson0",
            "course_link": "https://example.com/mcp"
        }
    ]

    mock.reset_sources.return_value = None

    return mock


# ============================================================================
# FastAPI Test App and Client Fixtures
# ============================================================================

@pytest.fixture
def test_app():
    """
    Create a test FastAPI app without static file mounting to avoid import issues.
    This app only has the API endpoints for testing.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any, Union

    # Create a minimal test app
    app = FastAPI(title="Test RAG System")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Dict[str, Any]]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Store a mock RAG system on the app for tests to access
    app.state.rag_system = None

    # Define API endpoints inline
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            if not app.state.rag_system:
                raise HTTPException(status_code=500, detail="RAG system not initialized")

            session_id = request.session_id
            if not session_id:
                session_id = app.state.rag_system.session_manager.create_session()

            answer, sources = app.state.rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            if not app.state.rag_system:
                raise HTTPException(status_code=500, detail="RAG system not initialized")

            analytics = app.state.rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def mock_rag_system(mock_vector_store, mock_anthropic_client, mock_tool_manager):
    """Mock RAGSystem for API testing"""
    mock = Mock()
    mock.session_manager.create_session.return_value = "test-session-123"
    mock.query.return_value = (
        "MCP is a protocol for AI communication with external data sources.",
        [
            {
                "course_title": "Introduction to MCP",
                "lesson_number": 0,
                "lesson_link": "https://example.com/mcp/lesson0",
                "course_link": "https://example.com/mcp"
            }
        ]
    )
    mock.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Introduction to MCP"]
    }
    return mock


@pytest.fixture
def test_client(test_app, mock_rag_system):
    """
    Create a test client with a mocked RAG system.
    This avoids issues with static file mounting and provides controlled test behavior.
    """
    test_app.state.rag_system = mock_rag_system
    return TestClient(test_app)


@pytest.fixture
def test_client_no_rag(test_app):
    """Test client without RAG system initialized (for error testing)"""
    return TestClient(test_app)
