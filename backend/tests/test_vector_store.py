"""
Unit tests for vector_store.py

Tests the VectorStore to verify:
1. CRITICAL: Search behavior with MAX_RESULTS=0 vs correct values
2. Course name resolution with semantic search
3. Filter construction for courses and lessons
4. SearchResults dataclass functionality
5. Error handling
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import SearchResults, VectorStore


class TestSearchResults:
    """Tests for SearchResults dataclass"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB response"""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"course": "MCP"}, {"course": "Python"}]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert len(results.documents) == 2
        assert results.documents[0] == "doc1"
        assert len(results.metadata) == 2
        assert results.metadata[0]["course"] == "MCP"
        assert len(results.distances) == 2
        assert results.error is None

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB response"""
        chroma_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = SearchResults.from_chroma(chroma_results)

        assert results.is_empty()
        assert len(results.documents) == 0

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("Course not found")

        assert results.is_empty()
        assert results.error == "Course not found"
        assert len(results.documents) == 0

    def test_is_empty(self):
        """Test is_empty() method"""
        empty = SearchResults(documents=[], metadata=[], distances=[])
        not_empty = SearchResults(documents=["doc"], metadata=[{}], distances=[0.1])

        assert empty.is_empty()
        assert not not_empty.is_empty()


class TestVectorStoreInit:
    """Tests for VectorStore initialization"""

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_init_with_correct_max_results(self, mock_embedding, mock_client):
        """Test that VectorStore stores max_results correctly"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_or_create_collection = Mock(return_value=Mock())

        store = VectorStore(
            chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5
        )

        assert store.max_results == 5

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_init_with_zero_max_results_bug(self, mock_embedding, mock_client):
        """
        CRITICAL TEST: VectorStore initialized with MAX_RESULTS=0

        This reproduces the configuration bug where MAX_RESULTS=0
        causes all searches to return 0 results.
        """
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_or_create_collection = Mock(return_value=Mock())

        store = VectorStore(
            chroma_path="./test_db",
            embedding_model="all-MiniLM-L6-v2",
            max_results=0,  # BUG: This should never be 0
        )

        # Verify the bug exists
        assert store.max_results == 0

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_init_creates_collections(self, mock_embedding, mock_client):
        """Test that VectorStore creates both required collections"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        collection_mock = Mock()
        mock_client_instance.get_or_create_collection = Mock(
            return_value=collection_mock
        )

        store = VectorStore(
            chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5
        )

        # Verify both collections were created
        assert mock_client_instance.get_or_create_collection.call_count == 2

        # Verify collection names
        calls = mock_client_instance.get_or_create_collection.call_args_list
        collection_names = [call.kwargs["name"] for call in calls]
        assert "course_catalog" in collection_names
        assert "course_content" in collection_names


class TestVectorStoreSearch:
    """Tests for VectorStore.search() method"""

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_with_correct_max_results(self, mock_embedding, mock_client):
        """Test that search uses max_results correctly when set to 5"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["result1", "result2"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.2]],
        }
        mock_client_instance.get_or_create_collection = Mock(
            return_value=mock_collection
        )

        store = VectorStore(
            chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5
        )

        results = store.search(query="test query")

        # Verify search was called with correct n_results
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args
        assert call_args.kwargs["n_results"] == 5

        # Verify results are returned
        assert len(results.documents) == 2

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_with_zero_max_results_returns_empty(
        self, mock_embedding, mock_client
    ):
        """
        CRITICAL TEST: Search with MAX_RESULTS=0 returns empty results

        This demonstrates the bug: when max_results=0, ChromaDB is asked
        for 0 results, so it returns nothing.
        """
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_collection = Mock()
        # ChromaDB returns empty arrays when n_results=0
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_client_instance.get_or_create_collection = Mock(
            return_value=mock_collection
        )

        store = VectorStore(
            chroma_path="./test_db",
            embedding_model="all-MiniLM-L6-v2",
            max_results=0,  # BUG REPRODUCTION
        )

        results = store.search(query="test query")

        # Verify ChromaDB was asked for 0 results
        call_args = mock_collection.query.call_args
        assert call_args.kwargs["n_results"] == 0

        # Verify we get empty results (THE BUG)
        assert results.is_empty()
        assert len(results.documents) == 0

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_with_custom_limit_overrides_max_results(
        self, mock_embedding, mock_client
    ):
        """Test that explicit limit parameter overrides max_results"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["r1", "r2", "r3"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.1, 0.2, 0.3]],
        }
        mock_client_instance.get_or_create_collection = Mock(
            return_value=mock_collection
        )

        store = VectorStore(
            chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5
        )

        # Call with explicit limit
        results = store.search(query="test", limit=3)

        # Verify limit was used instead of max_results
        call_args = mock_collection.query.call_args
        assert call_args.kwargs["n_results"] == 3

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_with_course_filter(self, mock_embedding, mock_client):
        """Test search with course_name filter"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_catalog = Mock()
        mock_catalog.query.return_value = {
            "documents": [["course text"]],
            "metadatas": [[{"title": "Introduction to MCP"}]],
        }

        mock_content = Mock()
        mock_content.query.return_value = {
            "documents": [["content"]],
            "metadatas": [[{"course_title": "Introduction to MCP"}]],
            "distances": [[0.1]],
        }

        call_count = [0]

        def get_collection(name, embedding_function):
            call_count[0] += 1
            if "catalog" in name:
                return mock_catalog
            return mock_content

        mock_client_instance.get_or_create_collection = get_collection

        store = VectorStore(
            chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5
        )

        results = store.search(query="test", course_name="MCP")

        # Verify catalog was queried to resolve course name
        assert mock_catalog.query.called

        # Verify content search included course filter
        call_args = mock_content.query.call_args
        assert "where" in call_args.kwargs
        # Filter should contain the resolved course title
        assert call_args.kwargs["where"] == {"course_title": "Introduction to MCP"}

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_with_lesson_filter(self, mock_embedding, mock_client):
        """Test search with lesson_number filter"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["content"]],
            "metadatas": [[{"lesson_number": 1}]],
            "distances": [[0.1]],
        }
        mock_client_instance.get_or_create_collection = Mock(
            return_value=mock_collection
        )

        store = VectorStore(
            chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5
        )

        results = store.search(query="test", lesson_number=1)

        # Verify filter was applied
        call_args = mock_collection.query.call_args
        assert call_args.kwargs["where"] == {"lesson_number": 1}

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_handles_exceptions(self, mock_embedding, mock_client):
        """Test that search handles exceptions gracefully"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Database error")
        mock_client_instance.get_or_create_collection = Mock(
            return_value=mock_collection
        )

        store = VectorStore(
            chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5
        )

        results = store.search(query="test")

        # Should return error results instead of crashing
        assert results.error is not None
        assert "Search error" in results.error
        assert results.is_empty()


class TestVectorStoreBuildFilter:
    """Tests for _build_filter() method"""

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_build_filter_no_params(self, mock_embedding, mock_client):
        """Test filter building with no parameters"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_or_create_collection = Mock(return_value=Mock())

        store = VectorStore("./test", "model", 5)

        filter_dict = store._build_filter(course_title=None, lesson_number=None)

        assert filter_dict is None

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_build_filter_course_only(self, mock_embedding, mock_client):
        """Test filter building with only course"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_or_create_collection = Mock(return_value=Mock())

        store = VectorStore("./test", "model", 5)

        filter_dict = store._build_filter(
            course_title="Test Course", lesson_number=None
        )

        assert filter_dict == {"course_title": "Test Course"}

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_build_filter_lesson_only(self, mock_embedding, mock_client):
        """Test filter building with only lesson"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_or_create_collection = Mock(return_value=Mock())

        store = VectorStore("./test", "model", 5)

        filter_dict = store._build_filter(course_title=None, lesson_number=2)

        assert filter_dict == {"lesson_number": 2}

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_build_filter_both_params(self, mock_embedding, mock_client):
        """Test filter building with both course and lesson"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_or_create_collection = Mock(return_value=Mock())

        store = VectorStore("./test", "model", 5)

        filter_dict = store._build_filter(course_title="Test Course", lesson_number=2)

        assert "$and" in filter_dict
        assert {"course_title": "Test Course"} in filter_dict["$and"]
        assert {"lesson_number": 2} in filter_dict["$and"]


class TestVectorStoreResolveCourseName:
    """Tests for _resolve_course_name() method"""

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_resolve_course_name_found(self, mock_embedding, mock_client):
        """Test course name resolution when course exists"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_catalog = Mock()
        mock_catalog.query.return_value = {
            "documents": [["Introduction to MCP"]],
            "metadatas": [[{"title": "Introduction to MCP"}]],
        }

        def get_collection(name, embedding_function):
            if "catalog" in name:
                return mock_catalog
            return Mock()

        mock_client_instance.get_or_create_collection = get_collection

        store = VectorStore("./test", "model", 5)

        resolved = store._resolve_course_name("MCP")

        assert resolved == "Introduction to MCP"

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_resolve_course_name_not_found(self, mock_embedding, mock_client):
        """Test course name resolution when course doesn't exist"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_catalog = Mock()
        mock_catalog.query.return_value = {"documents": [[]], "metadatas": [[]]}

        def get_collection(name, embedding_function):
            if "catalog" in name:
                return mock_catalog
            return Mock()

        mock_client_instance.get_or_create_collection = get_collection

        store = VectorStore("./test", "model", 5)

        resolved = store._resolve_course_name("Nonexistent Course")

        assert resolved is None

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_resolve_course_name_handles_exceptions(self, mock_embedding, mock_client):
        """Test that course name resolution handles exceptions"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_catalog = Mock()
        mock_catalog.query.side_effect = Exception("Database error")

        def get_collection(name, embedding_function):
            if "catalog" in name:
                return mock_catalog
            return Mock()

        mock_client_instance.get_or_create_collection = get_collection

        store = VectorStore("./test", "model", 5)

        resolved = store._resolve_course_name("MCP")

        # Should return None instead of crashing
        assert resolved is None
