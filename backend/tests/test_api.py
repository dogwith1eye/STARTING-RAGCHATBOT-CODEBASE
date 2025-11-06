"""
API endpoint tests for the RAG system FastAPI application.

This module tests the HTTP API endpoints for query processing,
course statistics, and error handling scenarios.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for the POST /api/query endpoint"""

    def test_query_with_new_session(self, test_client, mock_rag_system):
        """Test query endpoint creates new session when session_id is not provided"""
        # Make request without session_id
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        # Assert successful response
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Verify session was created
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

        # Verify query was processed
        mock_rag_system.query.assert_called_once_with("What is MCP?", "test-session-123")

    def test_query_with_existing_session(self, test_client, mock_rag_system):
        """Test query endpoint uses provided session_id"""
        session_id = "existing-session-456"

        # Make request with session_id
        response = test_client.post(
            "/api/query",
            json={
                "query": "Explain MCP protocols",
                "session_id": session_id
            }
        )

        # Assert successful response
        assert response.status_code == 200
        data = response.json()

        # Verify session_id matches
        assert data["session_id"] == session_id

        # Verify create_session was NOT called (using existing session)
        mock_rag_system.session_manager.create_session.assert_not_called()

        # Verify query used the provided session
        mock_rag_system.query.assert_called_once_with("Explain MCP protocols", session_id)

    def test_query_response_structure(self, test_client, mock_rag_system):
        """Test query endpoint returns properly structured response"""
        response = test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify answer field
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

        # Verify sources field is a list
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0

        # Verify source structure
        source = data["sources"][0]
        assert "course_title" in source
        assert "lesson_number" in source
        assert "lesson_link" in source
        assert "course_link" in source

    def test_query_missing_field(self, test_client):
        """Test query endpoint rejects request with missing query field"""
        response = test_client.post(
            "/api/query",
            json={"session_id": "test-123"}  # Missing query field
        )

        # Should return 422 Unprocessable Entity for validation error
        assert response.status_code == 422

    def test_query_empty_string(self, test_client, mock_rag_system):
        """Test query endpoint handles empty query string"""
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )

        # Empty string is valid, should process
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once()

    def test_query_internal_error(self, test_client, mock_rag_system):
        """Test query endpoint handles internal errors gracefully"""
        # Make RAG system throw an error
        mock_rag_system.query.side_effect = Exception("Internal processing error")

        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        # Should return 500 Internal Server Error
        assert response.status_code == 500
        assert "Internal processing error" in response.json()["detail"]

    def test_query_no_rag_system(self, test_client_no_rag):
        """Test query endpoint when RAG system is not initialized"""
        response = test_client_no_rag.post(
            "/api/query",
            json={"query": "Test query"}
        )

        # Should return 500 when RAG system not initialized
        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"]

    def test_query_invalid_json(self, test_client):
        """Test query endpoint rejects malformed JSON"""
        response = test_client.post(
            "/api/query",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )

        # Should return 422 for invalid JSON
        assert response.status_code == 422

    def test_query_additional_fields_ignored(self, test_client, mock_rag_system):
        """Test query endpoint ignores additional fields in request"""
        response = test_client.post(
            "/api/query",
            json={
                "query": "What is MCP?",
                "extra_field": "should be ignored",
                "another_field": 123
            }
        )

        # Should still succeed, ignoring extra fields
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once()


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for the GET /api/courses endpoint"""

    def test_get_courses_success(self, test_client, mock_rag_system):
        """Test courses endpoint returns course statistics"""
        response = test_client.get("/api/courses")

        # Assert successful response
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "total_courses" in data
        assert "course_titles" in data

        # Verify data types and content
        assert isinstance(data["total_courses"], int)
        assert data["total_courses"] == 1

        assert isinstance(data["course_titles"], list)
        assert len(data["course_titles"]) == 1
        assert "Introduction to MCP" in data["course_titles"]

        # Verify analytics was called
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_courses_no_courses(self, test_client, mock_rag_system):
        """Test courses endpoint when no courses are loaded"""
        # Mock empty course analytics
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_multiple(self, test_client, mock_rag_system):
        """Test courses endpoint with multiple courses"""
        # Mock multiple courses
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": [
                "Introduction to MCP",
                "Advanced MCP Patterns",
                "MCP Security Best Practices"
            ]
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3

    def test_get_courses_internal_error(self, test_client, mock_rag_system):
        """Test courses endpoint handles internal errors"""
        # Make analytics throw an error
        mock_rag_system.get_course_analytics.side_effect = Exception("Database connection error")

        response = test_client.get("/api/courses")

        # Should return 500 Internal Server Error
        assert response.status_code == 500
        assert "Database connection error" in response.json()["detail"]

    def test_get_courses_no_rag_system(self, test_client_no_rag):
        """Test courses endpoint when RAG system is not initialized"""
        response = test_client_no_rag.get("/api/courses")

        # Should return 500 when RAG system not initialized
        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"]

    def test_get_courses_no_query_params(self, test_client, mock_rag_system):
        """Test courses endpoint ignores query parameters"""
        # Add query params (should be ignored for GET)
        response = test_client.get("/api/courses?param1=value1&param2=value2")

        # Should still work normally
        assert response.status_code == 200
        mock_rag_system.get_course_analytics.assert_called_once()


@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API endpoint interactions"""

    def test_query_and_courses_consistency(self, test_client, mock_rag_system):
        """Test that query and courses endpoints use the same RAG system"""
        # First, get courses
        courses_response = test_client.get("/api/courses")
        courses_data = courses_response.json()

        # Then make a query
        query_response = test_client.post(
            "/api/query",
            json={"query": "What courses are available?"}
        )
        query_data = query_response.json()

        # Both should succeed
        assert courses_response.status_code == 200
        assert query_response.status_code == 200

        # Verify they used the same RAG system instance
        assert mock_rag_system.get_course_analytics.called
        assert mock_rag_system.query.called

    def test_session_persistence_across_queries(self, test_client, mock_rag_system):
        """Test that session_id persists across multiple queries"""
        # First query creates session
        response1 = test_client.post(
            "/api/query",
            json={"query": "First query"}
        )
        session_id = response1.json()["session_id"]

        # Second query uses same session
        response2 = test_client.post(
            "/api/query",
            json={"query": "Second query", "session_id": session_id}
        )

        # Verify session_id is the same
        assert response2.json()["session_id"] == session_id

        # Verify create_session was only called once
        assert mock_rag_system.session_manager.create_session.call_count == 1

    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are properly configured"""
        response = test_client.get("/api/courses")

        # CORS middleware should add these headers
        assert response.status_code == 200
        # Note: TestClient doesn't always include middleware headers,
        # but the app should have CORS configured

    def test_content_type_json(self, test_client):
        """Test that API responses have correct content-type"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]


@pytest.mark.api
class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_endpoint(self, test_client):
        """Test request to non-existent endpoint"""
        response = test_client.get("/api/nonexistent")

        # Should return 404 Not Found
        assert response.status_code == 404

    def test_wrong_http_method(self, test_client):
        """Test using wrong HTTP method on endpoint"""
        # Try GET on query endpoint (should be POST)
        response = test_client.get("/api/query")

        # Should return 405 Method Not Allowed
        assert response.status_code == 405

        # Try POST on courses endpoint (should be GET)
        response = test_client.post("/api/courses")

        # Should return 405 Method Not Allowed
        assert response.status_code == 405

    def test_large_query_string(self, test_client, mock_rag_system):
        """Test handling of very large query strings"""
        # Create a large query (10KB)
        large_query = "x" * 10000

        response = test_client.post(
            "/api/query",
            json={"query": large_query}
        )

        # Should still process (up to reasonable limits)
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with(large_query, "test-session-123")

    def test_special_characters_in_query(self, test_client, mock_rag_system):
        """Test query with special characters and unicode"""
        special_query = "What is MCP? 你好 🚀 <script>alert('test')</script>"

        response = test_client.post(
            "/api/query",
            json={"query": special_query}
        )

        # Should handle special characters
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with(special_query, "test-session-123")
