# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) Chatbot** for querying educational course materials. The system uses a **tool-based architecture** where Claude decides when to invoke search tools, rather than always injecting context into prompts.

## Running the Application

```bash
# Start the server (creates necessary directories and launches FastAPI)
./run.sh

# Alternative: Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

Server runs at `http://localhost:8000` (frontend + API)

**Prerequisites:**
- Python 3.13+
- `uv` package manager
- `.env` file with `ANTHROPIC_API_KEY=your_key_here`

## Dependencies

```bash
# Install/sync dependencies
uv sync
```

Key dependencies: FastAPI, ChromaDB, Anthropic SDK, Sentence-Transformers

## Architecture

### High-Level Flow

```
User Query → FastAPI (/api/query) → RAGSystem → AIGenerator
                                         ↓
                                    ToolManager
                                         ↓
                                  CourseSearchTool
                                         ↓
                                    VectorStore (ChromaDB)
```

### Tool-Based RAG Pattern

**Critical architectural decision:** This system uses **tool-calling** instead of direct context injection:

1. **Claude receives tools** with their definitions (not the search results)
2. **Claude decides** whether to invoke `search_course_content` tool
3. **Tool executes** → Vector search in ChromaDB → Returns formatted results
4. **Claude synthesizes** final answer from tool results

This differs from traditional RAG where you retrieve context first and inject it into the prompt.

### Core Components

**Backend Structure:**
- `app.py` - FastAPI application with two endpoints:
  - `POST /api/query` - Process user queries
  - `GET /api/courses` - Get course statistics
- `rag_system.py` - **Main orchestrator** that coordinates all components
- `ai_generator.py` - Claude API integration with tool execution loop
- `search_tools.py` - Tool framework (`Tool` base class, `CourseSearchTool`, `ToolManager`)
- `vector_store.py` - ChromaDB interface with two collections:
  - `course_catalog` - Course metadata (for semantic course name resolution)
  - `course_content` - Text chunks with embeddings
- `document_processor.py` - Parses structured course documents and creates sentence-based chunks
- `session_manager.py` - Conversation history (limited to `MAX_HISTORY=2` exchanges)
- `config.py` - Centralized configuration
- `models.py` - Pydantic models (`Course`, `Lesson`, `CourseChunk`)

**Frontend:** Vanilla HTML/CSS/JS in `frontend/` directory (served as static files by FastAPI)

### Document Processing Pipeline

1. **Input Format:** Documents in `/docs` folder with structure:
   ```
   Course Title: [title]
   Course Link: [url]
   Course Instructor: [name]

   Lesson 0: [title]
   Lesson Link: [url]
   [lesson content]
   ```

2. **Processing Steps:**
   - Parse course metadata and lesson boundaries
   - Split lesson content into sentence-based chunks (800 chars, 100 overlap)
   - Add contextual prefixes: `"Course {title} Lesson {num} content: {chunk}"`
   - Generate embeddings using `all-MiniLM-L6-v2`
   - Store in ChromaDB with metadata (course_title, lesson_number, chunk_index)

3. **Startup:** Documents auto-load from `../docs` on server startup (see `app.py:startup_event`)

### Configuration Values

Key settings in `backend/config.py`:
- `CHUNK_SIZE = 800` - Characters per chunk
- `CHUNK_OVERLAP = 100` - Overlap between chunks
- `MAX_RESULTS = 5` - Search results limit
- `MAX_HISTORY = 2` - Conversation exchanges to retain
- `ANTHROPIC_MODEL = "claude-sonnet-4-20250514"`
- `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` (384-dimensional embeddings)

### Vector Search Behavior

The `VectorStore.search()` method:
1. **Course name resolution:** If `course_name` provided, performs semantic search in `course_catalog` to find exact title
2. **Filter construction:** Builds ChromaDB `where` filters for course and/or lesson
3. **Content search:** Queries `course_content` collection with embeddings
4. **Returns:** `SearchResults` dataclass with documents, metadata, distances

**Important:** The tool tracks sources via `last_sources` attribute for UI display.

### Session Management

Sessions are stored in-memory (lost on restart). Each session:
- Has unique UUID
- Stores conversation history as formatted string
- Limited to `MAX_HISTORY` exchanges (configurable)
- Passed to Claude as part of system prompt context

### Tool Execution Flow

When Claude invokes a tool:
1. `AIGenerator` detects `stop_reason="tool_use"`
2. Calls `_handle_tool_execution()` which:
   - Extracts tool name and parameters from response
   - Executes via `ToolManager.execute_tool()`
   - Adds tool results as user message
   - Makes second API call to Claude with results
   - Returns final synthesized answer

## Common Issues

**ChromaDB persistence:** Data stored in `./chroma_db` (created by `run.sh`). To reset, delete this directory.

**Course name matching:** Uses semantic search, so partial names work (e.g., "MCP" matches "Introduction to MCP"). Exact matching not required.

**API key:** Must be in `.env` file at repository root. Use `.env.example` as template.

**Windows users:** Use Git Bash to run shell scripts.

## AI System Prompt

Located in `ai_generator.py:SYSTEM_PROMPT` (lines 8-30). Key instructions:
- Use search tool **only for course-specific questions**
- **One search per query maximum**
- Answer general questions without searching
- No meta-commentary about search process
- Responses must be brief, educational, clear, example-supported
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- don't run the server using ./run.sh I will start it myself