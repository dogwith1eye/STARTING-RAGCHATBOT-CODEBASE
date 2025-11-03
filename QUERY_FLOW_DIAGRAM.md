# Query Flow Diagram: User Query Processing

## Complete System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 FRONTEND                                     │
│                            (frontend/script.js)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    User types: "What is prompt caching?"
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │  sendMessage() - Lines 45-96                        │
        │  • Disable input                                    │
        │  • Add user message to UI                           │
        │  • Show loading spinner                             │
        │  • Prepare request payload                          │
        └─────────────────────────────────────────────────────┘
                                      │
                                      │ POST /api/query
                                      │ {
                                      │   query: "What is prompt caching?",
                                      │   session_id: "abc123"
                                      │ }
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND - API LAYER                             │
│                              (backend/app.py)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────────────────────────────┐
        │  @app.post("/api/query") - Lines 56-74             │
        │  • Parse QueryRequest                               │
        │  • Get/create session_id                            │
        │  • Call rag_system.query()                          │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RAG SYSTEM ORCHESTRATOR                             │
│                         (backend/rag_system.py)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────────────────────────────┐
        │  query() - Lines 102-140                            │
        │  1. Build prompt with instructions                  │
        │  2. Get conversation history from session           │
        │  3. Get tool definitions from ToolManager           │
        │  4. Call AIGenerator with tools                     │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │         SessionManager                              │
        │  • Retrieve last 2 exchanges                        │
        │  • Format as: "User: ...\nAssistant: ..."         │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AI GENERATION LAYER                                 │
│                        (backend/ai_generator.py)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────────────────────────────┐
        │  generate_response() - Lines 43-87                  │
        │  • Build system prompt + history                    │
        │  • Prepare tool definitions                         │
        │  • Call Anthropic API                               │
        └─────────────────────────────────────────────────────┘
                                      │
                     ╔════════════════════════════════╗
                     ║   ANTHROPIC CLAUDE API CALL    ║
                     ║                                ║
                     ║  Model: claude-sonnet-4        ║
                     ║  Temperature: 0                ║
                     ║  Max Tokens: 800               ║
                     ║  Tools: [search_course_content]║
                     ║  Tool Choice: auto             ║
                     ╚════════════════════════════════╝
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  Claude analyzes query       │
                    │  Decides: "This is a         │
                    │  course-specific question    │
                    │  - I need to search!"        │
                    └──────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  Response:                   │
                    │  stop_reason: "tool_use"     │
                    │  tool: "search_course_content"│
                    │  input: {                    │
                    │    query: "prompt caching"   │
                    │  }                           │
                    └──────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │  _handle_tool_execution() - Lines 89-135           │
        │  • Extract tool name & parameters                   │
        │  • Call tool_manager.execute_tool()                 │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             TOOL EXECUTION LAYER                             │
│                         (backend/search_tools.py)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────────────────────────────┐
        │  ToolManager.execute_tool()                         │
        │  • Find registered tool by name                     │
        │  • Call CourseSearchTool.execute()                  │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │  CourseSearchTool.execute() - Lines 52-86          │
        │  • Call vector_store.search()                       │
        │  • Format results with headers                      │
        │  • Track sources in last_sources                    │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VECTOR SEARCH LAYER                                 │
│                        (backend/vector_store.py)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────────────────────────────┐
        │  search() - Lines 61-100                            │
        │  1. Resolve course name (if provided)               │
        │  2. Build filter dictionary                         │
        │  3. Query ChromaDB                                  │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
                     ╔════════════════════════════════╗
                     ║         ChromaDB Query         ║
                     ║                                ║
                     ║  Collection: course_content    ║
                     ║  Query: "prompt caching"       ║
                     ║  Embedding: all-MiniLM-L6-v2   ║
                     ║  Vector: 384 dimensions        ║
                     ║  n_results: 5                  ║
                     ║  Method: Cosine similarity     ║
                     ╚════════════════════════════════╝
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  Returns Top 5 Results:      │
                    │                              │
                    │  1. [Course A - Lesson 3]    │
                    │     "Prompt caching          │
                    │      retains results..."     │
                    │     distance: 0.15           │
                    │                              │
                    │  2. [Course A - Lesson 3]    │
                    │     "You can use prompt      │
                    │      caching to..."          │
                    │     distance: 0.23           │
                    │                              │
                    │  3-5. [More results...]      │
                    └──────────────────────────────┘
                                      │
                                      │ SearchResults(documents, metadata, distances)
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │  _format_results() - Lines 88-114                   │
        │  • Add [Course - Lesson] headers                    │
        │  • Store sources: ["Course A - Lesson 3", ...]     │
        │  • Return formatted string                          │
        └─────────────────────────────────────────────────────┘
                                      │
                    Tool Result:      │
                    "[Course A - Lesson 3]
                    Prompt caching retains...

                    [Course A - Lesson 3]
                    You can use prompt caching..."
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BACK TO AI GENERATION LAYER                              │
│                        (backend/ai_generator.py)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────────────────────────────┐
        │  _handle_tool_execution() continued                 │
        │  • Build messages with tool results                 │
        │  • Make second Claude API call                      │
        └─────────────────────────────────────────────────────┘
                                      │
                     ╔════════════════════════════════╗
                     ║  SECOND CLAUDE API CALL        ║
                     ║                                ║
                     ║  Messages:                     ║
                     ║  1. User: "What is..."         ║
                     ║  2. Assistant: [tool_use]      ║
                     ║  3. User: [tool_result]        ║
                     ║                                ║
                     ║  NO TOOLS (synthesis only)     ║
                     ╚════════════════════════════════╝
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  Claude synthesizes answer   │
                    │  from tool results:          │
                    │                              │
                    │  "Prompt caching is a        │
                    │   feature that retains       │
                    │   processing results         │
                    │   between model              │
                    │   invocations to reduce      │
                    │   cost and latency..."       │
                    └──────────────────────────────┘
                                      │
                                      │ Returns final text
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BACK TO RAG SYSTEM ORCHESTRATOR                         │
│                         (backend/rag_system.py)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────────────────────────────┐
        │  query() continued - Lines 129-140                  │
        │  • Get sources from tool_manager                    │
        │  • Update conversation history                      │
        │  • Reset sources                                    │
        │  • Return (response, sources)                       │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │         SessionManager                              │
        │  • Add exchange to history                          │
        │  • Trim to MAX_HISTORY (2 exchanges)               │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BACK TO API LAYER                                   │
│                              (backend/app.py)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────────────────────────────┐
        │  query_documents() - Lines 68-72                    │
        │  • Build QueryResponse                              │
        │  • Return JSON                                      │
        └─────────────────────────────────────────────────────┘
                                      │
                    HTTP 200 OK       │
                    {                 │
                      "answer": "Prompt caching is...",
                      "sources": [
                        "Building Towards... - Lesson 3",
                        "Building Towards... - Lesson 3"
                      ],
                      "session_id": "abc123"
                    }                 │
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BACK TO FRONTEND                                │
│                            (frontend/script.js)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────────────────────────────┐
        │  sendMessage() continued - Lines 76-95              │
        │  • Parse JSON response                              │
        │  • Remove loading spinner                           │
        │  • Call addMessage(data.answer, 'assistant',        │
        │    data.sources)                                    │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │  addMessage() - Lines 113-138                       │
        │  • Convert markdown to HTML                         │
        │  • Create message div                               │
        │  • Add collapsible sources section                  │
        │  • Append to chat UI                                │
        │  • Scroll to bottom                                 │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │   USER SEES RESPONSE:        │
                    │                              │
                    │   ┌────────────────────────┐ │
                    │   │ Assistant              │ │
                    │   │                        │ │
                    │   │ Prompt caching is a    │ │
                    │   │ feature that retains   │ │
                    │   │ processing results...  │ │
                    │   │                        │ │
                    │   │ ▼ Sources              │ │
                    │   │   Building Towards...  │ │
                    │   │   - Lesson 3           │ │
                    │   └────────────────────────┘ │
                    └──────────────────────────────┘
```

## Key Flow Characteristics

### 🔄 Two Claude API Calls
1. **First call:** Claude receives query + tools → Decides to use tool
2. **Second call:** Claude receives tool results → Synthesizes answer

### 📊 Data Transformations
```
User Query String
  → API Request JSON
    → RAG Prompt
      → Tool Call Parameters
        → Vector Embedding (384D)
          → Search Results
            → Formatted Tool Response
              → Final Answer
                → JSON Response
                  → HTML Display
```

### ⏱️ Typical Timing Breakdown
- Frontend → API: ~10ms
- API → RAG System: <1ms
- Session Retrieval: <1ms
- First Claude Call: ~500-1000ms
- Tool Execution: ~50-100ms
  - Vector Search: ~30-50ms
  - Formatting: ~10ms
- Second Claude Call: ~1000-2000ms
- Response → Frontend: ~10ms
- **Total: ~1.5-3 seconds**

### 💾 State Persistence
- **Session history:** In-memory (lost on restart)
- **Vector database:** Persistent in `./chroma_db/`
- **Tool sources:** Temporary (cleared after each query)

### 🔍 Search Strategy
1. Query embedded using SentenceTransformer
2. Cosine similarity search in ChromaDB
3. Returns top 5 chunks (configurable via `MAX_RESULTS`)
4. Metadata filtering by course/lesson if specified
5. Results include contextual prefixes from chunking
