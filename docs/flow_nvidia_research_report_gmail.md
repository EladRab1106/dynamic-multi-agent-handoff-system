# Runtime Flow: NVIDIA Research Report Gmail Request

## 1. High-Level Overview

This system is a **distributed multi-agent LangGraph architecture** with strict service boundaries. The orchestrator process (`main.py` + `graph/build_graph.py`) runs in one Python process and coordinates remote agent services via HTTP. The Supervisor planning agent (`agents/supervisor/`) is itself a remote service that receives only a snapshot of available capability strings (e.g., `["research", "create_document", "gmail"]`) injected into its state context. It never sees agent names, URLs, or ports. The orchestrator maintains a `capability_index` mapping (capability → agent_name) that exists **only in the orchestrator's memory**—there is no shared memory, database, or registry between services. When the Supervisor returns a capability string like `"research"` in `state["next"]`, the orchestrator's `route_from_supervisor()` function maps it to `"Researcher"` and routes execution to the corresponding `RemoteGraph` node. Each agent (Researcher, DocumentCreator, Gmail) runs as an independent LangGraph service on its own port, fully self-contained with local config, state, and graph definitions.

---

## 2. Exact "Happy Path" Timeline

### Initialization Phase

1. **`python main.py` execution starts**
   - Location: `main.py:14` (`if __name__ == "__main__":`)
   - `load_dotenv()` is called (line 7) to load environment variables

2. **Capability Discovery**
   - Location: `main.py:22` → `graph/capability_discovery.py:33` (`discover_capabilities()`)
   - Reads `AGENT_SERVICES` environment variable (comma-separated URLs)
   - For each URL (e.g., `http://localhost:8001`), calls `_fetch_agent_metadata()`
   - Queries `GET {service_url}/graphs` endpoint
   - Extracts `metadata.agent_name` and `metadata.capabilities` from response
   - If `/graphs` fails, falls back to port-based mapping (see `graph/capability_discovery.py:177-198`)
   - Returns `capability_index: Dict[str, str]`:
     ```python
     {
       "research": "Researcher",
       "create_document": "DocumentCreator",
       "gmail": "Gmail"
     }
     ```

3. **Capability List Extraction**
   - Location: `main.py:23`
   - `capabilities = sorted(capability_index.keys())` → `["create_document", "gmail", "research"]`
   - This list contains **only capability strings**, no agent names

4. **Graph Building**
   - Location: `main.py:49` → `graph/build_graph.py:16` (`build_graph(capability_index=...)`)
   - Creates `RemoteGraph` instances for each agent:
     - `Researcher`: `http://localhost:8001` (from `RESEARCHER_SERVICE_URL` env var, default: 8001)
     - `DocumentCreator`: `http://localhost:8002` (from `DOCUMENT_CREATOR_SERVICE_URL`, default: 8002)
     - `Gmail`: `http://localhost:8000` (from `GMAIL_SERVICE_URL`, default: 8000)
     - `Supervisor`: `http://localhost:8004` (from `SUPERVISOR_SERVICE_URL`, default: 8004)
   - Builds `capability_to_agent` mapping from `capability_index` (line 54-56)
   - Defines `route_from_supervisor()` function (line 70-95) that maps capability → agent_name
   - Returns compiled graph with entry point `"Supervisor"`

5. **Initial State Creation**
   - Location: `main.py:54-60`
   - Creates `init_state: AgentState`:
     ```python
     {
       "messages": [HumanMessage(content="research the company nvidia, make report file on it and send the file via gmail to eladrabinovitch1106@gmail.com")],
       "next": "Supervisor",
       "context": {
         "capabilities": ["create_document", "gmail", "research"]  # Only strings, no agent names
       }
     }
     ```

6. **Workflow Invocation**
   - Location: `main.py:67` → `workflow.invoke(init_state)`
   - Graph execution begins at `"Supervisor"` node

### Execution Phase

7. **First Supervisor Call**
   - Location: `graph/build_graph.py:60` → `RemoteGraph("supervisor", url="http://localhost:8004")`
   - HTTP POST to `http://localhost:8004/runs/stream` (LangServe endpoint)
   - Payload: Full `AgentState` with `context["capabilities"]` snapshot
   - Supervisor service receives request at `agents/supervisor/graph.py:34` → `supervisor_node()`

8. **Supervisor Planning**
   - Location: `agents/supervisor/supervisor.py:28` (`supervisor_node()`)
   - Reads `ctx["capabilities"]` = `["create_document", "gmail", "research"]` (line 41)
   - Builds system prompt with dynamic capability list (line 72, 128-164)
   - Extracts user request: `"research the company nvidia, make report file on it and send the file via gmail to eladrabinovitch1106@gmail.com"` (line 74-92)
   - Invokes LLM with structured output: `llm.with_structured_output(Plan)` (line 175)
   - LLM returns `Plan(steps=["research", "create_document", "gmail"])` (line 179)
   - Stores plan in context: `ctx["plan"] = ["research", "create_document", "gmail"]` (line 203)
   - Sets `ctx["current_step_index"] = 0` (line 204)
   - Returns: `{"next": "research", "context": ctx}` (line 402)
   - **Note**: Supervisor returns capability string `"research"`, NOT agent name `"Researcher"`

9. **Orchestrator Routing (Research)**
   - Location: `graph/build_graph.py:70` (`route_from_supervisor()`)
   - Receives `state["next"] = "research"`
   - Maps `"research"` → `"Researcher"` via `capability_to_agent["research"]` (line 84-86)
   - Routes to `"Researcher"` node

10. **Researcher Agent Execution**
    - Location: `graph/build_graph.py:61` → `RemoteGraph("researcher", url="http://localhost:8001")`
    - HTTP POST to `http://localhost:8001/runs/stream`
    - Researcher agent processes request, performs web search/research on NVIDIA
    - Returns completion contract in AIMessage:
      ```json
      {
        "completed_capability": "research",
        "data": {
          "research_summary": "...",
          "tool_used": true
        }
      }
      ```
    - Graph edge returns to `"Supervisor"` (line 66)

11. **Second Supervisor Call (After Research)**
    - Location: `agents/supervisor/supervisor.py:210` (STEP 2: Progress tracking)
    - Reads `ctx["plan"] = ["research", "create_document", "gmail"]` (line 211)
    - Reads `ctx["current_step_index"] = 0` (line 212)
    - Extracts completion contract from last message via `extract_completed_capability()` (line 294)
    - Detects `completed_capability = "research"` (line 301)
    - Advances: `ctx["current_step_index"] = 1` (line 305)
    - Reads `plan[1] = "create_document"` (line 349)
    - Returns: `{"next": "create_document", "context": ctx}` (line 402)

12. **Orchestrator Routing (Document Creation)**
    - Location: `graph/build_graph.py:70`
    - Maps `"create_document"` → `"DocumentCreator"` (line 84-86)
    - Routes to `"DocumentCreator"` node

13. **DocumentCreator Agent Execution**
    - Location: `graph/build_graph.py:62` → `RemoteGraph("document_creator", url="http://localhost:8002")`
    - HTTP POST to `http://localhost:8002/runs/stream`
    - DocumentCreator reads research summary from conversation history
    - Calls `write_markdown_file()` tool to create file
    - Returns completion contract:
      ```json
      {
        "completed_capability": "create_document",
        "data": {
          "file_path": "outputs/report_20260117_123456.md",
          "abs_file_path": "/Users/.../outputs/report_20260117_123456.md"
        }
      }
      ```
    - Graph edge returns to `"Supervisor"` (line 67)

14. **Third Supervisor Call (After Document Creation)**
    - Location: `agents/supervisor/supervisor.py:210`
    - Detects `completed_capability = "create_document"` (line 294)
    - Advances: `ctx["current_step_index"] = 2` (line 305)
    - Reads `plan[2] = "gmail"` (line 349)
    - Returns: `{"next": "gmail", "context": ctx}` (line 402)

15. **Orchestrator Routing (Gmail)**
    - Location: `graph/build_graph.py:70`
    - Maps `"gmail"` → `"Gmail"` (line 84-86)
    - Routes to `"Gmail"` node

16. **Gmail Agent Execution**
    - Location: `graph/build_graph.py:63` → `RemoteGraph("gmail", url="http://localhost:8000")`
    - HTTP POST to `http://localhost:8000/runs/stream`
    - Gmail agent reads `file_path` from DocumentCreator completion contract in conversation history
    - Calls `gmail_send()` tool with `attachments=[abs_file_path]`
    - Sends email to `eladrabinovitch1106@gmail.com`
    - Returns completion contract:
      ```json
      {
        "completed_capability": "gmail",
        "data": {
          "status": "sent",
          "to": "eladrabinovitch1106@gmail.com",
          "subject": "...",
          "tool_used": true
        }
      }
      ```
    - Graph edge returns to `"Supervisor"` (line 68)

17. **Fourth Supervisor Call (After Gmail)**
    - Location: `agents/supervisor/supervisor.py:210`
    - Detects `completed_capability = "gmail"` (line 294)
    - Advances: `ctx["current_step_index"] = 3` (line 305)
    - Checks: `idx (3) >= len(plan) (3)` → **True** (line 344)
    - Returns: `{"next": "FINISH", "context": ctx}` (line 346)

18. **Workflow Completion**
    - Location: `graph/build_graph.py:70`
    - `route_from_supervisor()` receives `"FINISH"` (line 80)
    - Routes to `END` (line 104)
    - `workflow.invoke()` returns final state to `main.py:67`
    - Prints completion message (line 69-74)

---

## 3. Capability Discovery Deep Dive

### Configuration Location

Agent service URLs are configured via the `AGENT_SERVICES` environment variable:
- **File**: `.env` (loaded by `load_dotenv()` in `main.py:7`)
- **Format**: Comma-separated URLs
- **Example**: `AGENT_SERVICES=http://localhost:8001,http://localhost:8002,http://localhost:8000,http://localhost:8004`

### Discovery Process

1. **Entry Point**: `graph/capability_discovery.py:33` (`discover_capabilities()`)
2. **Reads Environment**: `os.getenv("AGENT_SERVICES")` (line 56)
3. **For Each URL**: Calls `_fetch_agent_metadata(service_url)` (line 80)

### Metadata Fetching

**Primary Method**: Query `/graphs` endpoint
- Location: `graph/capability_discovery.py:156` (`_fetch_agent_metadata()`)
- HTTP GET: `{base_url}/graphs` (line 202)
- Expected response format:
  ```json
  {
    "graphs": [
      {
        "graph_id": "researcher",
        "metadata": {
          "agent_name": "Researcher",
          "capabilities": ["research"]
        }
      }
    ]
  }
  ```
- Validates: `metadata.agent_name` (non-empty string), `metadata.capabilities` (non-empty list[str]) (line 237-259)

**Fallback Method**: Port-based mapping
- Location: `graph/capability_discovery.py:289` (`_try_port_based_fallback()`)
- Triggered if `/graphs` endpoint fails (connection error, timeout, 404, etc.)
- Uses hardcoded `PORT_METADATA_MAP` (line 177-198):
  ```python
  {
    8001: {"agent_name": "Researcher", "capabilities": ["research"]},
    8002: {"agent_name": "DocumentCreator", "capabilities": ["create_document"]},
    8000: {"agent_name": "Gmail", "capabilities": ["gmail"]},
    8004: {"agent_name": "Supervisor", "capabilities": []}  # Supervisor has no capabilities
  }
  ```

### Capability Index Shape

**Return Value**: `Dict[str, str]` mapping capability → agent_name
- **Location**: `graph/capability_discovery.py:33` return type
- **Example for this run**:
  ```python
  {
    "research": "Researcher",
    "create_document": "DocumentCreator",
    "gmail": "Gmail"
  }
  ```
- **Storage**: Lives **only** in orchestrator process memory (`main.py` variable, passed to `build_graph()`)
- **No Shared Memory**: Each agent service has no knowledge of this mapping. The Supervisor never sees it.

---

## 4. Supervisor Deep Dive

### File Structure

- **Main Logic**: `agents/supervisor/supervisor.py` → `supervisor_node(state: AgentState)`
- **Graph Wrapper**: `agents/supervisor/graph.py` → `build_supervisor_graph()`
- **Schemas**: `agents/supervisor/schemas.py` → `Plan(steps: List[str])`
- **Utils**: `agents/supervisor/utils.py` → `extract_completed_capability()`
- **State**: `agents/supervisor/state.py` → `AgentState` TypedDict
- **Config**: `agents/supervisor/config.py` → `llm` (ChatOpenAI instance)

### Capability Reading

- **Location**: `agents/supervisor/supervisor.py:41`
- **Code**: `available_capabilities = ctx.get("capabilities", [])`
- **Source**: Injected by orchestrator in `init_state["context"]["capabilities"]` (`main.py:58`)
- **Format**: `["create_document", "gmail", "research"]` (sorted list of strings)
- **No Agent Names**: Supervisor never sees `"Researcher"`, `"DocumentCreator"`, or `"Gmail"`

### System Prompt Construction

- **Location**: `agents/supervisor/supervisor.py:128-164`
- **Dynamic Capability List**: `capabilities_text = "\n".join(f"- {c}" for c in sorted(available_capabilities))` (line 72)
- **JSON Escaping**: Uses `{{{{ "steps": [...] }}}}` (4 braces) because:
  - `f"{{{{"` → `{{` (escaped in f-string)
  - `ChatPromptTemplate` expects `{{` to render as literal `{`
  - Final prompt contains: `{{ "steps": [...] }}` (2 braces = JSON)
- **Example Generation**: Dynamically built from `available_capabilities` (line 94-126)

### Structured Output Parsing

- **Location**: `agents/supervisor/supervisor.py:175`
- **Code**: `plan_obj = (planning_prompt | llm.with_structured_output(Plan)).invoke(...)`
- **Schema**: `agents/supervisor/schemas.py:11` → `Plan(BaseModel): steps: List[str]`
- **Result**: Pydantic model with `plan_obj.steps = ["research", "create_document", "gmail"]`
- **Extraction**: `steps: List[str] = list(plan_obj.steps or [])` (line 179)

### Decision Logic

- **Empty Plan**: If `steps == []` (line 186)
  - LLM decided no capabilities needed
  - Returns direct answer via Supervisor's LLM (line 187-201)
  - Sets `"next": "FINISH"`
- **Non-Empty Plan**: If `steps != []` (line 203)
  - Stores plan: `ctx["plan"] = steps` (line 203)
  - Initializes tracking: `ctx["current_step_index"] = 0` (line 204)
  - Returns first capability: `{"next": steps[0], "context": ctx}` (line 402)

### Progress Tracking

- **Plan Storage**: `ctx["plan"] = ["research", "create_document", "gmail"]` (line 203)
- **Current Step**: `ctx["current_step_index"]` (line 204, 212, 305)
- **Completed Capabilities**: `ctx["completed_capabilities"]` (line 205, 213, 303)
- **Retry Count**: `ctx["agent_retry_count"]` (line 206, 214, 307)
- **Advancement**: When `extract_completed_capability()` finds a completion contract, increments `current_step_index` (line 305)

### Completion Contract Detection

- **Location**: `agents/supervisor/supervisor.py:293-296`
- **Function**: `extract_completed_capability(message_content)` from `agents/supervisor/utils.py:46`
- **Process**:
  1. Parses last agent message (AIMessage or dict) (line 238-291)
  2. Extracts content string
  3. Calls `parse_completion_message()` (`agents/supervisor/utils.py:11`)
  4. Tries `json.loads()` on full content (line 23)
  5. If fails, finds JSON boundaries `{...}` and extracts (line 30-37)
  6. Returns `contract.get("completed_capability")` (line 54)
- **Example**: `{"completed_capability": "research", "data": {...}}` → returns `"research"`

---

## 5. Remote Agent Independence & Boundaries

### Researcher Agent

- **Folder**: `agents/researcher/`
- **Standalone Command**: `cd agents/researcher && langgraph dev --port 8001`
- **Graph Definition**: `agents/researcher/graph.py` → `build_researcher_graph()`
- **LangGraph Config**: `agents/researcher/langgraph.json` → `"researcher": "graph:build_researcher_graph"`
- **Independence**:
  - Local config: `agents/researcher/config.py` (LLM instance)
  - Local state: `agents/researcher/state.py` (AgentState)
  - Local tools: `agents/researcher/tools.py` (web search tools)
  - No imports from root-level modules
- **HTTP Endpoint**: `POST http://localhost:8001/runs/stream`
- **Completion Contract**: Must return JSON with `"completed_capability": "research"`

### DocumentCreator Agent

- **Folder**: `agents/document_creator/`
- **Standalone Command**: `cd agents/document_creator && langgraph dev --port 8002`
- **Graph Definition**: `agents/document_creator/graph.py` → `build_document_creator_graph()`
- **LangGraph Config**: `agents/document_creator/langgraph.json` → `"document_creator": "graph:build_document_creator_graph"`
- **Independence**:
  - Local config: `agents/document_creator/config.py`
  - Local state: `agents/document_creator/state.py`
  - Local tools: `agents/document_creator/tools.py` (includes `write_markdown_file()`)
- **HTTP Endpoint**: `POST http://localhost:8002/runs/stream`
- **Completion Contract**: Must return JSON with `"completed_capability": "create_document"` and `"data.file_path"`

### Gmail Agent

- **Folder**: `agents/gmail/`
- **Standalone Command**: `cd agents/gmail && langgraph dev --port 8000`
- **Graph Definition**: `agents/gmail/graph.py` → `build_gmail_graph()`
- **LangGraph Config**: `agents/gmail/langgraph.json` → `"gmail": "graph:build_gmail_graph"`
- **Independence**:
  - Local config: `agents/gmail/config.py`
  - Local state: `agents/gmail/state.py`
  - Local tools: `agents/gmail/tools.py` (includes `gmail_send()`, `gmail_search()`)
  - OAuth credentials: `agents/gmail/gmail_auth.py`
- **HTTP Endpoint**: `POST http://localhost:8000/runs/stream`
- **Completion Contract**: Must return JSON with `"completed_capability": "gmail"` and `"data.status"`

### Supervisor Agent

- **Folder**: `agents/supervisor/`
- **Standalone Command**: `cd agents/supervisor && langgraph dev --port 8004`
- **Graph Definition**: `agents/supervisor/graph.py` → `build_supervisor_graph()`
- **LangGraph Config**: `agents/supervisor/langgraph.json` → `"supervisor": "graph:build_supervisor_graph"`
- **Independence**:
  - Local config: `agents/supervisor/config.py`
  - Local state: `agents/supervisor/state.py`
  - Local schemas: `agents/supervisor/schemas.py`
  - Local utils: `agents/supervisor/utils.py`
- **HTTP Endpoint**: `POST http://localhost:8004/runs/stream`
- **Output**: Returns `{"next": "<capability_string>"}` or `{"next": "FINISH"}`

---

## 6. HTTP Call Chain

### Call Location

- **Orchestrator → Supervisor**: `graph/build_graph.py:60` → `RemoteGraph("supervisor", url="http://localhost:8004")`
- **Orchestrator → Researcher**: `graph/build_graph.py:61` → `RemoteGraph("researcher", url="http://localhost:8001")`
- **Orchestrator → DocumentCreator**: `graph/build_graph.py:62` → `RemoteGraph("document_creator", url="http://localhost:8002")`
- **Orchestrator → Gmail**: `graph/build_graph.py:63` → `RemoteGraph("gmail", url="http://localhost:8000")`

### Request Details

- **URL**: `http://localhost:{PORT}/runs/stream` (LangServe streaming endpoint)
- **Method**: HTTP POST
- **Payload**: Full `AgentState` serialized as JSON:
  ```json
  {
    "messages": [
      {"type": "human", "content": "research the company nvidia..."},
      {"type": "ai", "content": "..."}
    ],
    "next": "Supervisor",
    "context": {
      "capabilities": ["create_document", "gmail", "research"],
      "plan": ["research", "create_document", "gmail"],
      "current_step_index": 1,
      "completed_capabilities": ["research"]
    }
  }
  ```

### Response Handling

- **Format**: Streaming response from LangServe (or blocking if using `/runs/invoke`)
- **Content**: Updated `AgentState` with agent's response:
  - New `AIMessage` in `messages` array
  - Completion contract embedded in message content
- **State Merge**: LangGraph's `RemoteGraph` automatically merges response into global state
- **Next Call**: Supervisor receives updated state with agent's message appended

### Example Flow

1. **Supervisor → Researcher**:
   - Request: `state["messages"]` = `[HumanMessage("research nvidia...")]`
   - Response: `state["messages"]` = `[HumanMessage(...), AIMessage('{"completed_capability": "research", ...}')]`
   - State merged: Next Supervisor call sees research completion

2. **Supervisor → DocumentCreator**:
   - Request: `state["messages"]` includes research summary
   - Response: `state["messages"]` includes document creation completion with `file_path`
   - State merged: Next Supervisor call sees document completion

3. **Supervisor → Gmail**:
   - Request: `state["messages"]` includes `file_path` from DocumentCreator
   - Response: `state["messages"]` includes email send completion
   - State merged: Next Supervisor call sees gmail completion

---

## 7. Where to See Logs

### Orchestrator Process (main.py)

- **Terminal**: Where you run `python main.py`
- **Logs Include**:
  - Capability discovery: `INFO:graph.capability_discovery:Discovered agent Researcher with capabilities ['research']`
  - Graph building: `INFO:graph.build_graph:Graph built with capability routing: 3 capabilities mapped to agents`
  - Workflow execution: `INFO:__main__:Starting workflow execution...`
  - HTTP requests: `INFO:httpx:HTTP Request: POST http://localhost:8004/runs/stream "HTTP/1.1 200 OK"`

### Supervisor Service

- **Terminal**: `cd agents/supervisor && langgraph dev --port 8004`
- **Logs Include**:
  - Planning: `INFO:supervisor:Supervisor: LLM returned plan with 3 steps: ['research', 'create_document', 'gmail']`
  - Progress: `INFO:supervisor:Supervisor: Agent completed 'research'. Advancing to step 1/3`
  - Routing: `INFO:supervisor:Supervisor: Dispatching step 2/3: capability=create_document`

### Researcher Service

- **Terminal**: `cd agents/researcher && langgraph dev --port 8001`
- **Logs Include**:
  - Tool usage: Tool execution logs from web search
  - Completion: Agent returning completion contract

### DocumentCreator Service

- **Terminal**: `cd agents/document_creator && langgraph dev --port 8002`
- **Logs Include**:
  - File creation: `write_markdown_file()` tool execution
  - File path: Logged file path in completion contract

### Gmail Service

- **Terminal**: `cd agents/gmail && langgraph dev --port 8000`
- **Logs Include**:
  - Email sending: `gmail_send()` tool execution
  - Attachment handling: File path resolution and attachment inclusion

### Debugging Tips

- **If Supervisor returns empty plan**: Check Supervisor logs for LLM reasoning
- **If routing fails**: Check orchestrator logs for `Unknown capability` errors
- **If agent doesn't complete**: Check agent logs for completion contract format
- **If HTTP calls fail**: Check service is running and port matches environment variable

---

## 8. Appendix: Quick Commands

### Start All Services

**Terminal 1 - Supervisor**:
```bash
cd agents/supervisor
langgraph dev --port 8004
```

**Terminal 2 - Researcher**:
```bash
cd agents/researcher
langgraph dev --port 8001
```

**Terminal 3 - DocumentCreator**:
```bash
cd agents/document_creator
langgraph dev --port 8002
```

**Terminal 4 - Gmail**:
```bash
cd agents/gmail
langgraph dev --port 8000
```

**Terminal 5 - Orchestrator**:
```bash
python main.py
```

### Required Environment Variables

Create `.env` file in project root:
```bash
# Agent service URLs for capability discovery
AGENT_SERVICES=http://localhost:8001,http://localhost:8002,http://localhost:8000,http://localhost:8004

# Optional: Override default service URLs (if different ports)
RESEARCHER_SERVICE_URL=http://localhost:8001
DOCUMENT_CREATOR_SERVICE_URL=http://localhost:8002
GMAIL_SERVICE_URL=http://localhost:8000
SUPERVISOR_SERVICE_URL=http://localhost:8004

# OpenAI API key (required for all agents)
OPENAI_API_KEY=sk-...

# Gmail OAuth credentials (required for Gmail agent)
# See agents/gmail/gmail_auth.py for setup
```

### Verification

After starting all services, verify they're running:
```bash
# Check Supervisor
curl http://localhost:8004/graphs

# Check Researcher
curl http://localhost:8001/graphs

# Check DocumentCreator
curl http://localhost:8002/graphs

# Check Gmail
curl http://localhost:8000/graphs
```

Each should return JSON with `metadata.agent_name` and `metadata.capabilities`.

---

## Summary

This flow demonstrates a **fully distributed, capability-driven multi-agent system** where:
- The orchestrator discovers capabilities at runtime
- The Supervisor plans using only capability strings (never agent names)
- The orchestrator routes capability strings to agent names
- Each agent is fully independent and self-contained
- No shared memory exists between services
- All communication is via HTTP using LangServe/LangGraph protocols

The NVIDIA request flows through: **Supervisor (plan) → Researcher → Supervisor (advance) → DocumentCreator → Supervisor (advance) → Gmail → Supervisor (finish)**.
