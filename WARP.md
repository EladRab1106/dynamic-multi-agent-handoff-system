# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

This repo implements a distributed, capability-driven multi-agent system built on LangGraph / LangChain. A local orchestrator (`main.py` + `graph/`) coordinates remote agent services (Researcher, DocumentCreator, Gmail, DirectAnswer, Supervisor), which each run as independent LangGraph servers or FastAPI/LangServe services.

Each agent is responsible for a specific **capability** and communicates completion via a JSON "completion contract" embedded in its final `AIMessage`.

---

## Common commands

All commands assume you are at the repo root: `dynamic-multi-agent-handoff-system`.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment configuration

Create a root `.env` (loaded by `main.py` and several configs) with at least:

```bash
# Multi-agent capability discovery (comma-separated LangGraph server URLs)
AGENT_SERVICES=http://localhost:8001,http://localhost:8002,http://localhost:8000,http://localhost:8004

# Optional overrides for orchestrator → agent URLs
RESEARCHER_SERVICE_URL=http://localhost:8001
DOCUMENT_CREATOR_SERVICE_URL=http://localhost:8002
GMAIL_SERVICE_URL=http://localhost:8000
SUPERVISOR_SERVICE_URL=http://localhost:8004

# Shared LLM key
OPENAI_API_KEY=...

# Researcher-specific
TAVILY_API_KEY=...

# Gmail shared tool (root-level tools/gmail_tool.py)
GMAIL_TOKEN_PATH=token.json
GMAIL_SENDER_ADDRESS=you@example.com
```

Some agents also load their own `.env` from their directory (for example `agents/gmail/config.py`); see those configs if you change per-agent behavior.

### Run the distributed multi-agent flow (LangGraph servers + orchestrator)

**1. Start Supervisor LangGraph server**

```bash
cd agents/supervisor
langgraph dev --port 8004
```

**2. Start worker agents as LangGraph servers (each in its own terminal):**

```bash
# Researcher
cd agents/researcher
langgraph dev --port 8001

# Document Creator
cd agents/document_creator
langgraph dev --port 8002

# Gmail
cd agents/gmail
langgraph dev --port 8000

# Direct Answer (optional capability)
cd agents/direct_answer
langgraph dev --port 8003
```

Each agent directory has a `langgraph.json` that maps its graph ID to `graph.build_*_graph`.

**3. Run the orchestrator CLI**

From the repo root, in a separate terminal:

```bash
python main.py
```

`main.py` will:
- Discover capabilities from all configured agent services via `graph/capability_discovery.py`.
- Build the orchestration graph via `graph/build_graph.py`.
- Prompt you for a natural-language request and run the full multi-agent workflow.

### Run individual agent HTTP services (FastAPI + LangServe)

Each agent also exposes a FastAPI service (useful for local debugging or direct integration). These **do not** participate in `AGENT_SERVICES` discovery; they are separate from the LangGraph servers above.

From repo root:

- **Researcher**
  - Startup helper:
    ```bash
    python agents/researcher/run_service.py
    ```
  - Or directly with `uvicorn` (from repo root):
    ```bash
    uvicorn agents.researcher.service:app --host 0.0.0.0 --port 8001 --reload
    ```

- **Document Creator**
  - Startup helper:
    ```bash
    python agents/document_creator/run_service.py
    ```
  - Or:
    ```bash
    uvicorn agents.document_creator.service:app --host 0.0.0.0 --port 8002 --reload
    ```

- **Gmail**
  - Startup helper:
    ```bash
    python agents/gmail/run_service.py
    ```
  - Or:
    ```bash
    uvicorn agents.gmail.service:app --host 0.0.0.0 --port 8000 --reload
    ```
  - Requires Gmail OAuth credentials as described in `agents/gmail/README.md` and `agents/gmail/gmail_auth.py`.

- **Direct Answer**
  - Startup helper:
    ```bash
    python agents/direct_answer/run_service.py
    ```
  - Or:
    ```bash
    uvicorn agents.direct_answer.service:app --host 0.0.0.0 --port 8003 --reload
    ```

Once running, each service exposes:
- `GET /` health/metadata summary
- LangServe chain/graph endpoints under `/agent/*` and `/graph/*`
- A convenience `POST /chat` endpoint (see each agent README for payload examples)

### Tests

System tests live under `tests/system/` and exercise agents over HTTP.

**Run all system tests** (requires relevant services to be running, and may skip Gmail-related tests if Gmail is not configured):

```bash
pytest tests/system
```

**Run a single test module** (for quicker iteration):

```bash
# Full multi-step flow: research → document → gmail
pytest tests/system/test_full_flow.py

# Researcher only
pytest tests/system/test_researcher_agent.py

# Document Creator only
pytest tests/system/test_document_creator_agent.py

# Gmail only
pytest tests/system/test_gmail_agent.py

# Direct Answer only
pytest tests/system/test_direct_answer_agent.py
```

**Run a specific test function** (pytest node ID):

```bash
pytest tests/system/test_full_flow.py::test_full_flow_research_to_document
```

> Note: `pytest` is not pinned in `requirements.txt` and may need to be installed separately in your environment.

### Docker Compose

There is a `docker-compose.yml` and per-agent Dockerfiles under `agents/*/Dockerfile`. They build images that run the agents using their `run_service` entrypoints. Use this when you want containerized agent services rather than local Python processes:

```bash
docker-compose up --build
```

This will start containers exposing ports 8000 (Gmail), 8001 (Researcher), 8002 (DocumentCreator), 8003 (DirectAnswer). The Supervisor orchestrator is not part of the current `docker-compose.yml` and should be run separately if needed.

---

## High-level architecture

### 1. Orchestrator process (root-level graph)

**Entry point**: `main.py`

- Loads environment (`dotenv`), then calls `graph.capability_discovery.discover_capabilities()` to discover all running agent services and their capabilities via each service's `/graphs` endpoint, with a port-based fallback.
- Builds a `capability_index: Dict[capability, agent_name]` such as `{"research": "Researcher", "create_document": "DocumentCreator", "gmail": "Gmail"}`.
- Calls `graph.build_graph.build_graph(capability_index)` to assemble the runtime `StateGraph[AgentState]`.
- Creates the initial `AgentState` with:
  - `messages`: a single `HumanMessage` containing the user's request.
  - `next`: `"Supervisor"` (graph entry node).
  - `context.capabilities`: sorted list of capability strings (not agent names).
- Invokes the graph with `workflow.invoke(init_state)` and prints the final state summary.

**Graph composition** (`graph/build_graph.py`):

- Defines nodes as `RemoteGraph` wrappers pointing at each remote LangGraph service:
  - `Supervisor` → `SUPERVISOR_SERVICE_URL` (default `http://localhost:8004`)
  - `Researcher` → `RESEARCHER_SERVICE_URL` (default `http://localhost:8001`)
  - `DocumentCreator` → `DOCUMENT_CREATOR_SERVICE_URL` (default `http://localhost:8002`)
  - `Gmail` → `GMAIL_SERVICE_URL` (default `http://localhost:8000`)
- Sets `Supervisor` as entry point.
- Adds edges so that each worker agent returns to `Supervisor` after completing its work.
- Uses `route_from_supervisor(state)` as a conditional edge function:
  - Interprets `state["next"]` as **either**:
    - A capability string like `"research"`, `"create_document"`, `"gmail"`, which is mapped via `capability_index` to an agent node name (e.g. `"Researcher"`).
    - Or the sentinel `"FINISH"`, which routes to `END`.

**State model** (`models/state.py`):

- `AgentState` is a TypedDict with:
  - `messages`: a LangGraph-accumulated sequence of `BaseMessage`.
  - `next`: routing hint used only by the Supervisor and orchestrator.
  - `context`: arbitrary dict carrying plan, capabilities, and progress metadata.

### 2. Capability discovery

**Module**: `graph/capability_discovery.py`

- Reads `AGENT_SERVICES` from the environment (comma-separated URLs) to know which LangGraph servers to query.
- For each service URL:
  - Fetches `GET {base_url}/graphs`.
  - Searches each graph entry for `metadata` containing `agent_name` and non-empty list of `capabilities`.
  - Falls back to a hard-coded port → metadata map if `/graphs` is unavailable (helpful during migration).
- Builds and returns `capability_index: Dict[capability, agent_name]` used by the orchestrator and Supervisor.
- This mechanism is the **only** source of truth for which capabilities exist and which agents provide them.

### 3. Supervisor agent (planning and routing decisions)

**Location**: `agents/supervisor/`

- `supervisor.py` defines `supervisor_node(state: AgentState)`, which encapsulates planning and progress tracking.
- `graph.py` wraps `supervisor_node` into a single-node LangGraph, compiled and exposed as a graph named `"supervisor"` via `langgraph.json`.
- `schemas.py` defines a Pydantic `Plan` model with `steps: List[str]` for structured output.
- `state.py` defines a Supervisor-local `AgentState` (aligned with the root `models/state.AgentState`).

**Key behaviors:**

- On the first invocation (no `ctx["plan"]` yet):
  - Reads `ctx["capabilities"]` (strings only; no URLs or agent names).
  - Extracts the original human request from `state["messages"]`.
  - Builds a system prompt that:
    - Lists available capability strings.
    - Instructs the LLM to return **only** JSON `{"steps": ["capability_1", ...]}`.
  - Calls `llm.with_structured_output(Plan)` to obtain an ordered capability list.
  - If `steps` is empty, Supervisor answers directly and sets `next = "FINISH"`.
  - Otherwise, stores the plan and initializes:
    - `context["plan"]`, `context["current_step_index"]`, `context["completed_capabilities"]`, `context["agent_retry_count"]`.
  - Returns `{"next": first_capability, "context": ctx}`.

- On subsequent invocations:
  - Looks at the latest agent `AIMessage` in `messages`.
  - Uses `agents/utils.parse_completion_message` / `extract_completed_capability` to parse the JSON completion contract and extract `completed_capability`.
  - If the capability for the current step is marked complete, advances `current_step_index` and updates bookkeeping.
  - If all steps are done, returns `{"next": "FINISH", "context": ctx}`.
  - Otherwise, returns `{"next": <next_capability>, "context": ctx}` so the orchestrator can route to the appropriate agent.

Supervisor **never** sees agent names or URLs; it operates purely over capability strings and completion contracts.

### 4. Per-agent services and graphs

Each agent directory under `agents/` follows a similar pattern and is intentionally self-contained so it can be moved to its own repo if needed.

Common structure (example: `agents/researcher/`):

- `config.py` – local LLM configuration, usually a `ChatOpenAI` instance. Some agents (e.g. Gmail) also load and validate service-specific credentials here.
- `state.py` – agent-local `AgentState` TypedDict mirroring the root structure.
- `base_agent.py` – a capability-agnostic runtime that wires prompts, tools, and a tool-execution loop, returning `AIMessage` instances.
- `*_agent.py` – the core agent definition (system prompt + tool bindings + optional tool-usage tracking). Each agent is responsible for producing a valid completion contract JSON in its final `AIMessage.content`.
- `tools.py` (where present) – LangChain/`@tool`-decorated tool functions implementing the agent's side effects (e.g. web search for Researcher, Markdown file writing for DocumentCreator, Gmail send/search for Gmail).
- `graph.py` – wraps the agent chain in a LangGraph `StateGraph` to:
  - Reset per-call tool-usage trackers.
  - Invoke the agent with `AgentState["messages"]`.
  - Optionally post-process the completion contract (e.g. annotating `data.tool_used`).
- `service.py` – FastAPI + LangServe application exposing both the chain and graph as HTTP endpoints for standalone use.
- `langgraph.json` – declares the agent graph(s) and dependencies so `langgraph dev`/`langgraph serve` can run it as a server.

Notable agent-specific behavior:

- **Researcher** (`agents/researcher/`)
  - Uses Tavily-based tools to perform web research.
  - Expected completion contract: `{"completed_capability": "research", "data": {"research_summary": {...}, ...}}`.

- **Document Creator** (`agents/document_creator/`)
  - Uses a `write_markdown_file` tool to create a report on disk and return both relative and absolute file paths.
  - Completion contract: `{"completed_capability": "create_document", "data": {"file_path": ..., "abs_file_path": ...}}`.

- **Gmail** (`agents/gmail/`)
  - Provides tools for Gmail search and send, plus a helper to read local file content.
  - System prompt enforces a two-phase behavior: intent classification (search vs send) followed by the appropriate tool call.
  - Completion contracts distinguish between search vs send and whether attachments were actually used.

- **Direct Answer** (`agents/direct_answer/`)
  - Minimal agent with no tools that always returns a strict completion contract:
    `{"completed_capability": "direct_answer", "data": {"answer": "..."}}`.

### 5. Completion contract protocol

Cross-agent orchestration relies on a simple but strict convention:

- Agents must end their run with an `AIMessage` whose `.content` is (or contains) JSON of the form:

  ```json
  {
    "completed_capability": "<capability_name>",
    "data": { ... }
  }
  ```

- `agents/utils.py` provides helpers to:
  - Build such messages (`build_completion_message`).
  - Parse them even if embedded in extra text (`parse_completion_message`).
  - Extract the `completed_capability` for Supervisor progress tracking.

System tests under `tests/system/` assert that each agent returns a valid completion contract with the expected `completed_capability` and key `data` fields.

---

## How future Warp agents should use this file

- Prefer running the full LangGraph-based, capability-discovery flow (`langgraph dev` per agent + `python main.py`) when working on orchestrator or cross-agent behavior.
- Use the per-agent FastAPI/LangServe services (via `uvicorn` or the `run_service.py` helpers) when debugging or modifying a single agent in isolation.
- When making changes that affect completion contracts or capabilities, update:
  - The relevant agent's system prompt and tools.
  - Any tests under `tests/system/` that depend on the contract shape.
  - If you add a new capability/agent, ensure its LangGraph server exposes proper metadata via `/graphs` so `graph/capability_discovery.py` can discover it.
