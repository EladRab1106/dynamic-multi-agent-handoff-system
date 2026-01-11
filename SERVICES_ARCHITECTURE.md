# Multi-Agent Services Architecture

This document describes the microservices architecture for the multi-agent system. All agents now run as independent FastAPI services using LangServe.

## Architecture Overview

The system has been converted from a monolithic architecture to a microservices architecture where each agent runs as its own API service:

- **Gmail Agent**: `http://localhost:8000`
- **Researcher Agent**: `http://localhost:8001`
- **Document Creator Agent**: `http://localhost:8002`
- **Direct Answer Agent**: `http://localhost:8003`

The LangGraph orchestrator makes HTTP requests to these services instead of calling agent functions directly.

## Service Structure

Each agent service follows the same structure:

```
agents/
  <agent_name>/
    service.py          # FastAPI service with LangServe integration
    run_service.py      # Startup script
    example_client.py   # Example Python client
    README.md           # Service documentation
```

## Running the Services

### Start All Services

You can start each service in separate terminals:

```bash
# Terminal 1: Gmail Agent
python agents/gmail/run_service.py

# Terminal 2: Researcher Agent
python agents/researcher/run_service.py

# Terminal 3: Document Creator Agent
python agents/document_creator/run_service.py

# Terminal 4: Direct Answer Agent
python agents/direct_answer/run_service.py
```

### Using Docker Compose (Future)

You can also run all services using Docker Compose (configuration can be added).

## Service Endpoints

Each service exposes:

- **LangServe Endpoints**:
  - `POST /agent/invoke` - Invoke the agent chain
  - `POST /agent/stream` - Stream agent responses
  - `GET /agent/playground` - Interactive playground

- **Custom Endpoints**:
  - `GET /` - Health check
  - `POST /chat` - Simple chat interface

- **API Documentation**:
  - `GET /docs` - Swagger UI
  - `GET /redoc` - ReDoc documentation

## LangGraph Integration

The LangGraph in `graph/build_graph.py` now uses HTTP adapters for all agents:

- `researcher_node()` → calls `http://localhost:8001/agent/invoke`
- `gmail_node()` → calls `http://localhost:8000/agent/invoke`
- `document_creator_node()` → calls `http://localhost:8002/agent/invoke`
- `direct_answer_node()` → calls `http://localhost:8003/agent/invoke`

### Environment Variables

You can configure service URLs via environment variables:

```bash
export GMAIL_SERVICE_URL=http://localhost:8000
export RESEARCHER_SERVICE_URL=http://localhost:8001
export DOCUMENT_CREATOR_SERVICE_URL=http://localhost:8002
export DIRECT_ANSWER_SERVICE_URL=http://localhost:8003
```

### Fallback Behavior

All HTTP adapters include fallback to local execution if the API service is unavailable. This provides resilience during development and deployment.

## Running the Full System

1. **Start all agent services** (in separate terminals or as background processes)

2. **Run the LangGraph orchestrator**:
   ```bash
   python main.py
   ```

The graph will automatically route requests to the appropriate agent services via HTTP.

## Benefits

1. **Scalability**: Each agent can be scaled independently
2. **Isolation**: Agent failures don't crash the entire system
3. **Flexibility**: Agents can be deployed separately or replaced
4. **Testing**: Each service can be tested independently
5. **Monitoring**: Each service can be monitored separately

## Development Workflow

1. Make changes to an agent's service code
2. Restart that specific service
3. The LangGraph will automatically use the updated service

## Production Considerations

- Configure CORS appropriately for production
- Add authentication/authorization as needed
- Use environment variables for all configuration
- Set up proper logging and monitoring
- Consider using a service mesh or API gateway
- Implement health checks and circuit breakers
