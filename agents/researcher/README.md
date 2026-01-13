# Researcher Agent Service

A FastAPI service exposing the Researcher Agent using LangServe. This service provides LangServe endpoints for full agent interaction and a chat endpoint for natural language queries.

## Features

- **LangServe Integration**: Full agent chain exposed via LangServe endpoints (`/agent/invoke`, `/agent/stream`, etc.)
- **Chat Interface**: Natural language chat endpoint that uses the agent's reasoning capabilities
- **Web Search**: Uses Tavily search tool for external information gathering
- **CORS Enabled**: Ready for frontend integration

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

Ensure you have the `TAVILY_API_KEY` environment variable set for web search functionality.

## Running the Service

### Option 1: Using the startup script

```bash
python agents/researcher/run_service.py
```

### Option 2: Using uvicorn directly

```bash
uvicorn agents.researcher.service:app --host 0.0.0.0 --port 8001 --reload
```

The service will be available at `http://localhost:8001`

## API Endpoints

### Health Check
- **GET** `/` - Service health and endpoint information

### LangServe Agent Endpoints
- **POST** `/agent/invoke` - Invoke the agent chain
- **POST** `/agent/stream` - Stream agent responses
- **GET** `/agent/playground` - Interactive playground (if enabled)

### Chat Endpoint

#### Chat with Agent
- **POST** `/chat`
  ```json
  {
    "message": "What are the latest developments in AI?"
  }
  ```

## Example Usage

### Using curl

```bash
# Chat with agent
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Research the latest trends in machine learning"
  }'
```

### Using Python requests

```python
import requests

BASE_URL = "http://localhost:8001"

# Chat with agent
response = requests.post(
    f"{BASE_URL}/chat",
    json={"message": "What are the benefits of renewable energy?"}
)
print(response.json())
```

## Environment Variables

Make sure these are set:
- `TAVILY_API_KEY`: API key for Tavily search service
- `OPENAI_API_KEY`: For the LLM

## API Documentation

Once the service is running, visit:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`
