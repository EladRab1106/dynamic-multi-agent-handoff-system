# Direct Answer Agent Service

A FastAPI service exposing the Direct Answer Agent using LangServe. This service provides LangServe endpoints for full agent interaction and a chat endpoint for direct answers.

## Features

- **LangServe Integration**: Full agent chain exposed via LangServe endpoints (`/agent/invoke`, `/agent/stream`, etc.)
- **Chat Interface**: Natural language chat endpoint that uses the agent's reasoning capabilities
- **Direct Answers**: Provides concise and helpful responses without requiring tools
- **CORS Enabled**: Ready for frontend integration

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Running the Service

### Option 1: Using the startup script

```bash
python agents/direct_answer/run_service.py
```

### Option 2: Using uvicorn directly

```bash
uvicorn agents.direct_answer.service:app --host 0.0.0.0 --port 8003 --reload
```

The service will be available at `http://localhost:8003`

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
    "message": "What is the capital of France?"
  }
  ```

## Example Usage

### Using curl

```bash
# Chat with agent
curl -X POST "http://localhost:8003/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing in simple terms"
  }'
```

### Using Python requests

```python
import requests

BASE_URL = "http://localhost:8003"

# Chat with agent
response = requests.post(
    f"{BASE_URL}/chat",
    json={"message": "What is machine learning?"}
)
print(response.json())
```

## Environment Variables

Make sure these are set:
- `OPENAI_API_KEY`: For the LLM

## API Documentation

Once the service is running, visit:
- Swagger UI: `http://localhost:8003/docs`
- ReDoc: `http://localhost:8003/redoc`
