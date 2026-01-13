# Gmail Agent Service

A FastAPI service exposing the Gmail Agent using LangServe. This service provides both LangServe endpoints for full agent interaction and direct tool endpoints for programmatic access.

## Features

- **LangServe Integration**: Full agent chain exposed via LangServe endpoints (`/agent/invoke`, `/agent/stream`, etc.)
- **Direct Tool Endpoints**: Simple REST endpoints for `send-email` and `read-email`
- **Chat Interface**: Natural language chat endpoint that uses the agent's reasoning capabilities
- **CORS Enabled**: Ready for frontend integration

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

Ensure you have Gmail OAuth credentials set up (see `gmail_auth.py`).

## Running the Service

### Option 1: Using the startup script

```bash
python agents/gmail/run_service.py
```

### Option 2: Using uvicorn directly

```bash
uvicorn agents.gmail.service:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- **GET** `/` - Service health and endpoint information

### LangServe Agent Endpoints
- **POST** `/agent/invoke` - Invoke the agent chain
- **POST** `/agent/stream` - Stream agent responses
- **GET** `/agent/playground` - Interactive playground (if enabled)

### Direct Tool Endpoints

#### Send Email
- **POST** `/send-email`
  ```json
  {
    "recipient": "user@example.com",
    "subject": "Test Email",
    "body": "This is a test email",
    "attachments": ["/path/to/file.pdf"]  // optional
  }
  ```

#### Read/Search Email
- **POST** `/read-email`
  ```json
  {
    "query": "from:example@gmail.com subject:important"
  }
  ```

#### Chat with Agent
- **POST** `/chat`
  ```json
  {
    "message": "Send an email to john@example.com about the meeting"
  }
  ```

## Example Usage

### Using curl

```bash
# Send an email
curl -X POST "http://localhost:8000/send-email" \
  -H "Content-Type: application/json" \
  -d '{
    "recipient": "user@example.com",
    "subject": "Hello",
    "body": "This is a test email"
  }'

# Search emails
curl -X POST "http://localhost:8000/read-email" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "from:example@gmail.com"
  }'

# Chat with agent
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find emails from last week about the project"
  }'
```

### Using Python requests

```python
import requests

BASE_URL = "http://localhost:8000"

# Send email
response = requests.post(
    f"{BASE_URL}/send-email",
    json={
        "recipient": "user@example.com",
        "subject": "Test",
        "body": "Hello from the API"
    }
)
print(response.json())

# Search emails
response = requests.post(
    f"{BASE_URL}/read-email",
    json={"query": "from:example@gmail.com"}
)
print(response.json())

# Chat with agent
response = requests.post(
    f"{BASE_URL}/chat",
    json={"message": "Send an email to john@example.com"}
)
print(response.json())
```

## Integration with LangGraph

This service can be integrated into your existing LangGraph setup by:

1. **Replacing the agent node**: Instead of calling `gmail_node` directly, make HTTP requests to this service
2. **Using as a microservice**: Deploy this service separately and have your orchestrator call it via HTTP
3. **Hybrid approach**: Use direct tool endpoints for simple operations and the agent endpoint for complex reasoning

### Example Integration

```python
import requests
from models.state import AgentState

def gmail_node_via_api(state: AgentState):
    """Gmail node that calls the API service instead of running locally."""
    # Extract the user's message
    user_message = state["messages"][-1].content
    
    # Call the agent service
    response = requests.post(
        "http://localhost:8000/chat",
        json={"message": user_message}
    )
    
    result = response.json()
    
    # Return in the same format as the original node
    from langchain_core.messages import AIMessage
    return {
        "messages": [AIMessage(content=result["response"])],
        "context": state.get("context", {}),
    }
```

## Environment Variables

Make sure these are set:
- `GMAIL_TOKEN_PATH`: Path to Gmail OAuth token (default: `token.json`)
- `GMAIL_SENDER_ADDRESS`: Email address to send from
- `OPENAI_API_KEY`: For the LLM

## API Documentation

Once the service is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
