# Document Creator Agent Service

A FastAPI service exposing the Document Creator Agent using LangServe. This service provides LangServe endpoints for full agent interaction and a chat endpoint for document creation.

## Features

- **LangServe Integration**: Full agent chain exposed via LangServe endpoints (`/agent/invoke`, `/agent/stream`, etc.)
- **Chat Interface**: Natural language chat endpoint that uses the agent's reasoning capabilities
- **Document Creation**: Converts research data into clean Markdown reports
- **File Output**: Automatically saves created documents to the `outputs/` directory
- **CORS Enabled**: Ready for frontend integration

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Running the Service

### Option 1: Using the startup script

```bash
python agents/document_creator/run_service.py
```

### Option 2: Using uvicorn directly

```bash
uvicorn agents.document_creator.service:app --host 0.0.0.0 --port 8002 --reload
```

The service will be available at `http://localhost:8002`

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
    "message": "Convert the research data into a markdown report"
  }
  ```

  Response includes:
  - `response`: Status message
  - `file_path`: Path to the created markdown file

## Example Usage

### Using curl

```bash
# Create a document
curl -X POST "http://localhost:8002/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a markdown report from the research findings"
  }'
```

### Using Python requests

```python
import requests

BASE_URL = "http://localhost:8002"

# Create a document
response = requests.post(
    f"{BASE_URL}/chat",
    json={"message": "Convert the research data into a clean markdown report"}
)
result = response.json()
print(f"Response: {result['response']}")
print(f"File: {result['file_path']}")
```

## Environment Variables

Make sure these are set:
- `OPENAI_API_KEY`: For the LLM

## Output Directory

Created documents are saved to the `outputs/` directory with filenames like:
- `report_20240109_153318.md`

## API Documentation

Once the service is running, visit:
- Swagger UI: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`
