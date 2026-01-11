"""
FastAPI service for Researcher Agent using LangServe.

This service exposes the Researcher agent as a REST API with:
- LangServe endpoints for full agent interaction
- Chat endpoint for natural language interaction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agents.researcher_agent import build_researcher_agent


# Request/Response Models
class AgentChatRequest(BaseModel):
    """Request model for agent chat interaction."""
    message: str


class AgentChatResponse(BaseModel):
    """Response model for agent chat interaction."""
    response: str


# Initialize FastAPI app
app = FastAPI(
    title="Researcher Agent Service",
    description="REST API for Researcher Agent using LangServe",
    version="1.0.0",
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the agent chain
researcher_agent_chain = build_researcher_agent()

# Add LangServe routes for the agent chain
# This exposes standard LangServe endpoints like /invoke, /stream, etc.
add_routes(
    app,
    researcher_agent_chain,
    path="/agent",
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Researcher Agent Service",
        "status": "running",
        "endpoints": {
            "agent": "/agent/invoke - Full agent interaction via LangServe",
            "chat": "/chat - Simple chat interface with agent",
        }
    }


@app.post("/chat", response_model=AgentChatResponse)
async def chat_with_agent(request: AgentChatRequest):
    """
    Simple chat endpoint that uses the full agent chain.
    
    This endpoint uses the agent's reasoning capabilities to handle
    natural language requests about research and web search.
    """
    try:
        # Invoke the agent with the user's message
        result = researcher_agent_chain.invoke({
            "messages": [HumanMessage(content=request.message)]
        })
        
        # Extract the response from the agent's result
        # The result is typically a message object
        if hasattr(result, 'content'):
            response_text = result.content
        elif isinstance(result, dict) and 'messages' in result:
            # Handle case where result is a dict with messages
            messages = result['messages']
            if messages and hasattr(messages[-1], 'content'):
                response_text = messages[-1].content
            else:
                response_text = str(result)
        else:
            response_text = str(result)
        
        return AgentChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
