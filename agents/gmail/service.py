"""
FastAPI service for Gmail Agent using LangServe.

This service exposes the Gmail agent as a REST API with:
- LangServe endpoints for full agent interaction
- Direct tool endpoints for send-email and read-email
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import List, Optional

from agents.gmail_agent import build_gmail_agent
from tools.gmail_tool import gmail_tool


# Request/Response Models
class SendEmailRequest(BaseModel):
    """Request model for sending an email."""
    recipient: str
    subject: str
    body: str
    attachments: Optional[List[str]] = None


class SendEmailResponse(BaseModel):
    """Response model for sending an email."""
    status: str
    message: str


class ReadEmailRequest(BaseModel):
    """Request model for reading/searching emails."""
    query: str


class ReadEmailResponse(BaseModel):
    """Response model for reading emails."""
    results: str
    status: str


class AgentChatRequest(BaseModel):
    """Request model for agent chat interaction."""
    message: str


class AgentChatResponse(BaseModel):
    """Response model for agent chat interaction."""
    response: str


# Initialize FastAPI app
app = FastAPI(
    title="Gmail Agent Service",
    description="REST API for Gmail Agent using LangServe",
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
gmail_agent_chain = build_gmail_agent()

# Add LangServe routes for the agent chain
# This exposes standard LangServe endpoints like /invoke, /stream, etc.
add_routes(
    app,
    gmail_agent_chain,
    path="/agent",
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Gmail Agent Service",
        "status": "running",
        "endpoints": {
            "agent": "/agent/invoke - Full agent interaction via LangServe",
            "send_email": "/send-email - Direct email sending",
            "read_email": "/read-email - Direct email search",
            "chat": "/chat - Simple chat interface with agent",
        }
    }


@app.post("/send-email", response_model=SendEmailResponse)
async def send_email(request: SendEmailRequest):
    """
    Direct endpoint for sending emails.
    
    This endpoint bypasses the agent and directly calls the Gmail tool.
    Useful for programmatic email sending without LLM reasoning.
    """
    try:
        result = gmail_tool.invoke({
            "action": "send",
            "recipient": request.recipient,
            "subject": request.subject,
            "body": request.body,
            "attachments": request.attachments or [],
        })
        
        if result == "EMAIL_SENT":
            return SendEmailResponse(
                status="success",
                message="Email sent successfully"
            )
        else:
            return SendEmailResponse(
                status="error",
                message=f"Failed to send email: {result}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error sending email: {str(e)}"
        )


@app.post("/read-email", response_model=ReadEmailResponse)
async def read_email(request: ReadEmailRequest):
    """
    Direct endpoint for searching/reading emails.
    
    This endpoint bypasses the agent and directly calls the Gmail tool.
    Useful for programmatic email search without LLM reasoning.
    """
    try:
        result = gmail_tool.invoke({
            "action": "search",
            "query": request.query,
        })
        
        if result == "NO_RESULTS":
            return ReadEmailResponse(
                status="success",
                results="No emails found matching the query"
            )
        else:
            return ReadEmailResponse(
                status="success",
                results=result
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading emails: {str(e)}"
        )


@app.post("/chat", response_model=AgentChatResponse)
async def chat_with_agent(request: AgentChatRequest):
    """
    Simple chat endpoint that uses the full agent chain.
    
    This endpoint uses the agent's reasoning capabilities to handle
    natural language requests about Gmail operations.
    """
    try:
        # Invoke the agent with the user's message
        result = gmail_agent_chain.invoke({
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
