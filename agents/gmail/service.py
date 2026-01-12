from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import List, Optional

from agents.gmail_agent import build_gmail_agent
from tools.gmail_tool import gmail_tool


class SendEmailRequest(BaseModel):
    recipient: str
    subject: str
    body: str
    attachments: Optional[List[str]] = None


class SendEmailResponse(BaseModel):
    status: str
    message: str


class ReadEmailRequest(BaseModel):
    query: str


class ReadEmailResponse(BaseModel):
    results: str
    status: str


class AgentChatRequest(BaseModel):
    message: str


class AgentChatResponse(BaseModel):
    response: str


app = FastAPI(
    title="Gmail Agent Service",
    description="REST API for Gmail Agent using LangServe",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gmail_agent_chain = build_gmail_agent()

add_routes(
    app,
    gmail_agent_chain,
    path="/agent",
)


@app.get("/")
async def root():
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
    try:
        result = gmail_agent_chain.invoke({
            "messages": [HumanMessage(content=request.message)]
        })
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

