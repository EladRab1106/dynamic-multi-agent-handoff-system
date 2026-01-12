import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agents.document_creator_agent import build_document_creator_agent


class AgentChatRequest(BaseModel):
    message: str


class AgentChatResponse(BaseModel):
    response: str
    file_path: str = None


app = FastAPI(
    title="Document Creator Agent Service",
    description="REST API for Document Creator Agent using LangServe",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_creator_agent_chain = build_document_creator_agent()


add_routes(
    app,
    document_creator_agent_chain,
    path="/agent",
)


@app.get("/")
async def root():
    return {
        "service": "Document Creator Agent Service",
        "status": "running",
        "endpoints": {
            "agent": "/agent/invoke - Full agent interaction via LangServe",
            "chat": "/chat - Simple chat interface with agent",
        }
    }


@app.post("/chat", response_model=AgentChatResponse)
async def chat_with_agent(request: AgentChatRequest):
    try:
        result = document_creator_agent_chain.invoke({
            "messages": [HumanMessage(content=request.message)]
        })
        
        if hasattr(result, 'content'):
            response_text = result.content
        elif isinstance(result, dict) and 'messages' in result:
            messages = result['messages']
            if messages and hasattr(messages[-1], 'content'):
                response_text = messages[-1].content
            else:
                response_text = str(result)
        else:
            response_text = str(result)
        
        os.makedirs("outputs", exist_ok=True)
        file_path = f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response_text)
        
        return AgentChatResponse(
            response=f"REPORT_CREATED: {file_path}",
            file_path=file_path
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )



