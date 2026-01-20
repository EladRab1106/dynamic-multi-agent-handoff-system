import os
from datetime import datetime
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agents.document_creator.document_creator_agent import build_document_creator_agent
from agents.document_creator.graph import build_document_creator_graph


logger = logging.getLogger(__name__)


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
document_creator_graph = build_document_creator_graph()

# Expose the chain endpoint (for backward compatibility)
add_routes(
    app,
    document_creator_agent_chain,
    path="/agent",
)

# Expose the graph endpoint (for RemoteGraph integration)
add_routes(
    app,
    document_creator_graph,
    path="/graph",
)


@app.get("/")
async def root():
    return {
        "service": "Document Creator Agent Service",
        "status": "running",
        "endpoints": {
            "agent": "/agent/invoke - Full agent interaction via LangServe (chain)",
            "graph": "/graph/invoke - LangGraph endpoint for RemoteGraph integration",
            "chat": "/chat - Simple chat interface with agent",
            "metadata": "/metadata - Agent metadata for capability discovery",
        }
    }


@app.get("/metadata")
async def metadata():
    """
    Agent metadata endpoint for capability discovery.
    
    Returns static declarative metadata describing the agent's capabilities.
    This endpoint is used by the Supervisor for dynamic capability discovery.
    """
    return {
        "agent_name": "DocumentCreator",
        "assistant_id": "graph",
        "capabilities": ["create_document"]
    }


@app.post("/chat", response_model=AgentChatResponse)
async def chat_with_agent(request: AgentChatRequest):
    logger.info("/chat endpoint called", extra={"message_preview": request.message[:200]})
    try:
        result = document_creator_agent_chain.invoke({
            "messages": [HumanMessage(content=request.message)]
        })
        logger.info(
            "Document creator agent chain invoked successfully from /chat",
            extra={"result_type": str(type(result))},
        )
        
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
        abs_file_path = os.path.abspath(file_path)
        logger.info(
            "Writing /chat response to report file",
            extra={"file_path": file_path, "abs_file_path": abs_file_path},
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response_text)
        
        return AgentChatResponse(
            response=f"REPORT_CREATED: {file_path}",
            file_path=file_path
        )
    except Exception as e:
        logger.exception("Error processing /chat request")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )



