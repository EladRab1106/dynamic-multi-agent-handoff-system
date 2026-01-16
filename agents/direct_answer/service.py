from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agents.direct_answer.direct_answer_agent import build_direct_answer_agent
from agents.direct_answer.graph import build_direct_answer_graph


class AgentChatRequest(BaseModel):
    message: str


class AgentChatResponse(BaseModel):
    response: str



app = FastAPI(
    title="Direct Answer Agent Service",
    description="REST API for Direct Answer Agent using LangServe",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

direct_answer_agent_chain = build_direct_answer_agent()
direct_answer_graph = build_direct_answer_graph()

# Expose the chain endpoint (for backward compatibility)
add_routes(
    app,
    direct_answer_agent_chain,
    path="/agent",
)

# Expose the graph endpoint (for RemoteGraph integration)
add_routes(
    app,
    direct_answer_graph,
    path="/graph",
)


@app.get("/")
async def root():
    return {
        "service": "Direct Answer Agent Service",
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
        "agent_name": "DirectAnswer",
        "assistant_id": "graph",
        "capabilities": ["direct_answer"]
    }


@app.post("/chat", response_model=AgentChatResponse)
async def chat_with_agent(request: AgentChatRequest):

    try:
        result = direct_answer_agent_chain.invoke({
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
        
        return AgentChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )



