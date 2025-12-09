import os, operator
from typing import Annotated, Sequence, TypedDict, Literal, Dict, Any, List
from datetime import datetime

from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Gmail API
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from base64 import urlsafe_b64encode

# Pydantic
from pydantic import BaseModel

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class GmailInput(BaseModel):
    action: Literal["search", "send"]
    query: str = ""
    recipient: str = ""
    subject: str = ""
    body: str = ""



@tool(args_schema=GmailInput)
def gmail_tool(action, query: str = "", recipient: str = "", subject: str = "", body: str = ""):
    """Search Gmail or send an email."""
    creds = Credentials(
        token=None,
        refresh_token=os.getenv("GOOGLE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=[
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.modify",
        ],
    )

    service = build("gmail", "v1", credentials=creds)

    if action == "search":
        results = service.users().messages().list(
            userId="me", q=query, maxResults=5
        ).execute()

        messages = results.get("messages", [])

        if not messages:
            return "NO_RESULTS"

        out = []
        for m in messages:
            msg = service.users().messages().get(
                userId="me",
                id=m["id"],
                format="metadata",
                metadataHeaders=["From", "Subject"],
            ).execute()

            headers = msg.get("payload", {}).get("headers", [])
            frm = next((h["value"] for h in headers if h["name"] == "From"), "")
            subj = next((h["value"] for h in headers if h["name"] == "Subject"), "")
            snippet = msg.get("snippet", "")

            out.append(f"FROM: {frm}\nSUBJECT: {subj}\nSNIPPET: {snippet}")

        return "\n\n".join(out)

    if action == "send":
        message = MIMEText(body)
        message["to"] = recipient
        message["subject"] = subject
        message["from"] = os.getenv("GMAIL_SENDER_ADDRESS")

        raw = {"raw": urlsafe_b64encode(message.as_bytes()).decode()}
        service.users().messages().send(userId="me", body=raw).execute()
        return "EMAIL_SENT"

    return "ERROR"



def create_agent(llm, tools, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("messages"),
    ])
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    context: Dict[str, Any]



PlanStep = Literal["Researcher", "DocumentCreator", "Gmail", "DirectAnswer"]

class Plan(TypedDict):
    steps: List[PlanStep]


tavily = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))

# ---------- Researcher ----------
def researcher_node(state: AgentState):
    chain = create_agent(
        llm,
        [tavily],
        (
            "You are the Researcher Agent. "
            "Use the Tavily search tool whenever you need external information. "
            "When finished, summarize your findings in JSON with keys: "
            "topic, summary, key_points, sources."
        ),
    )
    result = chain.invoke({"messages": state["messages"]})

    ctx = dict(state.get("context", {}))
    ctx["last_task"] = "research"

    return {
        "messages": [result],
        "context": ctx,
    }

def document_creator_node(state: AgentState):
    os.makedirs("outputs", exist_ok=True)

    chain = create_agent(
        llm,
        [],
        "Convert the latest JSON research in the conversation into a clean Markdown report. "
        "Return ONLY Markdown with headings and bullet points."
    )

    md_msg = chain.invoke({"messages": state["messages"]})
    md_content = md_msg.content if hasattr(md_msg, "content") else str(md_msg)

    fp = f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(fp, "w") as f:
        f.write(md_content)

    ctx = dict(state.get("context", {}))
    ctx["file_path"] = fp
    ctx["last_task"] = "doc_created"

    return {
        "messages": [AIMessage(content=f"REPORT_CREATED: {fp}")],
        "context": ctx,
    }

def gmail_node(state: AgentState):
    ctx = dict(state.get("context", {}))

    chain = create_agent(
        llm,
        [gmail_tool],
        (
            "You are the Gmail Agent.\n"
            "Your job is to understand the user's intent regarding email actions.\n"
            "- Decide whether the user wants to SEARCH emails or SEND an email.\n"
            "- If SEARCH: infer an appropriate Gmail query.\n"
            "- If SEND: extract recipient, subject, and body.\n"
            "- Ask clarifying questions if any required field is missing.\n"
            "When you are ready, call the `gmail_tool` with the correct parameters.\n"
            "Think step-by-step, but respond concisely to the user."
        ),
    )

    result = chain.invoke({"messages": state["messages"]})

    ctx["last_task"] = "gmail"

    return {
        "messages": [result],
        "context": ctx,
    }

def direct_answer_node(state: AgentState):
    chain = create_agent(
        llm,
        [],
        "You are a helpful assistant. Answer the user directly."
    )
    result = chain.invoke({"messages": state["messages"]})

    ctx = dict(state.get("context", {}))
    ctx["last_task"] = "direct_answer"

    return {
        "messages": [result],
        "context": ctx,
    }

SUPERVISOR_PLANNING_PROMPT = """
You are the Supervisor/Planner in a multi-agent system.

Available workers:
- Researcher
- DocumentCreator
- Gmail
- DirectAnswer

Your job:
1. Analyze the conversation.
2. Determine the minimal required ordered list of workers.
3. Return ONLY JSON with a key 'steps' containing the ordered list.
"""

def supervisor_node(state: AgentState):
    ctx = dict(state.get("context", {}))

    if "plan" not in ctx:
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_PLANNING_PROMPT),
            MessagesPlaceholder("messages"),
        ])

        planning_chain = planning_prompt | llm.with_structured_output(Plan)
        plan_obj = planning_chain.invoke({"messages": state["messages"]})
        steps: List[PlanStep] = list(plan_obj.get("steps", []))
        if not steps:
            steps = ["DirectAnswer"]

        ctx["plan"] = steps
        ctx["current_step_index"] = 0

    plan: List[PlanStep] = ctx["plan"]
    idx = ctx.get("current_step_index", 0)
    last_task = ctx.get("last_task")

    task_to_worker = {
        "research": "Researcher",
        "doc_created": "DocumentCreator",
        "gmail": "Gmail",
        "direct_answer": "DirectAnswer",
    }

    if last_task is not None and idx < len(plan):
        worker_for_task = task_to_worker.get(last_task)
        if worker_for_task == plan[idx]:
            if worker_for_task == "DocumentCreator":
                # Advance only when a file was actually created
                if ctx.get("file_path"):
                    idx += 1
            else:
                idx += 1

        ctx["current_step_index"] = idx

    if idx >= len(plan):
        return {"next": "FINISH", "context": ctx}

    next_worker = plan[idx]
    return {"next": next_worker, "context": ctx}


graph = StateGraph(AgentState)

graph.add_node("Supervisor", supervisor_node)
graph.add_node("Researcher", researcher_node)
graph.add_node("ResearchTools", ToolNode([tavily]))
graph.add_node("DocumentCreator", document_creator_node)
graph.add_node("Gmail", gmail_node)
graph.add_node("GmailTools", ToolNode([gmail_tool]))
graph.add_node("DirectAnswer", direct_answer_node)

# Research flow: Researcher -> ResearchTools -> Supervisor
graph.add_edge("Researcher", "ResearchTools")
graph.add_edge("ResearchTools", "Supervisor")

# Gmail flow: Gmail -> GmailTools -> Supervisor
graph.add_edge("Gmail", "GmailTools")
graph.add_edge("GmailTools", "Supervisor")

# Other workers go directly back to Supervisor
graph.add_edge("DocumentCreator", "Supervisor")
graph.add_edge("DirectAnswer", "Supervisor")

graph.add_conditional_edges(
    "Supervisor",
    lambda state: state["next"],
    {
        "Researcher": "Researcher",
        "DocumentCreator": "DocumentCreator",
        "Gmail": "Gmail",
        "DirectAnswer": "DirectAnswer",
        "FINISH": END,
    }
)

graph.set_entry_point("Supervisor")
workflow = graph.compile()



if __name__ == "__main__":
    print("=== Multi-Agent System (Context-Aware Supervisor + Planning) ===")
    q = input("Enter your request: ")

    init_state: AgentState = {
        "messages": [HumanMessage(content=q)],
        "next": "Supervisor",
        "context": {},
    }

    for event in workflow.stream(init_state):
        if "__end__" in event:
            print("FINISHED.")
            break
        print(event)
        print("----")
