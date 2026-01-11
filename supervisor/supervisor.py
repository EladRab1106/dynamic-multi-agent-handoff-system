from typing import List

from models.schemas import Plan
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from models.state import AgentState
from config.config import llm
from agents.registry import CAPABILITY_INDEX


SUPERVISOR_PLANNING_PROMPT = """
You are the Supervisor/Planner in a multi-agent system.

Your job:
1. Analyze the conversation.
2. Determine the minimal ordered list of ACTIONS (capabilities) required.
3. Return ONLY JSON in the following format:

{{
  "steps": ["capability_1", "capability_2"]
}}

Rules:
- Use abstract capabilities (e.g. "research", "send_email", "create_document").
- Do NOT mention agent names.
- Use "direct_answer" if no tools are required.
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

        steps: List[str] = list(plan_obj.steps)
        if not steps:
            steps = ["direct_answer"]

        ctx["plan"] = steps
        ctx["current_step_index"] = 0

    plan: List[str] = ctx["plan"]
    idx = ctx.get("current_step_index", 0)

    # Check last message for completion status (agents now return final completion messages)
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'content'):
            content_lower = last_msg.content.lower()
            # Check if this message indicates completion of current step
            if plan[idx] == "send_email" and "email_sent" in content_lower:
                ctx["last_completed_capability"] = "send_email"
            elif plan[idx] == "search_email" and ("from:" in content_lower or "subject:" in content_lower):
                ctx["last_completed_capability"] = "search_email"
            elif plan[idx] == "research" and content_lower and "email_sent" not in content_lower:
                # Research is complete if we have content that's not an email status
                ctx["last_completed_capability"] = "research"
            elif plan[idx] == "create_document" and "report_created" in content_lower:
                ctx["last_completed_capability"] = "create_document"

    last_completed = ctx.get("last_completed_capability")

    if last_completed is not None and idx < len(plan):
        if last_completed == plan[idx]:
            idx += 1
            ctx["current_step_index"] = idx

    if idx >= len(plan):
        return {"next": "FINISH", "context": ctx}

    capability = plan[idx].strip().lower()
    agent_name = CAPABILITY_INDEX.get(capability)

    if not agent_name:
        raise ValueError(f"No agent registered for capability: {capability}")

    return {
        "next": agent_name,
        "context": ctx,
    }
