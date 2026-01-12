from typing import List

from models.schemas import Plan
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from models.state import AgentState
from config.config import llm
from agents.registry import CAPABILITY_INDEX
from agents.utils import extract_completed_capability


def supervisor_node(state: AgentState):
    ctx = dict(state.get("context", {}))

    # === STEP 1: Dynamic planning (only once) ===
    if "plan" not in ctx:
        available_capabilities = sorted(CAPABILITY_INDEX.keys())
        capabilities_text = "\n".join(f"- {c}" for c in available_capabilities)

        system_prompt = f"""
You are the Supervisor/Planner in a multi-agent system.

Available capabilities:
{capabilities_text}

Your job:
1. Analyze the user request.
2. Choose the minimal ordered list of capabilities needed.
3. Return ONLY JSON in the following format:

{{{{ "steps": ["capability_1", "capability_2"] }}}}

Rules:
- You MUST choose only from the list above.
- Do NOT invent new capabilities.
- Do NOT mention agent names.
- If no capability applies, return an empty list.
- If no tools are required, return ["direct_answer"].
"""

        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("messages"),
        ])

        plan_obj = (planning_prompt | llm.with_structured_output(Plan)).invoke(
            {"messages": state["messages"]}
        )

        steps: List[str] = list(plan_obj.steps or [])
        if not steps:
            steps = ["direct_answer"]

        ctx["plan"] = steps
        ctx["current_step_index"] = 0

    # === STEP 2: Progress tracking ===
    plan: List[str] = ctx["plan"]
    idx = ctx.get("current_step_index", 0)

    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            capability = extract_completed_capability(last_msg.content)
            if capability:
                ctx["last_completed_capability"] = capability

    if ctx.get("last_completed_capability") == plan[idx]:
        ctx["current_step_index"] = idx + 1
        idx += 1

    # === STEP 3: Finish ===
    if idx >= len(plan):
        return {"next": "FINISH", "context": ctx}

    # === STEP 4: Dispatch ===
    capability = plan[idx]
    agent_name = CAPABILITY_INDEX.get(capability)

    if not agent_name:
        return {
            "next": "FINISH",
            "context": {
                **ctx,
                "error": f"Unregistered capability: {capability}",
            },
        }

    return {
        "next": agent_name,
        "context": ctx,
    }
