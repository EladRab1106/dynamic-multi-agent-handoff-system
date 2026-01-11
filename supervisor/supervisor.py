from typing import List

from models.schemas import Plan
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from models.state import AgentState
from config.config import llm
from agents.registry import CAPABILITY_INDEX
from agents.utils import extract_completed_capability


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

    # STRICT JSON-BASED COMPLETION DETECTION
    # Only check agent outputs (AIMessage), never user messages (HumanMessage)
    # Only advance when explicit completion contract is detected
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        # Only process AIMessages (agent outputs), ignore HumanMessages (user input)
        # This ensures user messages can NEVER complete a step
        if isinstance(last_msg, AIMessage) and hasattr(last_msg, 'content'):
            content = last_msg.content
            # Extract completed capability from JSON contract
            capability = extract_completed_capability(content)
            if capability:
                # Set last_completed_capability from contract
                ctx["last_completed_capability"] = capability
            # If no contract found, do NOT set last_completed_capability
            # This prevents accidental step advancement
        # Explicitly ignore HumanMessage - user input never completes steps

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
