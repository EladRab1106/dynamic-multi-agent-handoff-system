from models.schemas import Plan, PlanStep
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from models.state import AgentState
from config.config import llm


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
                
                if ctx.get("file_path"):
                    idx += 1
            else:
                idx += 1

        ctx["current_step_index"] = idx

    if idx >= len(plan):
        return {"next": "FINISH", "context": ctx}

    next_worker = plan[idx]
    return {"next": next_worker, "context": ctx}