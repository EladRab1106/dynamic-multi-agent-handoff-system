# NOTE:
# Validation, enforcement, and correctness checks are intentionally disabled.
# Agents are currently trusted as authoritative.
# A dedicated Validation Agent will be introduced later to verify outputs.

from typing import List
import logging
import json

from models.schemas import Plan
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from models.state import AgentState
from config.config import llm
import agents.registry as registry
from agents.utils import extract_completed_capability

logger = logging.getLogger(__name__)


def supervisor_node(state: AgentState):
    ctx = dict(state.get("context", {}))

    # === STEP 1: Dynamic planning (only once) ===
    if "plan" not in ctx:
        available_capabilities = sorted(registry.CAPABILITY_INDEX.keys())
        capabilities_text = "\n".join(f"- {c}" for c in available_capabilities)

        # Extract original user request (first human message)
        original_request = None
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
                original_request = msg.content if hasattr(msg, 'content') else str(msg)
                break
        
        if not original_request:
            original_request = "Process the request"

        system_prompt = f"""
You are the Supervisor/Planner in a strict multi-agent system.

Available capabilities (EXACT STRINGS – must be used verbatim):
{capabilities_text}

Your job:
1. Analyze the user request.
2. Choose the minimal ordered list of capabilities needed.
3. Return ONLY JSON in the following format:

{{{{ "steps": ["capability_1", "capability_2"] }}}}

CRITICAL RULES:
- You MUST use ONLY the exact capability strings listed above.
- Do NOT paraphrase, rename, or invent capabilities.
- Do NOT use synonyms (e.g. "send_email", "generate_file").
- If you need to create a document, use: create_document
- If you need to send an email, use: gmail
- If no capability applies, return an empty list.
- If no tools are required, return ["direct_answer"].

Any deviation is considered a system failure.
"""

        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("messages"),
        ])

        # Use only the original request for planning
        plan_obj = (planning_prompt | llm.with_structured_output(Plan)).invoke(
            {"messages": [HumanMessage(content=original_request)]}
        )

        steps: List[str] = list(plan_obj.steps or [])
        if not steps:
            steps = ["direct_answer"]

        ctx["plan"] = steps
        ctx["current_step_index"] = 0
        ctx["completed_capabilities"] = []  # Track completed capabilities to prevent duplicates
        ctx["agent_retry_count"] = {}  # Track retry count per agent to prevent infinite loops
        
        logger.info(f"Supervisor: Created plan with {len(steps)} steps: {steps}")

    # === STEP 2: Progress tracking - trust agent output completely ===
    plan: List[str] = ctx["plan"]
    idx = ctx.get("current_step_index", 0)
    completed_capabilities = ctx.get("completed_capabilities", [])
    agent_retry_count = ctx.get("agent_retry_count", {})

    # Check the last agent message for completion contract
    # RemoteGraph may return messages as dicts or AIMessage objects
    messages = state.get("messages", [])
    capability = None
    
    # Find the last message that looks like an agent response
    # Check both AIMessage objects and dict representations
    last_agent_message = None
    last_agent_content = None
    
    # Log message types for debugging
    message_types = []
    for msg in messages[-5:]:  # Check last 5 messages
        if isinstance(msg, dict):
            msg_type = msg.get('type') or msg.get('message_type') or 'dict'
            message_types.append(msg_type)
        elif hasattr(msg, 'type'):
            message_types.append(msg.type)
        else:
            message_types.append(type(msg).__name__)
    logger.debug(f"Supervisor: Last 5 message types: {message_types}")
    
    # Look for the last message that's an AI response (not human, not system)
    for msg in reversed(messages):
        # Handle AIMessage objects
        if isinstance(msg, AIMessage):
            last_agent_message = msg
            last_agent_content = msg.content if hasattr(msg, 'content') else str(msg)
            break
        # Handle dict representations (from RemoteGraph)
        elif isinstance(msg, dict):
            msg_type = msg.get('type') or msg.get('message_type')
            # Check if it's an AI message
            if msg_type in ['ai', 'AIMessage', 'assistant']:
                last_agent_message = msg
                last_agent_content = msg.get('content') or msg.get('text') or str(msg)
                break
        # Handle messages with type attribute
        elif hasattr(msg, 'type'):
            if msg.type in ['ai', 'AIMessage', 'assistant']:
                last_agent_message = msg
                last_agent_content = getattr(msg, 'content', None) or str(msg)
                break
    
    if last_agent_content:
        capability = extract_completed_capability(last_agent_content)
        if capability:
            logger.info(f"Supervisor: Detected completion contract: capability={capability}")
    else:
        logger.debug(f"Supervisor: No agent message found. Total messages: {len(messages)}")
    
    # If agent emitted a completion contract, trust it and advance
    if capability:
        if capability not in completed_capabilities:
            completed_capabilities.append(capability)
            ctx["completed_capabilities"] = completed_capabilities
            ctx["current_step_index"] = idx + 1
            # Reset retry count for this capability
            if capability in agent_retry_count:
                del agent_retry_count[capability]
            ctx["agent_retry_count"] = agent_retry_count
            
            idx += 1
            logger.info(
                f"Supervisor: Agent completed '{capability}'. Advancing to step {idx}/{len(plan)}"
            )
        else:
            logger.info(
                f"Supervisor: Capability '{capability}' already completed. Skipping to next step."
            )
            ctx["current_step_index"] = idx + 1
            idx += 1
    # If agent returned a message but no completion contract, check retry count
    elif last_agent_message and idx < len(plan):
        # Agent returned something but no completion contract
        # After retries, treat as completion to prevent infinite loops
        current_capability = plan[idx] if idx < len(plan) else None
        retry_count = agent_retry_count.get(current_capability, 0)
        if retry_count >= 2:  # After 2 retries, assume agent is done
            logger.warning(
                f"Supervisor: Agent returned response without completion contract after {retry_count} attempts. "
                f"Treating as completion to prevent infinite loop."
            )
            # Advance anyway - agent is authoritative
            if current_capability and current_capability not in completed_capabilities:
                completed_capabilities.append(current_capability)
                ctx["completed_capabilities"] = completed_capabilities
                ctx["current_step_index"] = idx + 1
                if current_capability in agent_retry_count:
                    del agent_retry_count[current_capability]
                ctx["agent_retry_count"] = agent_retry_count
                idx += 1
                logger.info(f"Supervisor: Advanced despite missing completion contract. Step {idx}/{len(plan)}")

    # === STEP 3: Finish ===
    if idx >= len(plan):
        logger.info(f"Supervisor: All steps completed. Finishing.")
        return {"next": "FINISH", "context": ctx}

    # === STEP 4: Dispatch with RUNTIME CAPABILITY ASSERTION ===
    capability = plan[idx]
    
    # PREVENT DUPLICATE EXECUTION
    if capability in completed_capabilities:
        logger.info(
            f"Supervisor: Capability '{capability}' already completed. Skipping to next step."
        )
        ctx["current_step_index"] = idx + 1
        idx += 1
        if idx >= len(plan):
            return {"next": "FINISH", "context": ctx}
        capability = plan[idx]
    
    # PREVENT INFINITE LOOPS: Check retry count
    retry_count = agent_retry_count.get(capability, 0)
    MAX_RETRIES = 3
    
    if retry_count >= MAX_RETRIES:
        logger.error(
            f"Supervisor: Capability '{capability}' failed to complete after {MAX_RETRIES} attempts. "
            f"Advancing to next step to prevent infinite loop."
        )
        # Mark as completed to prevent further retries
        completed_capabilities.append(capability)
        ctx["completed_capabilities"] = completed_capabilities
        ctx["current_step_index"] = idx + 1
        idx += 1
        if idx >= len(plan):
            return {"next": "FINISH", "context": ctx}
        capability = plan[idx]
        retry_count = 0  # Reset for next capability
    
    # Increment retry count
    agent_retry_count[capability] = retry_count + 1
    ctx["agent_retry_count"] = agent_retry_count
    
    # HARD ASSERTION: Every planned capability MUST exist in CAPABILITY_INDEX
    if capability not in registry.CAPABILITY_INDEX:
        raise RuntimeError(
            f"Capability '{capability}' was requested in plan but no agent provides it. "
            f"Available capabilities: {sorted(registry.CAPABILITY_INDEX.keys())}. "
            f"This is a system configuration error - the capability must be discovered "
            f"before the Supervisor can use it."
        )
    
    agent_name = registry.CAPABILITY_INDEX[capability]
    
    logger.info(
        f"Supervisor: Dispatching step {idx+1}/{len(plan)}: capability={capability} → agent={agent_name} "
        f"(attempt {retry_count + 1}/{MAX_RETRIES})"
    )

    return {
        "next": agent_name,
        "context": ctx,
    }
