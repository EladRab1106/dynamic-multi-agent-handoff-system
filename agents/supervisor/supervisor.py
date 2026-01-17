# NOTE:
# Validation, enforcement, and correctness checks are intentionally disabled.
# Agents are currently trusted as authoritative.
# A dedicated Validation Agent will be introduced later to verify outputs.

"""
Supervisor Agent - Core planning and routing logic.

This module is fully self-contained and uses only relative imports.
"""

from typing import List, Dict, Any
import logging
import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Local relative imports (self-contained)
from state import AgentState
from config import llm
from schemas import Plan
from utils import extract_completed_capability

logger = logging.getLogger(__name__)


def supervisor_node(state: AgentState):
    ctx = dict(state.get("context", {}))
    
    # Preserve capabilities in context for subsequent invocations
    # Capabilities are injected by orchestrator and must persist across Supervisor calls
    if "capabilities" not in ctx:
        # If capabilities missing, try to get from state (shouldn't happen, but defensive)
        ctx["capabilities"] = state.get("context", {}).get("capabilities", [])

    # === STEP 1: Dynamic planning (only once) ===
    if "plan" not in ctx:
        # Read capabilities from context (provided by orchestrator)
        # Supervisor receives ONLY capability strings, not agent names or URLs
        available_capabilities = ctx.get("capabilities", [])
        
        if not available_capabilities:
            logger.warning("Supervisor: No capabilities provided in context. Treating as direct answer mode.")
            # No capabilities available - answer directly
            messages = state.get("messages", [])
            original_request = None
            for msg in messages:
                if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
                    original_request = msg.content if hasattr(msg, 'content') else str(msg)
                    break
            
            if not original_request:
                original_request = "Process the request"
            
            # Answer directly using Supervisor's LLM
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the user's question directly and clearly."),
                MessagesPlaceholder("messages"),
            ])
            
            response = (answer_prompt | llm).invoke({"messages": messages})
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            logger.info("Supervisor: No capabilities available - answering directly")
            return {
                "next": "FINISH",
                "context": {**ctx, "supervisor_mode": "direct"},
                "messages": [AIMessage(content=response_content)]
            }
        
        capabilities_text = "\n".join(f"- {c}" for c in sorted(available_capabilities))

        # Extract original user request (first human message)
        original_request = None
        for msg in state.get("messages", []):
            # Handle both HumanMessage objects and dict representations
            if isinstance(msg, HumanMessage):
                original_request = msg.content if hasattr(msg, 'content') else str(msg)
                break
            elif isinstance(msg, dict):
                if msg.get("type") == "human":
                    original_request = msg.get("content", str(msg))
                    break
            elif hasattr(msg, 'type') and msg.type == "human":
                original_request = msg.content if hasattr(msg, 'content') else str(msg)
                break
        
        if not original_request:
            original_request = "Process the request"
        
        logger.debug(f"Supervisor: Extracted original request: {original_request[:200]}...")

        # Build dynamic examples based on available capabilities
        # Examples should illustrate multi-step and single-step scenarios
        example_capabilities = sorted(available_capabilities)
        examples_text = ""
        
        if len(example_capabilities) >= 3:
            # Multi-step example with 3+ capabilities - format as JSON array
            cap1, cap2, cap3 = example_capabilities[0], example_capabilities[1], example_capabilities[2]
            examples_text = f'''User: "perform task requiring {cap1}, then {cap2}, then {cap3}"
Response: {{{{ "steps": ["{cap1}", "{cap2}", "{cap3}"] }}}}

User: "perform task requiring {cap1}"
Response: {{{{ "steps": ["{cap1}"] }}}}

User: "what is 2+2?"
Response: {{{{ "steps": [] }}}}'''
        elif len(example_capabilities) >= 2:
            cap1, cap2 = example_capabilities[0], example_capabilities[1]
            examples_text = f'''User: "perform task requiring {cap1} then {cap2}"
Response: {{{{ "steps": ["{cap1}", "{cap2}"] }}}}

User: "perform task requiring {cap1}"
Response: {{{{ "steps": ["{cap1}"] }}}}

User: "what is 2+2?"
Response: {{{{ "steps": [] }}}}'''
        elif len(example_capabilities) >= 1:
            cap1 = example_capabilities[0]
            examples_text = f'''User: "perform task requiring {cap1}"
Response: {{{{ "steps": ["{cap1}"] }}}}

User: "what is 2+2?"
Response: {{{{ "steps": [] }}}}'''
        else:
            examples_text = '''User: "what is 2+2?"
Response: {{{{ "steps": [] }}}}'''

        system_prompt = f"""
You are the Supervisor / Planner in a STRICT multi-agent system.

Available capabilities (EXACT STRINGS — must be used verbatim):
{capabilities_text}

Your task:
1. Analyze the user's request.
2. Decide which of the AVAILABLE capabilities listed above are REQUIRED.
3. Return a minimal, ordered list of capabilities.

Return ONLY valid JSON in this format:
{{{{ "steps": ["capability_1", "capability_2"] }}}}

STRICT RULES (VIOLATION = SYSTEM FAILURE):
- Use ONLY the listed capabilities.
- Use capability strings EXACTLY as shown.
- Do NOT invent, rename, or explain.
- Do NOT return text outside JSON.

You may return an EMPTY list ONLY if:
- The request is purely conversational
- Requires no external action
- And none of the capabilities apply

EXAMPLES:

{examples_text}

User request:
"{original_request}"

Analyze carefully and return the JSON plan now.
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
        
        # Log the plan for debugging
        logger.info(f"Supervisor: LLM returned plan with {len(steps)} steps: {steps}")
        logger.debug(f"Supervisor: Original request was: {original_request}")
        
        # If plan is empty, answer directly (LLM has decided no capabilities are needed)
        if not steps:
            messages = state.get("messages", [])
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the user's question directly and clearly."),
                MessagesPlaceholder("messages"),
            ])
            
            response = (answer_prompt | llm).invoke({"messages": messages})
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            logger.info("Supervisor: Plan is empty - answering directly")
            return {
                "next": "FINISH",
                "context": {**ctx, "supervisor_mode": "direct"},
                "messages": [AIMessage(content=response_content)]
            }

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
    
    # Validate capability exists in available capabilities
    available_capabilities = ctx.get("capabilities", [])
    if capability not in available_capabilities:
        available = sorted(available_capabilities) if available_capabilities else []
        raise RuntimeError(
            f"Capability '{capability}' was requested in plan but is not available. "
            f"Available capabilities: {available}. "
            f"This is a system configuration error - the capability must be discovered "
            f"and provided in context before the Supervisor can use it."
        )
    
    logger.info(
        f"Supervisor: Dispatching step {idx+1}/{len(plan)}: capability={capability} "
        f"(attempt {retry_count + 1}/{MAX_RETRIES})"
    )

    # Return capability string in "next" - orchestrator maps to agent_name
    return {
        "next": capability,  # Return capability string, not agent name
        "context": ctx,
    }
