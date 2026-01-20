"""
Capability-agnostic agent runtime with robust tool execution.

Local copy to ensure the agent is fully self-contained.

This variant adds lightweight logging so Direct Answer behaviour can be
observed in logs without being too noisy.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda
import json
import logging


logger = logging.getLogger(__name__)


def create_agent(llm, tools, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("messages"),
    ])
    
    if not tools:
        def run_without_tools(input_dict: dict):
            messages = input_dict.get("messages", [])
            if not messages:
                return AIMessage(content="No messages provided")
            return (prompt | llm).invoke({"messages": messages})
        return RunnableLambda(run_without_tools)
    
    tool_map = {tool.name: tool for tool in tools}
    
    def execute_tools_and_respond(state):
        """Execute tools in a loop until no more tool_calls remain."""
        if isinstance(state, dict):
            messages = state.get("messages", [])
        else:
            messages = state if isinstance(state, list) else []
        
        if not messages:
            logger.warning("Direct Answer agent called without any messages; returning fallback AIMessage")
            return AIMessage(content="No messages provided")
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info("Direct Answer agent loop iteration %s starting", iteration)
            
            # Get LLM response with tools bound
            try:
                llm_response = (prompt | llm.bind_tools(tools)).invoke({"messages": messages})
            except Exception as e:
                logger.exception("Direct Answer agent: error getting LLM response during tool loop")
                return AIMessage(content=f"Error getting LLM response: {str(e)}")
            
            # Check if response has tool_calls
            has_tool_calls = (
                hasattr(llm_response, "tool_calls")
                and llm_response.tool_calls
                and len(llm_response.tool_calls) > 0
            )
            logger.info(
                "Direct Answer agent: LLM response received",
                extra={
                    "has_tool_calls": bool(has_tool_calls),
                    "num_tool_calls": len(getattr(llm_response, "tool_calls", []) or []),
                },
            )
            
            # If no tool calls, we're done
            if not has_tool_calls:
                logger.info("Direct Answer agent: no tool calls in LLM response; returning final AIMessage")
                return llm_response
            
            # Execute ALL tool calls - CRITICAL: Every tool_call_id MUST get a ToolMessage
            tool_messages = []
            tool_call_ids_seen = set()
            
            for tool_call in llm_response.tool_calls:
                # Handle both dict and object formats
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")
                else:
                    tool_name = getattr(tool_call, "name", "")
                    tool_args = getattr(tool_call, "args", {})
                    tool_id = getattr(tool_call, "id", "")
                
                if not tool_id:
                    tool_id = f"call_{iteration}_{len(tool_messages)}"
                
                # Ensure we respond to each tool_call_id exactly once
                if tool_id in tool_call_ids_seen:
                    logger.warning(
                        "Direct Answer agent: duplicate tool_call_id '%s' encountered; skipping",
                        tool_id,
                    )
                    continue
                tool_call_ids_seen.add(tool_id)
                
                logger.info(
                    "Direct Answer agent: executing tool call",
                    extra={"tool_name": tool_name, "tool_call_id": tool_id},
                )
                
                if tool_name not in tool_map:
                    logger.error(
                        "Direct Answer agent: tool '%s' not found for tool_call_id '%s'",
                        tool_name,
                        tool_id,
                    )
                    tool_messages.append(
                        ToolMessage(
                            tool_call_id=tool_id,
                            content=f"Tool '{tool_name}' not found",
                        )
                    )
                    continue
                
                try:
                    tool_result = tool_map[tool_name].invoke(tool_args)
                    # Convert dict results to JSON string for ToolMessage
                    if isinstance(tool_result, dict):
                        content = json.dumps(tool_result)
                    else:
                        content = str(tool_result)
                    
                    logger.info(
                        "Direct Answer agent: tool call succeeded",
                        extra={"tool_name": tool_name, "tool_call_id": tool_id},
                    )
                    tool_messages.append(
                        ToolMessage(
                            tool_call_id=tool_id,
                            content=content,
                        )
                    )
                except Exception as e:
                    logger.exception(
                        "Direct Answer agent: error while executing tool '%s' for tool_call_id '%s'",
                        tool_name,
                        tool_id,
                    )
                    tool_messages.append(
                        ToolMessage(
                            tool_call_id=tool_id,
                            content=f"Tool error: {e}",
                        )
                    )
            
            # CRITICAL: Add assistant message with tool_calls, then ALL ToolMessages
            # This ensures OpenAI's requirement: tool_calls must be followed by ToolMessages
            messages = messages + [llm_response] + tool_messages
        
        logger.error(
            "Direct Answer agent: maximum iterations (%s) reached; agent may not have completed",
            max_iterations,
        )
        # If we hit max iterations, return the last response
        return AIMessage(content="Maximum iterations reached. Agent may not have completed.")
    
    def process_for_langserve(input_dict):
        return execute_tools_and_respond(input_dict)
    
    return RunnableLambda(process_for_langserve)
