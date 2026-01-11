from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from agents.utils import build_completion_message


def create_agent(llm, tools, system_prompt: str):
    """
    Create an agent chain that executes tools internally.
    
    The chain:
    1. Prompts the LLM with system prompt and messages
    2. Binds tools to the LLM
    3. Executes tools if tool_calls are present
    4. Returns final AIMessage with completion status
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("messages"),
    ])
    
    if not tools:
        return prompt | llm
    
    # Create tool executor
    tool_map = {tool.name: tool for tool in tools}
    
    def execute_tools_and_respond(state):
        """Execute tools and return final message."""
        # Handle both dict format (from LangServe) and direct messages
        if isinstance(state, dict):
            messages = state.get("messages", [])
        else:
            messages = state if isinstance(state, list) else []
        
        if not messages:
            return AIMessage(content="No messages provided")
        
        # Get LLM response with tool calls
        llm_response = (prompt | llm.bind_tools(tools)).invoke({"messages": messages})
        
        # Check if message has tool calls
        if not hasattr(llm_response, 'tool_calls') or not llm_response.tool_calls:
            # No tool calls, return as-is
            return llm_response
        
        # Execute tools
        tool_messages = []
        for tool_call in llm_response.tool_calls:
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
            tool_args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
            tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", "")
            
            if tool_name in tool_map:
                try:
                    tool_result = tool_map[tool_name].invoke(tool_args)
                    tool_messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_id
                        )
                    )
                except Exception as e:
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_id
                        )
                    )
        
        # Create final completion message with completion contract
        if tool_messages:
            # Check tool results for completion indicators
            tool_results_text = " ".join([str(msg.content) for msg in tool_messages])
            tool_results_lower = tool_results_text.lower()
            
            # Emit completion contract based on tool results
            if "email_sent" in tool_results_lower:
                # Gmail send completed
                return build_completion_message("send_email")
            elif "from:" in tool_results_lower or "subject:" in tool_results_lower:
                # Gmail search completed
                return build_completion_message("search_email", {"results": tool_results_text})
            else:
                # For research and other tools, return tool results
                # The agent's system prompt will instruct it to return completion contract
                return AIMessage(content=tool_results_text)
        else:
            # No tool messages, return original
            return llm_response
    
    # Chain that processes messages and executes tools
    # LangServe expects chains that accept {"messages": [...]} and return a message
    def process_for_langserve(input_dict):
        """Process LangServe input format."""
        return execute_tools_and_respond(input_dict)
    
    return RunnableLambda(process_for_langserve)
