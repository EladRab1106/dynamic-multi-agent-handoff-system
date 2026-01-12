from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda


def create_agent(llm, tools, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("messages"),
    ])
    
    if not tools:
        return prompt | llm
    
    tool_map = {tool.name: tool for tool in tools}
    
    def execute_tools_and_respond(state):
        if isinstance(state, dict):
            messages = state.get("messages", [])
        else:
            messages = state if isinstance(state, list) else []
        
        if not messages:
            return AIMessage(content="No messages provided")
        
        llm_response = (prompt | llm.bind_tools(tools)).invoke({"messages": messages})
        
        if not hasattr(llm_response, 'tool_calls') or not llm_response.tool_calls:
            return llm_response
        
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
        
        messages_with_tools = list(messages) + [llm_response] + tool_messages
        final_response = (prompt | llm.bind_tools(tools)).invoke({"messages": messages_with_tools})
        
        return final_response
    
    def process_for_langserve(input_dict):
        return execute_tools_and_respond(input_dict)
    
    return RunnableLambda(process_for_langserve)
