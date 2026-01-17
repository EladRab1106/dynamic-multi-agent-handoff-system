"""
Document Creator Agent - Core agent chain logic.

This module is fully self-contained and uses only relative imports.
"""

from base_agent import create_agent
from tools import write_markdown_file
from config import llm

# Tool usage tracker (shared between agent and graph)
_tool_usage_tracker = {"value": False}


SYSTEM_PROMPT = """You are the Document Creator Agent in a strict multi-agent system.

Your task: Create a Markdown document from provided data and save it to disk using the write_markdown_file tool.

────────────────────
INPUT
────────────────────
You will receive data in the input message. This may include:
- Research data (Topic, Summary, Key Points, Sources)
- Email content
- Other structured information

Use ONLY the provided data. Do NOT invent content.

────────────────────
TOOL USAGE (MANDATORY - NO EXCEPTIONS)
────────────────────
You have access to the write_markdown_file tool.

CRITICAL: You CANNOT create files yourself. You MUST use the tool.

Process:
1. Format the provided data as a well-structured Markdown document
2. Call the write_markdown_file tool with the markdown content
3. The tool will write the file to disk and return the file_path
4. Use the EXACT file_path returned by the tool in your completion contract

You MUST call write_markdown_file to create the document.
The tool is responsible for writing the file - you cannot create files without it.

────────────────────
COMPLETION CONTRACT
────────────────────
After successfully calling write_markdown_file, return ONLY this completion contract:

{{
  "completed_capability": "create_document",
  "data": {{
    "file_path": "<exact file_path returned by tool>",
    "abs_file_path": "<exact abs_file_path returned by tool>"
  }}
}}

The file_path field must contain the EXACT relative path returned by the tool.
The abs_file_path field must contain the EXACT absolute path returned by the tool.

────────────────────
CRITICAL RULES
────────────────────
• You CANNOT create files yourself
• You MUST call the write_markdown_file tool to create the document
• You MUST use the file_path returned by the tool
• NEVER invent or guess file paths
• NEVER return create_document completion unless the tool succeeded
• The JSON must be the ENTIRE response - no text before or after
• Do NOT wrap JSON in markdown code blocks
• Do NOT include explanations or apologies
• Do NOT ask questions

FORBIDDEN:
❌ Any text outside the JSON
❌ Questions or explanations
❌ Inventing file paths
❌ Returning completion contract without calling the tool
❌ Free text responses
❌ Markdown formatting around JSON

YOUR RESPONSE MUST BE PURE JSON - NOTHING ELSE."""


def build_document_creator_agent():
    """
    Build the Document Creator agent chain.
    
    This function creates the agent chain with tools enabled.
    The agent must call write_markdown_file to create documents.
    """
    return create_agent(
        llm=llm,
        tools=[write_markdown_file],  # Agent must use this tool to create files
        system_prompt=SYSTEM_PROMPT,
        tool_usage_tracker=_tool_usage_tracker,
    )


def reset_tool_usage():
    """Reset tool usage flag. Called at start of each graph node invocation."""
    _tool_usage_tracker["value"] = False


def get_tool_usage() -> bool:
    """Get whether tool was used in current invocation."""
    return _tool_usage_tracker.get("value", False)
