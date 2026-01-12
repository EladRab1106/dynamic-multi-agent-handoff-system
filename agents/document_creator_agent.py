"""
Generic Document Creator Agent.

Reads from ctx["document_source"] to create documents from any source type.
Completely decoupled from specific agents (Gmail, Researcher, etc.).
"""

import os
import json
from datetime import datetime
from typing import Dict, Any

from langchain_core.messages import AIMessage
from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec

from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are a generic document creation agent.\n"
    "You create documents from structured data provided to you.\n\n"
    "RULES:\n"
    "- You MUST create documents ONLY from the data provided\n"
    "- You are STRICTLY FORBIDDEN from inventing or assuming content\n"
    "- If no data is provided, you MUST explicitly state this\n"
    "- Use the data to create well-structured Markdown documents\n"
    "- When done, return ONLY this JSON format:\n"
    '{{"completed_capability": "create_document", "data": {{"markdown": "...", "file_path": "outputs/report_YYYYMMDD_HHMMSS.md"}}}}\n\n'
    "CRITICAL:\n"
    "- Return ONLY valid JSON - no markdown, no code blocks, no extra text\n"
    "- Include the full markdown content in the 'markdown' field\n"
    "- The file_path must follow the pattern: outputs/report_YYYYMMDD_HHMMSS.md"
)


def build_document_creator_agent():
    return create_agent(
        llm=llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )


def format_content_to_markdown(source_type: str, content: Any) -> str:
    """
    Convert document source content to Markdown format.
    
    Handles different content types:
    - dict: structured data (email, research, etc.)
    - str: plain text
    - list: bullet points
    """
    if content is None:
        return "# Document\n\nNo content was provided."
    
    if isinstance(content, str):
        return f"# Document\n\n{content}"
    
    if isinstance(content, list):
        markdown = "# Document\n\n"
        for item in content:
            if isinstance(item, dict):
                markdown += f"- {json.dumps(item, indent=2)}\n"
            else:
                markdown += f"- {item}\n"
        return markdown
    
    if isinstance(content, dict):
        # Handle different source types
        if source_type == "email":
            return f"""# Email Report

**From:** {content.get("from", "Unknown")}
**Subject:** {content.get("subject", "No Subject")}
**Date:** {content.get("date", "Unknown")}

---

{content.get("body", "No body content")}
"""
        elif source_type == "research":
            # Research data structure: {topic, summary, key_points, sources}
            markdown = f"""# Research Report

## Topic: {content.get("topic", "Unknown")}

### Summary

{content.get("summary", "No summary provided")}

### Key Points

"""
            key_points = content.get("key_points", [])
            if isinstance(key_points, list):
                for point in key_points:
                    markdown += f"- {point}\n"
            else:
                markdown += f"- {key_points}\n"
            
            sources = content.get("sources", [])
            if sources:
                markdown += "\n### Sources\n\n"
                if isinstance(sources, list):
                    for source in sources:
                        markdown += f"- {source}\n"
                else:
                    markdown += f"- {sources}\n"
            
            return markdown
        else:
            # Generic dict - convert to structured Markdown
            markdown = f"# Document\n\n"
            for key, value in content.items():
                markdown += f"## {key.title()}\n\n"
                if isinstance(value, (dict, list)):
                    markdown += f"```json\n{json.dumps(value, indent=2)}\n```\n\n"
                else:
                    markdown += f"{value}\n\n"
            return markdown
    
    # Fallback: convert to string
    return f"# Document\n\n{str(content)}"


def document_creator_node(state: AgentState):
    """
    Generic document creator that reads from ctx["document_source"].
    
    Supports any source type: email, research, llm, mixed, etc.
    """
    ctx = dict(state.get("context", {}))
    
    # Read from generic document_source contract
    document_source = ctx.get("document_source")
    
    if not document_source:
        # No document source provided
        markdown = "# Document\n\nNo document source was provided."
        file_path = f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        os.makedirs("outputs", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        
        ctx["file_path"] = file_path
        ctx["last_completed_capability"] = "create_document"
        
        content = json.dumps({
            "completed_capability": "create_document",
            "data": {
                "markdown": markdown,
                "file_path": file_path
            }
        })
        
        return {
            "messages": [AIMessage(content=content)],
            "context": ctx
        }
    
    # Extract source type and content
    source_type = document_source.get("type", "unknown")
    content = document_source.get("content")
    
    # Convert content to Markdown based on type
    markdown = format_content_to_markdown(source_type, content)
    
    # Save to file
    os.makedirs("outputs", exist_ok=True)
    file_path = f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    ctx["file_path"] = file_path
    ctx["last_completed_capability"] = "create_document"
    
    content_json = json.dumps({
        "completed_capability": "create_document",
        "data": {
            "markdown": markdown,
            "file_path": file_path
        }
    })
    
    return {
        "messages": [AIMessage(content=content_json)],
        "context": ctx
    }


register_agent(
    AgentSpec(
        name="DocumentCreator",
        capabilities=["create_document"],
        build_chain=build_document_creator_agent,
    )
)
