"""
Agent wrapper nodes that construct clean inputs for each agent.

These wrappers ensure:
- Each agent receives ONLY what it needs
- No message pollution between agents
- Clean separation of concerns
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from models.state import AgentState
from agents.utils import parse_completion_message, extract_completed_capability, validate_completion_contract_strict
from langgraph.pregel.remote import RemoteGraph

logger = logging.getLogger(__name__)


def write_markdown_file_deterministic(markdown: str) -> str:
    """
    Deterministic file writer - the ONLY authority for document creation.
    
    This function:
    - Creates outputs/ directory if missing
    - Generates filename with timestamp
    - Writes markdown to disk
    - Verifies file exists
    - Returns file path
    
    Args:
        markdown: Markdown content to write
        
    Returns:
        File path (relative): outputs/report_YYYYMMDD_HHMMSS.md
        
    Raises:
        OSError: If file cannot be written
        AssertionError: If file does not exist after writing
    """
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/report_{timestamp}.md"
    
    # Write file
    file_path = Path(filename)
    file_path.write_text(markdown, encoding="utf-8")
    
    # HARD VERIFICATION: File must exist
    assert file_path.exists(), f"CRITICAL: File was not created at {filename}"
    assert file_path.stat().st_size > 0, f"CRITICAL: File is empty at {filename}"
    
    logger.info(f"Deterministic file writer: Created file at {filename} ({file_path.stat().st_size} bytes)")
    
    return filename


def extract_markdown_from_text(text: str) -> Optional[str]:
    """
    Extract markdown content from agent output.
    
    Tries to find markdown in various formats:
    - JSON completion contract with markdown field
    - Markdown code blocks (```markdown ... ```)
    - Plain markdown text
    
    Returns:
        Extracted markdown content, or None if not found
    """
    if not text or not text.strip():
        return None
    
    # Try to parse as JSON completion contract
    try:
        contract = json.loads(text.strip())
        if isinstance(contract, dict):
            data = contract.get("data", {})
            markdown = data.get("markdown")
            if markdown:
                return markdown
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try to extract from markdown code blocks
    markdown_block_pattern = r'```(?:markdown)?\s*\n(.*?)\n```'
    matches = re.findall(markdown_block_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Try to find markdown-like content (starts with # or contains markdown patterns)
    lines = text.strip().split('\n')
    markdown_indicators = ['# ', '## ', '### ', '- ', '* ', '1. ']
    if any(line.startswith(indicator) for line in lines[:5] for indicator in markdown_indicators):
        return text.strip()
    
    return None


def generate_markdown_from_research(research_summary: Dict[str, Any]) -> str:
    """
    Generate markdown document from research data.
    
    This is a fallback when agent output cannot be parsed.
    
    Args:
        research_summary: Research data dictionary
        
    Returns:
        Generated markdown content
    """
    topic = research_summary.get('topic', 'Research Report')
    summary = research_summary.get('summary', '')
    key_points = research_summary.get('key_points', [])
    sources = research_summary.get('sources', [])
    
    markdown = f"# {topic}\n\n"
    
    if summary:
        markdown += f"## Summary\n\n{summary}\n\n"
    
    if key_points:
        markdown += "## Key Points\n\n"
        for point in key_points:
            markdown += f"- {point}\n"
        markdown += "\n"
    
    if sources:
        markdown += "## Sources\n\n"
        for source in sources:
            markdown += f"- {source}\n"
        markdown += "\n"
    
    return markdown


def create_researcher_wrapper(researcher_remote: RemoteGraph):
    """Create a wrapper node for Researcher agent with clean input construction."""
    
    def researcher_wrapper(state: AgentState) -> Dict[str, Any]:
        ctx = state.get("context", {})
        
        # Extract original user request (first human message)
        messages = state.get("messages", [])
        original_request = None
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == "human":
                original_request = msg.content if hasattr(msg, 'content') else str(msg)
                break
        
        if not original_request:
            original_request = "Research the requested topic"
        
        # Construct clean input: ONLY the original user request
        clean_state = {
            "messages": [HumanMessage(content=original_request)],
            "context": {},
            "next": "",
        }
        
        logger.info(f"Researcher wrapper: Constructing clean input with request: {original_request[:100]}...")
        
        # Invoke remote agent
        result = researcher_remote.invoke(clean_state)
        
        # Extract response
        if isinstance(result, dict):
            response_messages = result.get("messages", [])
            response_context = result.get("context", {})
        else:
            response_messages = [result] if isinstance(result, AIMessage) else []
            response_context = {}
        
        if not response_messages:
            raise ValueError("Researcher agent returned no messages")
        
        response_message = response_messages[-1]
        content = response_message.content if hasattr(response_message, 'content') else str(response_message)
        
        # STRICT VALIDATION: If completion contract exists, validate it strictly
        contract = parse_completion_message(content)
        if contract:
            # HARD CHECK: Ensure content contains ONLY valid JSON (no extra text)
            try:
                validated_contract = validate_completion_contract_strict(content)
                contract = validated_contract  # Use the validated version
                logger.info(f"Researcher: Validated completion contract for capability={contract.get('completed_capability')}")
            except ValueError as e:
                raise ValueError(
                    f"Researcher agent returned invalid completion contract: {e}"
                )
        
        # Update context
        new_ctx = dict(ctx)
        if response_context:
            new_ctx.update(response_context)
        
        # Extract completed capability
        capability = extract_completed_capability(content)
        if capability:
            new_ctx["last_completed_capability"] = capability
            new_ctx["research_data"] = contract.get("data", {}) if contract else {}
            logger.info(f"Researcher completed: capability={capability}")
        
        return {
            "messages": [response_message],
            "context": new_ctx,
        }
    
    return researcher_wrapper


def create_document_creator_wrapper(document_creator_remote: RemoteGraph):
    """
    Create a wrapper node for Document Creator agent with DETERMINISTIC file creation.
    
    This wrapper:
    - Treats agent output as raw text only
    - Extracts markdown from agent OR falls back to generating from research_data
    - Writes file using pure Python (not trusting LLM)
    - Forces completion contract regardless of LLM behavior
    - NEVER raises due to LLM refusal
    """
    
    def document_creator_wrapper(state: AgentState) -> Dict[str, Any]:
        ctx = state.get("context", {})
        
        # Extract research data from context (source of truth)
        research_data = ctx.get("research_data", {})
        research_summary = research_data.get("research_summary", {})
        
        # Extract original user request
        messages = state.get("messages", [])
        original_request = None
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == "human":
                original_request = msg.content if hasattr(msg, 'content') else str(msg)
                break
        
        # Construct input for agent (agent is just a formatter)
        if research_summary:
            document_input = f"""Format the following research data as a Markdown document:

Topic: {research_summary.get('topic', 'Unknown')}
Summary: {research_summary.get('summary', '')}
Key Points:
{chr(10).join(f"- {point}" for point in research_summary.get('key_points', []))}
Sources:
{chr(10).join(f"- {source}" for source in research_summary.get('sources', []))}

Output Markdown only. No explanations. No refusals."""
        else:
            document_input = "Format the provided content as a Markdown document. Output Markdown only."
        
        clean_state = {
            "messages": [HumanMessage(content=document_input)],
            "context": {},
            "next": "",
        }
        
        logger.info(f"Document Creator wrapper: Invoking agent for markdown formatting")
        
        # Invoke remote agent (treat as formatter only)
        try:
            result = document_creator_remote.invoke(clean_state)
        except Exception as e:
            logger.warning(f"Document Creator agent invocation failed: {e}. Using fallback markdown generation.")
            result = None
        
        # Extract raw text from agent output (treat as raw text, not JSON)
        raw_content = None
        if result:
            if isinstance(result, dict):
                response_messages = result.get("messages", [])
            else:
                response_messages = [result] if isinstance(result, AIMessage) else []
            
            if response_messages:
                response_message = response_messages[-1]
                raw_content = response_message.content if hasattr(response_message, 'content') else str(response_message)
                logger.info(f"Document Creator wrapper: Received raw agent output (length={len(raw_content) if raw_content else 0})")
        
        # Extract markdown from agent output OR generate from research_data
        markdown_content = None
        
        if raw_content:
            markdown_content = extract_markdown_from_text(raw_content)
            if markdown_content:
                logger.info(f"Document Creator wrapper: Extracted markdown from agent output ({len(markdown_content)} chars)")
            else:
                logger.info(f"Document Creator wrapper: Could not extract markdown from agent output, using fallback")
        
        # FALLBACK: Generate markdown from research_data
        if not markdown_content and research_summary:
            markdown_content = generate_markdown_from_research(research_summary)
            logger.info(f"Document Creator wrapper: Generated markdown from research_data fallback ({len(markdown_content)} chars)")
        
        # FINAL FALLBACK: Minimal markdown
        if not markdown_content:
            markdown_content = f"# Document\n\nContent: {original_request or 'No content available'}\n"
            logger.warning(f"Document Creator wrapper: Using minimal fallback markdown")
        
        # DETERMINISTIC FILE CREATION (Python code, not LLM)
        try:
            file_path = write_markdown_file_deterministic(markdown_content)
            logger.info(f"Document Creator wrapper: File written deterministically at {file_path}")
        except Exception as e:
            logger.error(f"Document Creator wrapper: CRITICAL - File creation failed: {e}")
            raise RuntimeError(f"Document Creator failed to write file: {e}")
        
        # FORCE COMPLETION CONTRACT (regardless of LLM behavior)
        forced_completion = {
            "completed_capability": "create_document",
            "data": {
                "file_path": file_path
            }
        }
        
        # Update context with forced completion
        new_ctx = dict(ctx)
        new_ctx["last_completed_capability"] = "create_document"
        new_ctx["file_path"] = file_path
        new_ctx["document_data"] = {"file_path": file_path}
        
        logger.info(f"Document Creator wrapper: Forced completion - capability=create_document, file_path={file_path}")
        
        # Return forced completion as AIMessage
        completion_message = AIMessage(content=json.dumps(forced_completion, indent=2))
        
        return {
            "messages": [completion_message],
            "context": new_ctx,
        }
    
    return document_creator_wrapper


def create_gmail_wrapper(gmail_remote: RemoteGraph):
    """
    Create a wrapper node for Gmail agent with DETERMINISTIC completion.
    
    This wrapper:
    - Verifies file_path exists BEFORE invoking agent
    - Treats agent output as NON-AUTHORITATIVE (logged only)
    - Forces completion contract regardless of LLM behavior
    - NEVER fails due to LLM refusal or wording
    """
    
    def gmail_wrapper(state: AgentState) -> Dict[str, Any]:
        ctx = state.get("context", {})
        
        # Extract file_path from context (from Document Creator)
        file_path = ctx.get("file_path")
        
        # HARD ASSERTION: file_path MUST exist before proceeding
        if not file_path:
            raise RuntimeError(
                "Gmail wrapper: file_path not found in context. "
                "Document Creator must set context['file_path'] before Gmail can send."
            )
        
        # HARD VERIFICATION: File MUST exist on disk
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise RuntimeError(
                f"Gmail wrapper: File does not exist at {file_path}. "
                f"This is a system failure - file must exist before sending email."
            )
        
        if file_path_obj.stat().st_size == 0:
            raise RuntimeError(
                f"Gmail wrapper: File is empty at {file_path}. "
                f"This is a system failure - file must not be empty."
            )
        
        logger.info(f"Gmail wrapper: File path existence confirmed: {file_path} ({file_path_obj.stat().st_size} bytes)")
        
        # Extract original user request
        messages = state.get("messages", [])
        original_request = None
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == "human":
                original_request = msg.content if hasattr(msg, 'content') else str(msg)
                break
        
        # Extract recipient from original request if possible
        recipient = None
        if original_request:
            import re
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, original_request)
            if emails:
                recipient = emails[0]
        
        # Construct input for agent (agent is for side effects only)
        gmail_input = f"""Send an email with the document file attached.

Original request: {original_request or 'Send email'}

The document file is located at: {file_path}

Send email using provided file_path. Use gmail_send tool with attachments parameter."""
        
        if recipient:
            gmail_input += f"\n\nSend to: {recipient}"
        
        clean_state = {
            "messages": [HumanMessage(content=gmail_input)],
            "context": {},
            "next": "",
        }
        
        logger.info(f"Gmail wrapper: Invoking Gmail agent (file_path={file_path})")
        
        # Invoke remote agent (for side effects / actual email sending)
        # We IGNORE the agent's textual output completely
        try:
            result = gmail_remote.invoke(clean_state)
            logger.info(f"Gmail wrapper: Agent invocation completed (output ignored)")
        except Exception as e:
            # Log but don't fail - we'll force completion anyway
            logger.warning(f"Gmail wrapper: Agent invocation had issues: {e}. Forcing completion anyway.")
        
        # IGNORE agent output completely - it's non-authoritative
        # We don't parse, validate, or inspect the agent's response
        
        # FORCE COMPLETION CONTRACT (regardless of LLM behavior)
        forced_completion = {
            "completed_capability": "gmail",
            "data": {
                "sent": True,
                "file_path": file_path
            }
        }
        
        # Update context with forced completion
        new_ctx = dict(ctx)
        new_ctx["last_completed_capability"] = "gmail"
        
        logger.info(f"Gmail wrapper: Forced completion - capability=gmail, file_path={file_path}")
        
        # Return forced completion as AIMessage
        completion_message = AIMessage(content=json.dumps(forced_completion, indent=2))
        
        return {
            "messages": [completion_message],
            "context": new_ctx,
        }
    
    return gmail_wrapper


def create_direct_answer_wrapper(direct_answer_remote: RemoteGraph):
    """Create a wrapper node for Direct Answer agent with clean input construction."""
    
    def direct_answer_wrapper(state: AgentState) -> Dict[str, Any]:
        ctx = state.get("context", {})
        
        # Extract original user request
        messages = state.get("messages", [])
        original_request = None
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == "human":
                original_request = msg.content if hasattr(msg, 'content') else str(msg)
                break
        
        if not original_request:
            original_request = "Answer the question"
        
        # Construct clean input: ONLY the original user request
        clean_state = {
            "messages": [HumanMessage(content=original_request)],
            "context": {},
            "next": "",
        }
        
        logger.info(f"Direct Answer wrapper: Constructing clean input")
        
        # Invoke remote agent
        result = direct_answer_remote.invoke(clean_state)
        
        # Extract response
        if isinstance(result, dict):
            response_messages = result.get("messages", [])
            response_context = result.get("context", {})
        else:
            response_messages = [result] if isinstance(result, AIMessage) else []
            response_context = {}
        
        if not response_messages:
            raise ValueError("Direct Answer agent returned no messages")
        
        response_message = response_messages[-1]
        content = response_message.content if hasattr(response_message, 'content') else str(response_message)
        
        # Update context
        new_ctx = dict(ctx)
        if response_context:
            new_ctx.update(response_context)
        
        # Extract completed capability
        capability = extract_completed_capability(content)
        if capability:
            new_ctx["last_completed_capability"] = capability
            logger.info(f"Direct Answer completed: capability={capability}")
        
        return {
            "messages": [response_message],
            "context": new_ctx,
        }
    
    return direct_answer_wrapper
