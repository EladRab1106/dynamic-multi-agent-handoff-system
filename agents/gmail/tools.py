"""
Gmail tools for Gmail agent.

Local copy to ensure the agent is fully self-contained.
All Gmail tools are owned entirely by the Gmail agent service.
"""

from langchain_core.tools import tool
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from base64 import urlsafe_b64decode, urlsafe_b64encode
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import tool usage tracker from gmail_agent
try:
    from gmail_agent import mark_tool_used
except ImportError:
    # Fallback if imported outside agent context
    def mark_tool_used():
        pass


logger = logging.getLogger(__name__)


SCOPES_SEARCH = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

SCOPES_SEND = [
    "https://www.googleapis.com/auth/gmail.send",
]


def load_gmail_credentials(scopes):
    """Load Gmail OAuth credentials."""
    token_path = os.getenv("GMAIL_TOKEN_PATH", "token.json")
    if not os.path.exists(token_path):
        raise RuntimeError(
            "Gmail token.json not found. "
            "Run gmail_auth.py to generate OAuth credentials."
        )
    return Credentials.from_authorized_user_file(token_path, scopes)


@tool
def gmail_search(query: str) -> Dict[str, Any]:
    """
    Search Gmail using a query string.
    
    Args:
        query: Gmail search query (e.g., "subject:Apple Inc. Financial Report 2023")
    
    Returns:
        Dict with email data if found, or {"email": None, "reason": "not_found"} if not found.
        If found, returns: {
            "email": {
                "from": "...",
                "subject": "...",
                "date": "...",
                "body": "..."
            }
        }
    """
    mark_tool_used()
    creds = load_gmail_credentials(SCOPES_SEARCH)
    service = build("gmail", "v1", credentials=creds)

    results = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=5,
    ).execute()

    messages = results.get("messages", [])
    if not messages:
        return {"email": None, "reason": "not_found"}

    # Get the first matching email with full body
    first_msg_id = messages[0]["id"]
    msg = service.users().messages().get(
        userId="me",
        id=first_msg_id,
        format="full",
    ).execute()

    headers = msg.get("payload", {}).get("headers", [])
    frm = next((h["value"] for h in headers if h["name"] == "From"), "")
    subj = next((h["value"] for h in headers if h["name"] == "Subject"), "")
    date = next((h["value"] for h in headers if h["name"] == "Date"), "")

    # Extract full body from message payload
    body = ""
    payload = msg.get("payload", {})

    # Handle different payload structures
    if "body" in payload and "data" in payload["body"]:
        body_data = payload["body"]["data"]
        body = urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
    elif "parts" in payload:
        # Multipart message - extract text from parts
        for part in payload.get("parts", []):
            if part.get("mimeType") == "text/plain":
                if "body" in part and "data" in part["body"]:
                    body_data = part["body"]["data"]
                    body = urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
                    break

    # Fallback to snippet if body extraction failed
    if not body:
        body = msg.get("snippet", "")

    return {
        "email": {
            "from": frm,
            "subject": subj,
            "date": date,
            "body": body
        }
    }


@tool
def gmail_send(
    to: str,
    subject: str,
    body: str,
    attachments: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Send an email via Gmail.
    
    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body text
        attachments: Optional list of file paths to attach
    
    Returns:
        Dict with status: {"status": "sent", "to": "...", "subject": "..."}
    """
    mark_tool_used()
    creds = load_gmail_credentials(SCOPES_SEND)
    service = build("gmail", "v1", credentials=creds)

    logger.info("gmail_send: preparing email", extra={"to": to, "subject": subject, "attachments_count": len(attachments or [])})

    msg = MIMEMultipart()
    msg["to"] = to
    msg["subject"] = subject
    msg["from"] = os.getenv("GMAIL_SENDER_ADDRESS", to)

    msg.attach(MIMEText(body, "plain"))

    for file_path in attachments or []:
        # Normalize and resolve path for robust logging/validation
        resolved_path = Path(file_path).expanduser().resolve()
        resolved_file_path = resolved_path.as_posix()

        logger.info(
            "gmail_send: processing attachment",
            extra={"original_path": file_path, "resolved_path": resolved_file_path},
        )

        if not resolved_path.exists():
            error_msg = f"Attachment not found: {resolved_file_path} (original: {file_path})"
            logger.error(f"gmail_send: {error_msg}")
            raise FileNotFoundError(error_msg)

        file_size = resolved_path.stat().st_size
        if file_size == 0:
            error_msg = f"Attachment file is empty: {resolved_file_path}"
            logger.error(f"gmail_send: {error_msg}")
            raise ValueError(error_msg)

        filename = resolved_path.name
        part = MIMEBase("application", "octet-stream")

        with open(resolved_file_path, "rb") as f:
            part.set_payload(f.read())

        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{filename}"',
        )

        msg.attach(part)
        # NOTE: logging.LogRecord already has a built-in "filename" attribute.
        # Using "filename" in extra will raise KeyError("Attempt to overwrite 'filename' in LogRecord").
        # Use a distinct key name for attachment filename to avoid breaking the tool.
        logger.info(
            "gmail_send: successfully attached file",
            extra={"attachment_filename": filename, "attachment_size_bytes": file_size},
        )

    raw = {
        "raw": urlsafe_b64encode(msg.as_bytes()).decode()
    }

    service.users().messages().send(
        userId="me",
        body=raw,
    ).execute()

    return {
        "status": "sent",
        "to": to,
        "subject": subject
    }


@tool
def materialize_base64_attachment(
    filename: str,
    file_base64: str,
    subdir: str = "attachments"
) -> Dict[str, Any]:
    """
    Decode a base64-encoded file and write it to disk so it can be attached to an email.

    Args:
        filename: Desired filename for the attachment (e.g., "report.md").
        file_base64: Base64-encoded file content produced by the Document Creator agent.
        subdir: Optional subdirectory under the base attachment directory (default: "attachments").

    Returns:
        {
          "file_path": "relative/path/to/attachment",
          "abs_file_path": "/absolute/path/to/attachment",
          "size_bytes": <file_size_in_bytes>
        }
    """
    mark_tool_used()

    if not filename:
        filename = "attachment.bin"

    # Choose a base directory for attachments (can be overridden via env)
    base_dir = Path(os.getenv("GMAIL_ATTACHMENT_DIR", "outputs")) / subdir
    base_dir.mkdir(parents=True, exist_ok=True)

    safe_name = os.path.basename(filename)
    target_path = base_dir / safe_name

    # If file already exists, add a numeric suffix to avoid overwriting
    counter = 1
    while target_path.exists():
        stem = Path(safe_name).stem
        suffix = Path(safe_name).suffix
        target_path = base_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    try:
        file_bytes = base64.b64decode(file_base64)
    except Exception:
        logger.exception("materialize_base64_attachment: failed to decode base64 content")
        raise

    target_path.write_bytes(file_bytes)
    size_bytes = target_path.stat().st_size

    abs_path = target_path.resolve()
    result = {
        "file_path": str(target_path),
        "abs_file_path": abs_path.as_posix(),
        "size_bytes": size_bytes,
    }

    # NOTE: logging.LogRecord already has a built-in "filename" attribute.
    # Using "filename" in extra will raise KeyError("Attempt to overwrite 'filename' in LogRecord").
    # Use a distinct key name (e.g. "attachment_filename") instead to avoid breaking the tool.
    logger.info(
        "materialize_base64_attachment: created attachment file",
        extra={
            "attachment_filename": safe_name,
            "file_path": result["file_path"],
            "abs_file_path": result["abs_file_path"],
            "size_bytes": size_bytes,
        },
    )

    return result


@tool
def read_file_content(file_path: str) -> Dict[str, Any]:
    """
    Read the content of a file.
    
    Args:
        file_path: Path to the file to read
    
    Returns:
        Dict with content and file_path:
        {
            "file_path": "...",
            "content": "..."
        }
    
    Raises:
        FileNotFoundError: If file does not exist
        OSError: If file cannot be read
    """
    mark_tool_used()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    content = Path(file_path).read_text(encoding="utf-8")
    
    return {
        "file_path": file_path,
        "content": content
    }
