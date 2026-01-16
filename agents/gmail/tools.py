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
from pathlib import Path
from typing import Dict, Any, Optional, List


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
    creds = load_gmail_credentials(SCOPES_SEND)
    service = build("gmail", "v1", credentials=creds)

    msg = MIMEMultipart()
    msg["to"] = to
    msg["subject"] = subject
    msg["from"] = os.getenv("GMAIL_SENDER_ADDRESS", to)

    msg.attach(MIMEText(body, "plain"))

    for file_path in attachments or []:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Attachment not found: {file_path}")

        filename = os.path.basename(file_path)
        part = MIMEBase("application", "octet-stream")

        with open(file_path, "rb") as f:
            part.set_payload(f.read())

        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{filename}"',
        )

        msg.attach(part)

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
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    content = Path(file_path).read_text(encoding="utf-8")
    
    return {
        "file_path": file_path,
        "content": content
    }
