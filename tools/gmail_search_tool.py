"""
Gmail search tool - searches for emails and returns structured data.
"""

from langchain_core.tools import tool
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from base64 import urlsafe_b64decode
import os
from typing import Dict, Any, Optional


SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]


def load_gmail_credentials():
    token_path = os.getenv("GMAIL_TOKEN_PATH", "token.json")
    if not os.path.exists(token_path):
        raise RuntimeError(
            "Gmail token.json not found. "
            "Run gmail_auth.py to generate OAuth credentials."
        )
    return Credentials.from_authorized_user_file(token_path, SCOPES)


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
    creds = load_gmail_credentials()
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
