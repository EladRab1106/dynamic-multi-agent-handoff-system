from langchain_core.tools import tool
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from models.schemas import GmailInput

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from base64 import urlsafe_b64encode
import os
from typing import List, Optional


SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
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


@tool(args_schema=GmailInput)
def gmail_tool(
    action: str,
    query: str = "",
    recipient: str = "",
    subject: str = "",
    body: str = "",
    attachments: Optional[List[str]] = None,
):
    """Search Gmail or send an email (supports attachments)."""

    creds = load_gmail_credentials()
    service = build("gmail", "v1", credentials=creds)

    # ------------------
    # SEARCH EMAILS
    # ------------------
    if action == "search":
        results = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=5,
        ).execute()

        messages = results.get("messages", [])
        if not messages:
            return "NO_RESULTS"

        out = []
        for m in messages:
            msg = service.users().messages().get(
                userId="me",
                id=m["id"],
                format="metadata",
                metadataHeaders=["From", "Subject"],
            ).execute()

            headers = msg.get("payload", {}).get("headers", [])
            frm = next((h["value"] for h in headers if h["name"] == "From"), "")
            subj = next((h["value"] for h in headers if h["name"] == "Subject"), "")
            snippet = msg.get("snippet", "")

            out.append(
                f"FROM: {frm}\nSUBJECT: {subj}\nSNIPPET: {snippet}"
            )

        return "\n\n".join(out)

    # ------------------
    # SEND EMAIL
    # ------------------
    if action == "send":
        msg = MIMEMultipart()
        msg["to"] = recipient
        msg["subject"] = subject
        msg["from"] = os.getenv("GMAIL_SENDER_ADDRESS", recipient)

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

        return "EMAIL_SENT"

    return "ERROR"
