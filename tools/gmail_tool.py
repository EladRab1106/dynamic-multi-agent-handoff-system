from langchain_core.tools import tool
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from models.schemas import GmailInput
from email.mime.text import MIMEText
from base64 import urlsafe_b64encode
import os

@tool(args_schema=GmailInput)
def gmail_tool(action, query: str = "", recipient: str = "", subject: str = "", body: str = ""):
    """Search Gmail or send an email."""
    creds = Credentials(
        token=None,
        refresh_token=os.getenv("GOOGLE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=[
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.modify",
        ],
    )

    service = build("gmail", "v1", credentials=creds)

    if action == "search":
        results = service.users().messages().list(
            userId="me", q=query, maxResults=5
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

            out.append(f"FROM: {frm}\nSUBJECT: {subj}\nSNIPPET: {snippet}")

        return "\n\n".join(out)

    if action == "send":
        message = MIMEText(body)
        message["to"] = recipient
        message["subject"] = subject
        message["from"] = os.getenv("GMAIL_SENDER_ADDRESS")

        raw = {"raw": urlsafe_b64encode(message.as_bytes()).decode()}
        service.users().messages().send(userId="me", body=raw).execute()
        return "EMAIL_SENT"

    return "ERROR"