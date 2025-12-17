from langchain_core.tools import tool
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from models.schemas import GmailInput

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from base64 import urlsafe_b64encode
import os


SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]


def load_gmail_credentials():
    token_path = os.getenv("GMAIL_TOKEN_PATH", "token.json")
    creds_path = os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")

    if os.path.exists(token_path):
        return Credentials.from_authorized_user_file(token_path, SCOPES)

    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
    creds = flow.run_local_server(port=0)

    with open(token_path, "w") as token_file:
        token_file.write(creds.to_json())

    return creds


@tool(args_schema=GmailInput)
def gmail_tool(
    action: str,
    query: str = "",
    recipient: str = "",
    subject: str = "",
    body: str = "",
    attachments=None,
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
    # SEND EMAIL (with attachments)
    # ------------------
    if action == "send":
        msg = MIMEMultipart()
        msg["to"] = recipient
        msg["subject"] = subject
        msg["from"] = os.getenv("GMAIL_SENDER_ADDRESS")

        # Body
        msg.attach(MIMEText(body, "plain"))

        # Attachments
        if attachments:
            for file_path in attachments:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Attachment not found: {file_path}"
                    )

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
