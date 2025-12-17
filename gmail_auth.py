import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

CREDS_PATH = "credentials.json"
TOKEN_PATH = "token.json"

def main():
    if os.path.exists(TOKEN_PATH):
        print("token.json already exists")
        return

    flow = InstalledAppFlow.from_client_secrets_file(
        CREDS_PATH,
        SCOPES
    )

    creds = flow.run_local_server(port=0)

    with open(TOKEN_PATH, "w") as token:
        token.write(creds.to_json())

    print("✅ token.json created successfully")

if __name__ == "__main__":
    main()
