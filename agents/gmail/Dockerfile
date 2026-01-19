# Dockerfile for Gmail Agent Service
# Runs on port 8080

FROM python:3.11-slim

WORKDIR /app

# Copy agent-specific requirements and install dependencies
COPY agents/gmail/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the Gmail agent code into the image (self-contained)
COPY agents/gmail/ ./

# langgraph.json and all agent modules now live directly under /app

# Expose the port the agent uses
EXPOSE 8080

# Start the Gmail agent as a LangGraph Agent Server from /app
# Use 0.0.0.0 so the container port is accessible from the host
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8080"]
