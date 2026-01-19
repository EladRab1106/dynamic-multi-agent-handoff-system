# Dockerfile for Researcher Agent Service
# Runs on port 8080

FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies (per-agent requirements)
COPY agents/researcher/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (maintaining structure)
COPY agents/ ./agents/
COPY config/ ./config/
COPY models/ ./models/
COPY tools/ ./tools/
COPY *.py ./

# Switch to the agent directory so langgraph can find langgraph.json
WORKDIR /app/agents/researcher

# Expose the port the agent uses
EXPOSE 8080

# Start the Researcher as a LangGraph Agent Server
# Use 0.0.0.0 so the container port is accessible from the host
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8080"]
