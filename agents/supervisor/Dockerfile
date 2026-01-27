# Dockerfile for Supervisor Agent Service
# Runs on port 8080
#
# NOTE: This Dockerfile assumes the build context is the parent directory
# that contains both:
#   - this repo at ./dynamic-multi-agent-handoff-system
#   - the AZTM repo at ./aztm-clone

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies for the Supervisor agent
COPY dynamic-multi-agent-handoff-system/agents/supervisor/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

# Install AZTM from the sibling azTM repository (no local copy in this repo required)
COPY aztm-clone/aztm/ /aztm
RUN pip install --no-cache-dir /aztm

# Copy only the Supervisor agent code into the image (self-contained)
COPY dynamic-multi-agent-handoff-system/agents/supervisor/ ./

# langgraph.json and all agent modules now live directly under /app

# Expose the port the agent uses
EXPOSE 8080

# Start the Supervisor as a LangGraph Agent Server from /app
# Use 0.0.0.0 so the container port is accessible from the host
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8080"]
