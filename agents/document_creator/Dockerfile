# Dockerfile for Document Creator Agent Service
# Runs on port 8080

FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies (per-agent requirements)
COPY agents/document_creator/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the Document Creator agent code into the image (self-contained)
COPY agents/document_creator/ ./

# Create outputs directory for generated documents
RUN mkdir -p outputs

# langgraph.json and all agent modules now live directly under /app

# Expose the port the agent uses
EXPOSE 8080

# Start the Document Creator as a LangGraph Agent Server from /app
# Use 0.0.0.0 so the container port is accessible from the host
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8080"]
