# Dockerfile for Document Creator Agent Service
# Runs on port 8002

FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (maintaining structure)
COPY agents/ ./agents/
COPY config/ ./config/
COPY models/ ./models/
COPY tools/ ./tools/
COPY *.py ./

# Create outputs directory for generated documents
RUN mkdir -p outputs

# Expose the port the agent uses
EXPOSE 8002

# Start the agent using its existing run_service entrypoint
CMD ["python", "-m", "agents.document_creator.run_service"]
