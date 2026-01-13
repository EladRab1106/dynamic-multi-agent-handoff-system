# Dockerfile for Researcher Agent Service
# Runs on port 8001

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

# Expose the port the agent uses
EXPOSE 8001

# Start the agent using its existing run_service entrypoint
CMD ["python", "-m", "agents.researcher.run_service"]
