# Base image
FROM python:3.11

# Install all required packages to run the model
RUN apt update

# Work directory
WORKDIR /app

# Copy requirements file
COPY pyproject.toml uv.lock ./
# Install dependencies
RUN pip install --no-cache-dir uv
RUN uv sync

# Copy model file
COPY ./src/model.pt .

# Copy sources
COPY src src

# Environment variables
ENV ENVIRONMENT=${ENVIRONMENT}
ENV LOG_LEVEL=${LOG_LEVEL}
ENV ENGINE_URL=${ENGINE_URL}
ENV MAX_TASKS=${MAX_TASKS}
ENV ENGINE_ANNOUNCE_RETRIES=${ENGINE_ANNOUNCE_RETRIES}
ENV ENGINE_ANNOUNCE_RETRY_DELAY=${ENGINE_ANNOUNCE_RETRY_DELAY}

# Exposed ports
EXPOSE 80

# Switch to src directory
WORKDIR "/app/src"
ENTRYPOINT ["uv", "run"]
# Command to run on start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
