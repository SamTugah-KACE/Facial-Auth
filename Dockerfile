# Use an official Python runtime as a parent image
ARG PYTHON_VERSION=3.10-slim
FROM python:${PYTHON_VERSION} as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
# Keeps Python from buffering stdout and stderr.
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /

# Copy requirements.txt into the container
COPY requirements.txt .

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --default-timeout=10000 --retries 10 --progress-bar off -r requirements.txt

# Copy the rest of the source code into the container
COPY . .

# Ensure the dataset directory exists and has appropriate permissions
RUN mkdir -p ./dataset && chmod -R 755 ./dataset

# Expose the port that the application listens on
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
