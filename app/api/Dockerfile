FROM python:3.12-slim

# Set the working directory
WORKDIR /app/api

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current code into the container
COPY . .

# Install cython first to avoid build errors
RUN pip install --upgrade pip && pip install cython

# Clone and install DocLayout-YOLO in editable mode
RUN pip install -e /app/api/DocLayout-YOLO

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8888

# Start the API
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]
