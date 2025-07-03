FROM python:3.12-slim

# Set the working directory
WORKDIR /app/api

# Install system dependencies and Tesseract with Vietnamese support
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-vie \
    libleptonica-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the current code into the container
COPY . .

# Install Python requirements (including pytesseract)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8888

# Start the API
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]
