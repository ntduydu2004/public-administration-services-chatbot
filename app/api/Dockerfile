FROM python:3.12

# Set the path to the API directory
WORKDIR /app/api

# Copy the codebase into the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y graphviz && \
    rm -rf /var/lib/apt/lists/*

# Install the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8888

ENTRYPOINT ["uvicorn", "main:app", "--host", "api", "--port", "8888", "--reload"]
