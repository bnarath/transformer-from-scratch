# Base image with PyTorch and GPU support
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest

# Set the working directory inside the container
WORKDIR /app

# Install any additional dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy local package code to the container
COPY . .

# Install Python dependencies (setup.py or requirements.txt)
RUN pip install --no-cache-dir .

# Give executable permission to run.sh script
RUN chmod +x /app/run.sh

# Set entrypoint
ENTRYPOINT ["./run.sh"]
