# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    make \
    cmake \
    libomp-dev \
    wget \
    curl \
    iproute2 \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Create build directory and compile
RUN make clean && \
    make network_worker && \
    make distributed_pipeline_docker

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose ports that workers will use
EXPOSE 8000 8001 8002 8003 8004

# Default command (will be overridden in docker-compose)
CMD ["./distributed_pipeline_docker"]
