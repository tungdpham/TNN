FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    net-tools \
    iputils-ping \
    build-essential \
    g++ \
    make \
    cmake \
    libomp-dev \
    libtbb-dev \
    wget \
    curl \
    iproute2 \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN chmod +x build.sh && \
    ./build.sh --clean && \
    chmod +x entry_point.sh 
    
# Expose ports that workers will use
EXPOSE 8000 8001 8002 8003 8004