FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    net-tools \
    iputils-ping \
    build-essential \
    g++ \
    make \
    cmake \
    git \
    libomp-dev \
    libtbb-dev \
    wget \
    curl \
    iproute2 \
    nlohmann-json3-dev \
    libnuma-dev \
    libspdlog-dev \
    protobuf-compiler \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update && apt-get install -y \
    intel-oneapi-mkl-devel \
    intel-oneapi-tbb \
    && rm -rf /var/lib/apt/lists/*

RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh"

WORKDIR /app

COPY . .

RUN mkdir -p /logs && chmod +x ./build.sh && ./build.sh --mkl

# Expose ports that workers will use
EXPOSE 8000 8001 8002 8003 8004