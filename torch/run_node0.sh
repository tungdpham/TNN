#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYFILE="${SCRIPT_DIR}/torch/torch_pipeline_parallel.py"

MODEL="${1:-resnet9_cifar10}"
MICRO_BS="${MICRO_BS:-4}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"

export GLOO_SOCKET_IFNAME=enp5s0f0np0
export NCCL_SOCKET_IFNAME=enp5s0f0np0
export MASTER_ADDR=10.10.0.1
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export DIST_BACKEND="${DIST_BACKEND}"
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "[Node0] MASTER_ADDR=${MASTER_ADDR}"
echo "[Node0] MASTER_PORT=${MASTER_PORT}"
echo "[Node0] GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
echo "[Node0] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "[Node0] hostname=$(hostname)"

torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  ${PYFILE} \
  --model "${MODEL}" \
  --micro-bs "${MICRO_BS}"