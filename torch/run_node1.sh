#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-resnet9_cifar10}"
MICRO_BS="${MICRO_BS:-4}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"

export GLOO_SOCKET_IFNAME=enp131s0f0np0
export NCCL_SOCKET_IFNAME=enp131s0f0np0
export MASTER_ADDR=10.10.0.1
export MASTER_PORT=29500
export DIST_BACKEND="${DIST_BACKEND}"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

echo "[Node1] MASTER_ADDR=${MASTER_ADDR}"
echo "[Node1] MASTER_PORT=${MASTER_PORT}"
echo "[Node1] GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
echo "[Node1] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "[Node1] hostname=$(hostname)"

torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=1 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  torch/torch_pipeline_parallel.py \
  --model "${MODEL}" \
  --micro-bs "${MICRO_BS}"