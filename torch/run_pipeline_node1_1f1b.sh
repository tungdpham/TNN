#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-resnet9_cifar10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MICRO_BS="${MICRO_BS:-64}"
SEQ_LEN="${SEQ_LEN:-512}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"

export GLOO_SOCKET_IFNAME=enp5s0f0np0
export NCCL_SOCKET_IFNAME=enp5s0f0np0
export NET_LOG_IFNAME=enp5s0f0np0

export MASTER_ADDR=10.10.0.2
export MASTER_PORT=29500
export DIST_BACKEND="${DIST_BACKEND}"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export BATCH_SIZE
export MICRO_BS
export SEQ_LEN

echo "[Node1-1F1B] MODEL=${MODEL}"
echo "[Node1-1F1B] MASTER_ADDR=${MASTER_ADDR}"
echo "[Node1-1F1B] MASTER_PORT=${MASTER_PORT}"
echo "[Node1-1F1B] GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
echo "[Node1-1F1B] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "[Node1-1F1B] MICRO_BS=${MICRO_BS}"
echo "[Node1-1F1B] hostname=$(hostname)"

torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  torch/torch_pipeline_parallel_1f1b.py \
  --model "${MODEL}" \
  --micro-bs "${MICRO_BS}" \
  --seq-len "${SEQ_LEN}"
