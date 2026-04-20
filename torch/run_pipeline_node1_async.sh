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
export BATCH_SIZE MICRO_BS SEQ_LEN
exec torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" torch/torch_pipeline_parallel_async_mb.py --model "${MODEL}" --micro-bs "${MICRO_BS}" --seq-len "${SEQ_LEN}"
