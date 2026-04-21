#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYFILE="${SCRIPT_DIR}/deepspeed_distributed.py"
HOSTFILE="${SCRIPT_DIR}/deepspeed_hostfile"
DS_CONFIG="${SCRIPT_DIR}/deepspeed_config.json"

MODEL="${1:-resnet9_cifar10}"
SEQ_LEN="${SEQ_LEN:-512}"
LOG_DIR="${LOG_DIR:-logs}"

export MASTER_ADDR=10.10.0.1
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

# Data paths
export CIFAR10_BIN_ROOT="${CIFAR10_BIN_ROOT:-data/cifar-10-batches-bin}"
export CIFAR100_BIN_ROOT="${CIFAR100_BIN_ROOT:-data/cifar-100-binary}"
export TINY_IMAGENET_ROOT="${TINY_IMAGENET_ROOT:-data/tiny-imagenet-200}"
export IMAGENET100_ROOT="${IMAGENET100_ROOT:-data/imagenet-100}"

echo "[Node0 DeepSpeed] MASTER_ADDR=${MASTER_ADDR}"
echo "[Node0 DeepSpeed] MASTER_PORT=${MASTER_PORT}"
echo "[Node0 DeepSpeed] hostname=$(hostname)"
echo "[Node0 DeepSpeed] model=${MODEL}"
echo "[Node0 DeepSpeed] hostfile=${HOSTFILE}"

deepspeed \
  --hostfile="${HOSTFILE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PYFILE}" \
  --model "${MODEL}" \
  --seq-len "${SEQ_LEN}" \
  --log-dir "${LOG_DIR}" \
  --deepspeed_config "${DS_CONFIG}"
