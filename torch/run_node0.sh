#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYFILE="${SCRIPT_DIR}/torch_pipeline_parallel.py"

export PYTHONUNBUFFERED=1

export IMAGENET100_ROOT=/home/mpeclab/TNN/data/imagenet-100
export BATCH_SIZE=128
export EPOCHS=20
export MICRO_BS=4
export PRINT_FREQ=1

export GLOO_SOCKET_IFNAME=enp131s0f0np0
export NCCL_SOCKET_IFNAME=enp131s0f0np0
export MASTER_ADDR=10.10.0.2
export MASTER_PORT=29500

export DIST_BACKEND=gloo
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=5
export NCCL_IB_ADDR_FAMILY=AF_INET
export NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

MODEL="${1:-resnet50_imagenet100}"
MICRO_BS="${MICRO_BS:-4}"

echo "========== NODE0 DEBUG =========="
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "PYFILE=${PYFILE}"
echo "PYFILE_EXISTS=$(test -f "${PYFILE}" && echo yes || echo no)"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "DIST_BACKEND=${DIST_BACKEND}"
echo "HCA=${NCCL_IB_HCA}"
echo "GID_INDEX=${NCCL_IB_GID_INDEX}"
echo "DATA=${IMAGENET100_ROOT}"
echo "DATA_EXISTS=$(test -d "${IMAGENET100_ROOT}" && echo yes || echo no)"
find "${IMAGENET100_ROOT}" -maxdepth 2 -type d | head -20 || true
find "${IMAGENET100_ROOT}" -type f \( -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" \) | head -5 || true
echo "TRAIN_X1_IMAGES=$(find "${IMAGENET100_ROOT}/train.X1" -type f \( -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
ip -br addr | grep -E "${GLOO_SOCKET_IFNAME}|${NCCL_SOCKET_IFNAME}" || true
echo "================================="

pkill -f torchrun || true
pkill -f torch_pipeline_parallel.py || true

torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PYFILE}" \
  --model "${MODEL}" \
  --micro-bs "${MICRO_BS}" \
  --print-freq 1