#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYFILE="${SCRIPT_DIR}/torch_pipeline_parallel.py"

export GLOO_SOCKET_IFNAME=enp5s0f0np0
export NCCL_SOCKET_IFNAME=enp5s0f0np0
export MASTER_ADDR=10.10.0.1
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export DIST_BACKEND=nccl

MODEL="${1:-resnet9_cifar10}"
MICRO_BS="${MICRO_BS:-4}"

python - <<'PYCHECK'
import os, socket
print('[Node0] MASTER_ADDR=', os.getenv('MASTER_ADDR'))
print('[Node0] MASTER_PORT=', os.getenv('MASTER_PORT'))
print('[Node0] GLOO_SOCKET_IFNAME=', os.getenv('GLOO_SOCKET_IFNAME'))
print('[Node0] NCCL_SOCKET_IFNAME=', os.getenv('NCCL_SOCKET_IFNAME'))
print('[Node0] hostname=', socket.gethostname())
PYCHECK

torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  torch/torch_pipeline_parallel.py \
  --model "${MODEL}" \
  --micro-bs "${MICRO_BS}"