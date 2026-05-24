# Running Pipeline Scripts

A quick-start guide for launching distributed pipeline training, monitoring hardware metrics, and plotting the results.

---

## 1. Distributed Training (Torch Workers)

Launch the Torch distributed pipeline by running the respective node scripts. 

```bash
# Run on Node 0
./torch/run_pipeline_node0_1f1b_full.sh gpt2_small

# Run on Node 1
./torch/run_pipeline_node1_1f1b_full.sh gpt2_small

```

> **Note:** To scale beyond 2 nodes, you must manually update the underlying training script configuration.

---

## 2. Infrastructure Monitoring

Use these scripts to monitor network/GPU throughput in real-time and generate visual reports.

### Hardware Monitor

Tracks RDMA and GPU telemetry at a high frequency (`--interval 0.2` seconds).

```bash
python3 torch/monitor_net_gpu.py \
  --iface enp131s0f0np0 \
  --rdma-dev mlx5_0 \
  --rdma-port 1 \
  --gpu 0 \
  --interval 0.2 \
  --csv logs/monitor_gpu_net_log_bs512.csv

```

### Telemetry Plotter

Generates a quick diagnostic visualization from a specific tracking CSV.

```bash
python3 torch/plot_net_gpu.py \
  --csv logs/tnn_resnet_tinyimagenet100_bs128.csv \
  --out logs/tnn_resnet_tinyimagenet100_bs128.png

```

---

## 3. Training Analysis & Comparison

Compare custom TNN metrics against standard PyTorch baselines across both training and validation sets. This script outputs a detailed, production-ready **4-subplot comparison chart**.

```bash
python3 torch/plot_compare_resnet50.py \
  --tnn logs/tnn_imagenet100_resnet50_batch_20260519_220500.csv \
  --torch logs/resnet50_imagenet100_20260519_105543_rank0_metrics.csv \
  --tnn-val logs/tnn_imagenet100_resnet50_val_20260519_220500.csv \
  --torch-val logs/resnet50_imagenet100_20260519_105543_rank0_epoch_summary.csv \
  --out plots/resnet50_comparison_4subplots.png \
  --smooth 10 \
  --double-column

```
