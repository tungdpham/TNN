#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def pick_cols(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot network/RDMA + GPU metrics from CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="logs/monitor_plot.png")
    parser.add_argument("--time-col", default="elapsed_sec")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError("CSV empty")

    if args.time_col not in df.columns:
        raise ValueError(f"Missing time column: {args.time_col}")

    x = df[args.time_col]

    # detect columns
    net_rx = pick_cols(df, ["net_rx_Gbps", "rx_Gbps"])
    net_tx = pick_cols(df, ["net_tx_Gbps", "tx_Gbps"])
    roce_rx = pick_cols(df, ["roce_rx_Gbps"])
    roce_tx = pick_cols(df, ["roce_tx_Gbps"])

    gpu_util = pick_cols(df, ["gpu_util_percent"])
    gpu_mem = pick_cols(df, ["gpu_mem_used_mb"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ---- Network / RDMA ----
    ax = axes[0]

    if net_rx:
        ax.plot(x, df[net_rx], label="NET RX (Gbps)")
    if net_tx:
        ax.plot(x, df[net_tx], label="NET TX (Gbps)")
    if roce_rx:
        ax.plot(x, df[roce_rx], linestyle="--", label="ROCE RX (Gbps)")
    if roce_tx:
        ax.plot(x, df[roce_tx], linestyle="--", label="ROCE TX (Gbps)")

    ax.set_title("Network / RDMA Throughput")
    ax.set_ylabel("Gbps")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- GPU util ----
    ax = axes[1]
    if gpu_util:
        ax.plot(x, df[gpu_util], label="GPU Util (%)")

    if "gpu_mem_util_percent" in df.columns:
        ax.plot(x, df["gpu_mem_util_percent"], label="GPU Mem Util (%)")

    ax.set_title("GPU Utilization")
    ax.set_ylabel("%")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- GPU memory ----
    ax = axes[2]
    if gpu_mem:
        ax.plot(x, df[gpu_mem], label="GPU Mem Used (MB)")

    if "gpu_mem_total_mb" in df.columns:
        ax.plot(x, df["gpu_mem_total_mb"], label="GPU Mem Total (MB)")

    ax.set_title("GPU Memory")
    ax.set_ylabel("MB")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"[OK] Saved plot: {args.out}")

    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    main()