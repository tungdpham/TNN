#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--metrics", required=True)
parser.add_argument("--net0", required=True)
parser.add_argument("--net1", default=None)
parser.add_argument("--outdir", default="plots")
args = parser.parse_args()

outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(args.metrics)
n0 = pd.read_csv(args.net0)
n1 = pd.read_csv(args.net1) if args.net1 else None

train = m[m["phase"]=="train_batch"]
ep = m[m["phase"]=="epoch_summary"]

if not train.empty:
    plt.figure(figsize=(8,4.5)); plt.plot(train["step"], train["avg_loss"]); plt.xlabel("Batch step"); plt.ylabel("Avg train loss"); plt.title("Training loss over batches"); plt.tight_layout(); plt.savefig(outdir/"train_loss_batches.png", dpi=150); plt.close()
    plt.figure(figsize=(8,4.5)); plt.plot(train["step"], train["accuracy_percent"]); plt.xlabel("Batch step"); plt.ylabel("Train accuracy (%)"); plt.title("Training accuracy over batches"); plt.tight_layout(); plt.savefig(outdir/"train_acc_batches.png", dpi=150); plt.close()

if not ep.empty:
    plt.figure(figsize=(8,4.5)); plt.plot(ep["epoch"], ep["accuracy_percent"]); plt.xlabel("Epoch"); plt.ylabel("Test accuracy (%)"); plt.title("Test accuracy over epochs"); plt.tight_layout(); plt.savefig(outdir/"test_acc_epochs.png", dpi=150); plt.close()
    plt.figure(figsize=(8,4.5)); plt.plot(ep["epoch"], ep["avg_loss"]); plt.xlabel("Epoch"); plt.ylabel("Test loss"); plt.title("Test loss over epochs"); plt.tight_layout(); plt.savefig(outdir/"test_loss_epochs.png", dpi=150); plt.close()

def plot_net(df, suffix):
    if df is None or df.empty: return
    for phase in sorted(df["phase"].dropna().unique()):
        d = df[df["phase"]==phase]
        if d.empty: continue
        plt.figure(figsize=(8,4.5))
        plt.plot(d["step"], d["tx_MBps"], label="TX MB/s")
        plt.plot(d["step"], d["rx_MBps"], label="RX MB/s")
        plt.xlabel("Step"); plt.ylabel("MB/s"); plt.title(f"Network traffic ({suffix}, {phase})")
        plt.legend(); plt.tight_layout(); plt.savefig(outdir/f"net_{suffix}_{phase}.png", dpi=150); plt.close()

plot_net(n0, "rank0")
plot_net(n1, "rank1")
