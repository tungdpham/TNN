#!/usr/bin/env python3
import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def find_latest(log_dir: str, pattern: str) -> str | None:
    """Return the most-recently-timestamped file matching the glob pattern."""
    matches = sorted(glob.glob(os.path.join(log_dir, pattern)))
    return matches[-1] if matches else None


def smooth(s, k):
    if k <= 1:
        return s
    return s.rolling(k, min_periods=1).mean()

def load_tunx(path, all_runs=False):
    df = pd.read_csv(path)

    for col in ["step", "batch_loss", "avg_loss", "accuracy_pct", "time_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing critical data
    df = df.dropna(subset=["step", "batch_loss", "avg_loss"])
    df = df.sort_index()

    df["global_step"] = range(1, len(df) + 1)
    
    # Use batch_size from CSV or default to 128 for ResNet-50
    batch_size = df["batch_size"].iloc[0] if "batch_size" in df.columns else 128
    
    # Calculate throughput (samples/sec) if time_ms is available
    if "time_ms" in df.columns:
        df["throughput"] = batch_size / (df["time_ms"] / 1000.0)
    
    # Calculate TFLOPS for ResNet-50 (approx 12.3 GFLOPs per image for training)
    if "time_ms" in df.columns:
        gflops_per_image = 12.3
        total_gflops = batch_size * gflops_per_image
        df["tflops"] = total_gflops / (df["time_ms"] / 1000.0) / 1000.0

    return df

def load_torch(path, all_runs=False):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    df = df[df["phase"] == "train_batch"].copy()

    for col in ["step", "batch_loss", "avg_loss", "accuracy_percent", "batch_total_time", "batch_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["step", "batch_loss", "avg_loss"])
    df = df.sort_index()

    df["global_step"] = range(1, len(df) + 1)
    
    # Use batch_size from CSV or default to 128 for ResNet-50
    batch_size = df["batch_size"].iloc[0] if "batch_size" in df.columns else 128
    
    # Calculate throughput (samples/sec)
    if "batch_total_time" in df.columns:
        df["throughput"] = batch_size / df["batch_total_time"]
    
    # Calculate TFLOPS for ResNet-50 (approx 12.3 GFLOPs per image for training)
    if "batch_total_time" in df.columns:
        gflops_per_image = 12.3
        total_gflops = batch_size * gflops_per_image
        df["tflops"] = total_gflops / df["batch_total_time"] / 1000.0

    return df

def load_tunx_val(path):
    """Load tunx validation data (epoch-level)"""
    df = pd.read_csv(path)
    
    for col in ["epoch", "val_accuracy_pct", "loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["epoch"])
    df = df.sort_values("epoch")
    
    return df

def load_torch_val(path):
    """Load PyTorch epoch summary data (epoch-level)"""
    df = pd.read_csv(path)
    
    for col in ["epoch", "val_accuracy_percent", "val_loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["epoch"])
    df = df.sort_values("epoch")
    
    return df

def setup_style():
    plt.style.use("default")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.45,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

def plot_combined(tunx, torch, tunx_val, torch_val, out, double_column, xmax, ymax):
    setup_style()

    # IEEE double column width is ~7.16 inches, height adjusted for 4 subplots
    figsize = (7.16, 5.5) if double_column else (3.5, 4.5)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Set common x-axis limits if specified
    xlim = (0, xmax) if xmax is not None else None

    # Subplot 1: Rolling Training Loss
    ax1.plot(
        tunx["global_step"],
        tunx["batch_loss_s"],
        label="tunx",
        color='#1f77b4'
    )
    ax1.plot(
        torch["global_step"],
        torch["batch_loss_s"],
        label="PyTorch",
        color='#ff7f0e'
    )
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Rolling Loss")
    ax1.set_title("(a) Training Loss")
    if xlim:
        ax1.set_xlim(xlim)
    if ymax is not None:
        ax1.set_ylim(0, ymax)
    ax1.minorticks_on()
    ax1.grid(True, which="major", linestyle="--", alpha=0.35)
    ax1.legend(frameon=True, loc='upper right')

    # Subplot 2: Validation Accuracy over Epochs
    if tunx_val is not None and torch_val is not None:
        if "val_accuracy_pct" in tunx_val.columns and "val_accuracy_percent" in torch_val.columns:
            ax2.plot(
                tunx_val["epoch"],
                tunx_val["val_accuracy_pct"],
                label="tunx",
                color='#1f77b4',
                marker='o',
                markersize=4
            )
            ax2.plot(
                torch_val["epoch"],
                torch_val["val_accuracy_percent"],
                label="PyTorch",
                color='#ff7f0e',
                marker='s',
                markersize=4
            )
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Validation Accuracy (%)")
            ax2.set_title("(b) Validation Accuracy")
            ax2.xaxis.set_major_locator(plt.MultipleLocator(5))
            ax2.minorticks_on()
            ax2.grid(True, which="major", linestyle="--", alpha=0.35)
            ax2.legend(frameon=True, loc='lower right')
        else:
            ax2.text(0.5, 0.5, 'Validation accuracy\ndata not available',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("(b) Validation Accuracy")
    else:
        ax2.text(0.5, 0.5, 'Validation data\nnot available',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("(b) Validation Accuracy")

    # Subplot 3: TFLOPS over Time
    if "tflops" in tunx.columns and "tflops" in torch.columns:
        ax3.plot(
            tunx["global_step"],
            tunx["tflops_s"],
            label="tunx",
            color='#1f77b4'
        )
        ax3.plot(
            torch["global_step"],
            torch["tflops_s"],
            label="PyTorch",
            color='#ff7f0e'
        )
        ax3.set_xlabel("Training Step")
        ax3.set_ylabel("TFLOPS")
        ax3.set_title("(c) Computational Throughput")
        if xlim:
            ax3.set_xlim(xlim)
        ax3.minorticks_on()
        ax3.grid(True, which="major", linestyle="--", alpha=0.35)
        ax3.legend(frameon=True, loc='lower right')
    else:
        ax3.text(0.5, 0.5, 'TFLOPS data\nnot available',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("(c) Computational Throughput")

    # Subplot 4: Throughput (Samples/sec)
    if "throughput" in tunx.columns and "throughput" in torch.columns:
        ax4.plot(
            tunx["global_step"],
            tunx["throughput_s"],
            label="tunx",
            color='#1f77b4'
        )
        ax4.plot(
            torch["global_step"],
            torch["throughput_s"],
            label="PyTorch",
            color='#ff7f0e'
        )
        ax4.set_xlabel("Training Step")
        ax4.set_ylabel("Throughput (samples/sec)")
        ax4.set_title("(d) Training Throughput")
        if xlim:
            ax4.set_xlim(xlim)
        ax4.minorticks_on()
        ax4.grid(True, which="major", linestyle="--", alpha=0.35)
        ax4.legend(frameon=True, loc='lower right')
    else:
        ax4.text(0.5, 0.5, 'Throughput data\nnot available',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("(d) Training Throughput")

    plt.tight_layout()

    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close()

    print("Saved:", out)

def add_smoothed_columns(df, smooth_k):
    df = df.copy()
    df["batch_loss_s"] = smooth(df["batch_loss"], smooth_k)
    df["avg_loss_s"] = smooth(df["avg_loss"], smooth_k)
    
    # Add smoothing for accuracy if available
    if "accuracy_pct" in df.columns:
        df["accuracy_pct_s"] = smooth(df["accuracy_pct"], smooth_k)
    if "accuracy_percent" in df.columns:
        df["accuracy_percent_s"] = smooth(df["accuracy_percent"], smooth_k)
    
    # Add smoothing for TFLOPS if available
    if "tflops" in df.columns:
        df["tflops_s"] = smooth(df["tflops"], smooth_k)
    
    # Add smoothing for throughput if available
    if "throughput" in df.columns:
        df["throughput_s"] = smooth(df["throughput"], smooth_k)
    
    return df

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tunx", default=None, help="tunx batch training metrics CSV (auto-detected if omitted)")
    parser.add_argument("--torch", default=None, help="PyTorch batch training metrics CSV (auto-detected if omitted)")
    parser.add_argument("--tunx-val", default=None, help="tunx epoch-level validation CSV (auto-detected if omitted)")
    parser.add_argument("--torch-val", default=None, help="PyTorch epoch summary CSV (auto-detected if omitted)")
    parser.add_argument("--out", default=None, help="Output image path (default: plots/resnet50_comparison.png)")
    parser.add_argument("--log-dir", default="logs", help="Directory to search for CSV logs (default: logs)")

    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--double-column", action="store_true")

    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)

    args = parser.parse_args()

    log_dir = args.log_dir

    # Auto-detect latest log files when paths are not explicitly provided
    if args.tunx is None:
        args.tunx = find_latest(log_dir, "tunx_*resnet50*batch*.csv")
        if args.tunx:
            print(f"Auto-detected tunx batch CSV: {args.tunx}")
        else:
            raise FileNotFoundError(f"No tunx ResNet-50 batch CSV found in '{log_dir}'. Pass --tunx explicitly.")

    if args.torch is None:
        args.torch = find_latest(log_dir, "resnet50_*rank0_metrics.csv")
        if args.torch:
            print(f"Auto-detected PyTorch metrics CSV: {args.torch}")
        else:
            raise FileNotFoundError(f"No PyTorch ResNet-50 metrics CSV found in '{log_dir}'. Pass --torch explicitly.")

    if args.tunx_val is None:
        args.tunx_val = find_latest(log_dir, "tunx_*resnet50*epoch*.csv")
        if args.tunx_val:
            print(f"Auto-detected tunx val CSV: {args.tunx_val}")

    if args.torch_val is None:
        args.torch_val = find_latest(log_dir, "resnet50_*rank0_epoch_summary.csv")
        if args.torch_val:
            print(f"Auto-detected PyTorch epoch summary CSV: {args.torch_val}")

    if args.out is None:
        os.makedirs("plots", exist_ok=True)
        args.out = os.path.join("plots", "resnet50_comparison.png")

    tunx = load_tunx(args.tunx, all_runs=True)
    torch = load_torch(args.torch, all_runs=True)

    tunx = add_smoothed_columns(tunx, args.smooth)
    torch = add_smoothed_columns(torch, args.smooth)

    print("tunx training points:", len(tunx))
    print("PyTorch training points:", len(torch))

    # Load validation data if provided
    tunx_val = None
    torch_val = None
    if args.tunx_val:
        tunx_val = load_tunx_val(args.tunx_val)
        print("tunx validation epochs:", len(tunx_val))
    if args.torch_val:
        torch_val = load_torch_val(args.torch_val)
        print("PyTorch validation epochs:", len(torch_val))

    plot_combined(
        tunx=tunx,
        torch=torch,
        tunx_val=tunx_val,
        torch_val=torch_val,
        out=args.out,
        double_column=args.double_column,
        xmax=args.xmax,
        ymax=args.ymax,
    )

if __name__ == "__main__":
    main()
