#!/usr/bin/env python3
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_latest(log_dir: str, pattern: str) -> str | None:
    """Return the most-recently-timestamped file matching the glob pattern."""
    matches = sorted(glob.glob(os.path.join(log_dir, pattern)))
    return matches[-1] if matches else None


def smooth(s, k):
    if k <= 1:
        return s
    return s.rolling(k, min_periods=1).mean()


def load_tnn(path, seq_len, tnn_batch_size, flops_per_token):
    df = pd.read_csv(path)

    for col in ["step", "batch_loss", "avg_loss", "time_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["step", "batch_loss", "avg_loss"])
    df = df.sort_index()
    df["global_step"] = range(1, len(df) + 1)

    if "time_ms" in df.columns and tnn_batch_size and seq_len:
        tokens_per_batch = tnn_batch_size * seq_len
        time_sec = df["time_ms"] / 1000.0
        df["tokens_per_sec"] = tokens_per_batch / time_sec
        df["tflops"] = (tokens_per_batch * flops_per_token) / time_sec / 1e12

    return df


def load_torch(path, seq_len, flops_per_token):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df[df["phase"] == "train_batch"].copy()

    for col in ["step", "batch_loss", "avg_loss", "batch_total_time", "batch_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["step", "batch_loss", "avg_loss"])
    df = df.sort_index()
    df["global_step"] = range(1, len(df) + 1)

    if "batch_total_time" in df.columns and "batch_size" in df.columns and seq_len:
        tokens_per_batch = df["batch_size"] * seq_len
        df["tokens_per_sec"] = tokens_per_batch / df["batch_total_time"]
        df["tflops"] = (tokens_per_batch * flops_per_token) / df["batch_total_time"] / 1e12

    return df


def load_tnn_val(path):
    """Load TNN epoch-level validation data."""
    df = pd.read_csv(path)

    for col in ["epoch", "val_perplexity", "val_loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["epoch"])
    df = df.sort_values("epoch")

    # Derive PPL from val_loss if val_perplexity is absent
    if "val_perplexity" not in df.columns and "val_loss" in df.columns:
        df["val_perplexity"] = np.exp(df["val_loss"])

    return df


def load_torch_val(path):
    """Load PyTorch epoch summary data, deriving PPL from val_loss."""
    df = pd.read_csv(path)

    for col in ["epoch", "val_loss", "val_accuracy_percent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["epoch"])
    df = df.sort_values("epoch")

    if "val_loss" in df.columns:
        df["val_perplexity"] = np.exp(df["val_loss"])

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


def plot_combined(tnn, torch_df, tnn_val, torch_val, out, double_column, xmax, ymax):
    setup_style()

    figsize = (7.16, 5.5) if double_column else (3.5, 4.5)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    xlim = (0, xmax) if xmax is not None else None

    # ── (a) Rolling Training Loss ────────────────────────────────────────────
    ax1.plot(tnn["global_step"], tnn["batch_loss_s"], label="TNN", color="#1f77b4")
    ax1.plot(torch_df["global_step"], torch_df["batch_loss_s"], label="PyTorch", color="#ff7f0e")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Rolling Loss")
    ax1.set_title("(a) Training Loss")
    if xlim:
        ax1.set_xlim(xlim)
    if ymax is not None:
        ax1.set_ylim(0, ymax)
    ax1.minorticks_on()
    ax1.grid(True, which="major", linestyle="--", alpha=0.35)
    ax1.legend(frameon=True, loc="upper right")

    # ── (b) Training Perplexity ──────────────────────────────────────────────
    ax2.plot(tnn["global_step"], tnn["train_perplexity_s"], label="TNN", color="#1f77b4")
    ax2.plot(torch_df["global_step"], torch_df["train_perplexity_s"], label="PyTorch", color="#ff7f0e")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("(b) Training Perplexity")
    if xlim:
        ax2.set_xlim(xlim)
    ax2.minorticks_on()
    ax2.grid(True, which="major", linestyle="--", alpha=0.35)
    ax2.legend(frameon=True, loc="upper right")

    # ── (c) TFLOPS ───────────────────────────────────────────────────────────
    if "tflops" in tnn.columns and "tflops" in torch_df.columns:
        ax3.plot(tnn["global_step"], tnn["tflops_s"], label="TNN", color="#1f77b4")
        ax3.plot(torch_df["global_step"], torch_df["tflops_s"], label="PyTorch", color="#ff7f0e")
        ax3.set_xlabel("Training Step")
        ax3.set_ylabel("TFLOPS")
        ax3.set_title("(c) Computational Throughput")
        if xlim:
            ax3.set_xlim(xlim)
        ax3.minorticks_on()
        ax3.grid(True, which="major", linestyle="--", alpha=0.35)
        ax3.legend(frameon=True, loc="lower right")
    else:
        ax3.text(0.5, 0.5, "TFLOPS data\nnot available",
                 ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("(c) Computational Throughput")

    # ── (d) Tokens/s ─────────────────────────────────────────────────────────
    if "tokens_per_sec" in tnn.columns and "tokens_per_sec" in torch_df.columns:
        ax4.plot(tnn["global_step"], tnn["tokens_per_sec_s"], label="TNN", color="#1f77b4")
        ax4.plot(torch_df["global_step"], torch_df["tokens_per_sec_s"], label="PyTorch", color="#ff7f0e")
        ax4.set_xlabel("Training Step")
        ax4.set_ylabel("Tokens / sec")
        ax4.set_title("(d) Training Throughput")
        if xlim:
            ax4.set_xlim(xlim)
        ax4.minorticks_on()
        ax4.grid(True, which="major", linestyle="--", alpha=0.35)
        ax4.legend(frameon=True, loc="lower right")
    else:
        ax4.text(0.5, 0.5, "Throughput data\nnot available",
                 ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("(d) Training Throughput")

    plt.tight_layout()
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


def add_smoothed_columns(df, smooth_k):
    df = df.copy()
    df["batch_loss_s"] = smooth(df["batch_loss"], smooth_k)
    df["avg_loss_s"] = smooth(df["avg_loss"], smooth_k)
    df["train_perplexity"] = np.exp(df["avg_loss"])
    df["train_perplexity_s"] = smooth(df["train_perplexity"], smooth_k)
    if "tflops" in df.columns:
        df["tflops_s"] = smooth(df["tflops"], smooth_k)
    if "tokens_per_sec" in df.columns:
        df["tokens_per_sec_s"] = smooth(df["tokens_per_sec"], smooth_k)
    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tnn", default=None, help="TNN batch training metrics CSV (auto-detected if omitted)")
    parser.add_argument("--torch", default=None, help="PyTorch batch training metrics CSV (auto-detected if omitted)")
    parser.add_argument("--tnn-val", default=None, help="TNN epoch-level validation CSV (auto-detected if omitted)")
    parser.add_argument("--torch-val", default=None, help="PyTorch epoch summary CSV (auto-detected if omitted)")
    parser.add_argument("--out", default=None, help="Output image path (default: plots/gpt2_comparison.png)")
    parser.add_argument("--log-dir", default="logs", help="Directory to search for CSV logs (default: logs)")

    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--double-column", action="store_true")

    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)

    # GPT-2 sequence length (tokens per sample) — used for tokens/s and TFLOPS
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length in tokens (default: 1024)")
    # TNN batch size is not stored in the TNN batch CSV, so it must be supplied
    parser.add_argument("--tnn-batch-size", type=int, default=None,
                        help="Batch size used in TNN run (required for tokens/s and TFLOPS)")
    # FLOPs per token for TFLOPS calculation; default = 6 * 117M (GPT-2 small)
    parser.add_argument("--flops-per-token", type=float, default=6 * 117e6,
                        help="FLOPs per token for TFLOPS estimate (default: 702M, GPT-2 small)")

    args = parser.parse_args()

    log_dir = args.log_dir

    # Auto-detect latest log files when paths are not explicitly provided
    if args.tnn is None:
        args.tnn = find_latest(log_dir, "tnn_*gpt2*batch*.csv")
        if args.tnn:
            print(f"Auto-detected TNN batch CSV: {args.tnn}")
        else:
            raise FileNotFoundError(f"No TNN GPT-2 batch CSV found in '{log_dir}'. Pass --tnn explicitly.")

    if args.torch is None:
        args.torch = find_latest(log_dir, "gpt2*rank0_metrics.csv")
        if args.torch:
            print(f"Auto-detected PyTorch metrics CSV: {args.torch}")
        else:
            raise FileNotFoundError(f"No PyTorch GPT-2 metrics CSV found in '{log_dir}'. Pass --torch explicitly.")

    if args.tnn_val is None:
        args.tnn_val = find_latest(log_dir, "tnn_*gpt2*epoch*.csv")
        if args.tnn_val:
            print(f"Auto-detected TNN val CSV: {args.tnn_val}")

    if args.torch_val is None:
        args.torch_val = find_latest(log_dir, "gpt2*rank0_epoch_summary.csv")
        if args.torch_val:
            print(f"Auto-detected PyTorch epoch summary CSV: {args.torch_val}")

    if args.out is None:
        os.makedirs("plots", exist_ok=True)
        args.out = os.path.join("plots", "gpt2_comparison.png")

    tnn = load_tnn(args.tnn, args.seq_len, args.tnn_batch_size, args.flops_per_token)
    torch_df = load_torch(args.torch, args.seq_len, args.flops_per_token)

    tnn = add_smoothed_columns(tnn, args.smooth)
    torch_df = add_smoothed_columns(torch_df, args.smooth)

    print("TNN points:", len(tnn))
    print("PyTorch points:", len(torch_df))

    tnn_val = None
    torch_val = None
    if args.tnn_val:
        tnn_val = load_tnn_val(args.tnn_val)
        print("TNN validation epochs:", len(tnn_val))
    if args.torch_val:
        torch_val = load_torch_val(args.torch_val)
        print("PyTorch validation epochs:", len(torch_val))

    plot_combined(
        tnn=tnn,
        torch_df=torch_df,
        tnn_val=tnn_val,
        torch_val=torch_val,
        out=args.out,
        double_column=args.double_column,
        xmax=args.xmax,
        ymax=args.ymax,
    )


if __name__ == "__main__":
    main()