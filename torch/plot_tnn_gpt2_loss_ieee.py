#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def smooth(s, k):
    if k <= 1:
        return s
    return s.rolling(k, min_periods=1).mean()


def split_last_run(df, step_col="step"):
    df = df.copy().reset_index(drop=True)

    if df.empty:
        return df

    run_id = 0
    run_ids = []

    prev_step = None

    for step in df[step_col]:
        if prev_step is not None and step <= prev_step:
            run_id += 1

        run_ids.append(run_id)
        prev_step = step

    df["_run_id"] = run_ids

    return df[df["_run_id"] == df["_run_id"].max()].copy()


# ===== TNN =====
def load_tnn(path, all_runs=False):
    df = pd.read_csv(path)

    for col in ["step", "batch_loss", "avg_loss"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["step", "batch_loss", "avg_loss"])
    df = df.sort_index()

    if not all_runs:
        df = split_last_run(df, "step")

    df["global_step"] = range(1, len(df) + 1)

    return df[["global_step", "step", "batch_loss", "avg_loss"]]


# ===== PyTorch =====
def load_torch(path, all_runs=False):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    df = df[df["phase"] == "train_batch"].copy()

    for col in ["step", "batch_loss", "avg_loss"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["step", "batch_loss", "avg_loss"])
    df = df.sort_index()

    if not all_runs:
        df = split_last_run(df, "step")

    df["global_step"] = range(1, len(df) + 1)

    return df[["global_step", "step", "batch_loss", "avg_loss"]]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tnn", required=True)
    parser.add_argument("--torch", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--smooth", type=int, default=1)

    parser.add_argument(
        "--plot",
        choices=["batch", "avg", "both"],
        default="both",
        help="batch = batch_loss, avg = avg_loss/train loss, both = both curves"
    )

    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--double-column", action="store_true")

    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)

    args = parser.parse_args()

    tnn = load_tnn(args.tnn, all_runs=args.all_runs)
    torch = load_torch(args.torch, all_runs=args.all_runs)

    tnn["batch_loss_s"] = smooth(tnn["batch_loss"], args.smooth)
    tnn["avg_loss_s"] = smooth(tnn["avg_loss"], args.smooth)

    torch["batch_loss_s"] = smooth(torch["batch_loss"], args.smooth)
    torch["avg_loss_s"] = smooth(torch["avg_loss"], args.smooth)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.6,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
    })

    figsize = (7.1, 3.0) if args.double_column else (3.5, 2.6)

    plt.figure(figsize=figsize)

    if args.plot in ["batch", "both"]:
        plt.plot(
            tnn["global_step"],
            tnn["batch_loss_s"],
            label="TNN Batch Loss"
        )

        plt.plot(
            torch["global_step"],
            torch["batch_loss_s"],
            label="PyTorch Batch Loss"
        )

    if args.plot in ["avg", "both"]:
        plt.plot(
            tnn["global_step"],
            tnn["avg_loss_s"],
            linestyle="--",
            label="TNN Avg Loss"
        )

        plt.plot(
            torch["global_step"],
            torch["avg_loss_s"],
            linestyle="--",
            label="PyTorch Avg Loss"
        )

    plt.xlabel("Step")

    if args.plot == "batch":
        plt.ylabel("Batch Loss")
    elif args.plot == "avg":
        plt.ylabel("Avg Loss")
    else:
        plt.ylabel("Loss")

    if args.xmax is not None:
        plt.xlim(0, args.xmax)

    if args.ymax is not None:
        plt.ylim(0, args.ymax)

    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig(args.out, dpi=600, bbox_inches="tight")

    print("Saved:", args.out)
    print("TNN points:", len(tnn))
    print("PyTorch points:", len(torch))


if __name__ == "__main__":
    main()