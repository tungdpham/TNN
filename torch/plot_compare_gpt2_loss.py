#!/usr/bin/env python3
import argparse
import os
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


def setup_style():
    plt.style.use("default")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.7,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.45,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def plot_one(tnn, torch, out, kind, smooth_k, double_column, xmax, ymax):
    setup_style()

    tnn_y = f"{kind}_loss_s"
    torch_y = f"{kind}_loss_s"

    figsize = (7.1, 2.8) if double_column else (3.5, 2.6)

    plt.figure(figsize=figsize)

    label_name = "Batch Loss" if kind == "batch" else "Avg Loss"

    plt.plot(
        tnn["global_step"],
        tnn[tnn_y],
        label=f"TNN {label_name}"
    )

    plt.plot(
        torch["global_step"],
        torch[torch_y],
        label=f"PyTorch {label_name}"
    )

    plt.xlabel("Step")
    plt.ylabel(label_name)

    if xmax is not None:
        plt.xlim(0, xmax)

    if ymax is not None:
        plt.ylim(0, ymax)

    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="--", alpha=0.35)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


def plot_combined(tnn, torch, out, plot, double_column, xmax, ymax):
    setup_style()

    figsize = (7.1, 2.8) if double_column else (3.5, 2.6)

    plt.figure(figsize=figsize)

    if plot in ["batch", "both"]:
        plt.plot(
            tnn["global_step"],
            tnn["batch_loss_s"],
            linestyle="-",
            label="TNN Batch Loss"
        )

        plt.plot(
            torch["global_step"],
            torch["batch_loss_s"],
            linestyle="-",
            label="PyTorch Batch Loss"
        )

    if plot in ["avg", "both"]:
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

    if plot == "batch":
        plt.ylabel("Batch Loss")
    elif plot == "avg":
        plt.ylabel("Avg Loss")
    else:
        plt.ylabel("Loss")

    if xmax is not None:
        plt.xlim(0, xmax)

    if ymax is not None:
        plt.ylim(0, ymax)

    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="--", alpha=0.35)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


def add_smoothed_columns(df, smooth_k):
    df = df.copy()
    df["batch_loss_s"] = smooth(df["batch_loss"], smooth_k)
    df["avg_loss_s"] = smooth(df["avg_loss"], smooth_k)
    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tnn", required=True)
    parser.add_argument("--torch", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--smooth", type=int, default=1)

    parser.add_argument(
        "--plot",
        choices=["batch", "avg", "both"],
        default="both"
    )

    parser.add_argument(
        "--mode",
        choices=["single", "separate"],
        default="single"
    )

    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--double-column", action="store_true")

    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)

    args = parser.parse_args()

    tnn = load_tnn(args.tnn, all_runs=args.all_runs)
    torch = load_torch(args.torch, all_runs=args.all_runs)

    tnn = add_smoothed_columns(tnn, args.smooth)
    torch = add_smoothed_columns(torch, args.smooth)

    print("TNN points:", len(tnn))
    print("PyTorch points:", len(torch))

    if args.mode == "single":
        plot_combined(
            tnn=tnn,
            torch=torch,
            out=args.out,
            plot=args.plot,
            double_column=args.double_column,
            xmax=args.xmax,
            ymax=args.ymax,
        )

    else:
        root, ext = os.path.splitext(args.out)

        if ext == "":
            ext = ".png"

        batch_out = root + "_batch_loss" + ext
        avg_out = root + "_avg_loss" + ext

        plot_one(
            tnn=tnn,
            torch=torch,
            out=batch_out,
            kind="batch",
            smooth_k=args.smooth,
            double_column=args.double_column,
            xmax=args.xmax,
            ymax=args.ymax,
        )

        plot_one(
            tnn=tnn,
            torch=torch,
            out=avg_out,
            kind="avg",
            smooth_k=args.smooth,
            double_column=args.double_column,
            xmax=args.xmax,
            ymax=args.ymax,
        )


if __name__ == "__main__":
    main()