"""
Plot convergence curves comparing TNN vs PyTorch training runs.

Usage:
    python visualizers/plot_convergence.py --experiment cifar10_resnet9
    python visualizers/plot_convergence.py --experiment cifar100_wrn16_8
    python visualizers/plot_convergence.py --experiment tiny_imagenet_resnet50
    python visualizers/plot_convergence.py --experiment cifar10_resnet9 --log-dir logs --out output_images/convergence_cifar10.png

The script searches logs/ for the most recent epoch CSV matching each run:
  TNN:     {experiment}_epoch_{timestamp}.csv
  PyTorch: torch_{experiment}_epoch_{timestamp}.csv
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_csv(log_dir: str, pattern: str) -> str | None:
    """Return the path of the most-recently-timestamped file matching glob pattern."""
    matches = sorted(glob.glob(os.path.join(log_dir, pattern)))
    return matches[-1] if matches else None


def read_epoch_csv(path: str) -> dict[str, list]:
    """
    Read an epoch CSV with columns:
      epoch, train_loss, train_accuracy_pct, val_loss, val_accuracy_pct
    Returns a dict of lists keyed by column name.
    """
    data: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy_pct": [],
        "val_loss": [],
        "val_accuracy_pct": [],
    }
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["train_loss"].append(float(row["train_loss"]))
            data["train_accuracy_pct"].append(float(row["train_accuracy_pct"]))
            data["val_loss"].append(float(row["val_loss"]))
            data["val_accuracy_pct"].append(float(row["val_accuracy_pct"]))
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot TNN vs PyTorch convergence curves.")
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        choices=["cifar10_resnet9", "cifar100_wrn16_8", "tiny_imagenet_resnet50"],
        help="Experiment name (used to find CSV files).",
    )
    parser.add_argument(
        "--log-dir", default="logs",
        help="Directory containing CSV log files (default: logs).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output image path. Defaults to output_images/convergence_{experiment}.png",
    )
    parser.add_argument(
        "--tnn-label", default="TNN",
        help="Legend label for the TNN run (default: TNN).",
    )
    parser.add_argument(
        "--torch-label", default="PyTorch",
        help="Legend label for the PyTorch run (default: PyTorch).",
    )
    args = parser.parse_args()

    experiment = args.experiment
    log_dir    = args.log_dir
    out_path   = args.out or os.path.join("output_images", f"convergence_{experiment}.png")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # ---- Find epoch CSVs ----
    tnn_path   = find_latest_csv(log_dir, f"{experiment}_epoch_*.csv")
    torch_path = find_latest_csv(log_dir, f"torch_{experiment}_epoch_*.csv")

    if tnn_path is None and torch_path is None:
        print(f"ERROR: No epoch CSV files found for experiment '{experiment}' in '{log_dir}'.")
        sys.exit(1)

    print(f"Experiment : {experiment}")
    print(f"TNN log    : {tnn_path   or '(none found)'}")
    print(f"PyTorch log: {torch_path or '(none found)'}")

    tnn_data   = read_epoch_csv(tnn_path)   if tnn_path   else None
    torch_data = read_epoch_csv(torch_path) if torch_path else None

    # ---- Plot ----
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Convergence: {experiment.replace('_', ' ').title()}", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    ax_train_loss = fig.add_subplot(gs[0, 0])
    ax_val_loss   = fig.add_subplot(gs[0, 1])
    ax_train_acc  = fig.add_subplot(gs[1, 0])
    ax_val_acc    = fig.add_subplot(gs[1, 1])

    COLOR_TNN   = "#1f77b4"   # matplotlib blue
    COLOR_TORCH = "#ff7f0e"   # matplotlib orange

    def _plot(ax, data, col, color, label, linestyle="-"):
        ax.plot(data["epoch"], data[col], color=color, label=label,
                linestyle=linestyle, linewidth=1.8, marker="o", markersize=3)

    for ax, col, title, ylabel in [
        (ax_train_loss, "train_loss",         "Training Loss",          "Loss"),
        (ax_val_loss,   "val_loss",            "Validation Loss",        "Loss"),
        (ax_train_acc,  "train_accuracy_pct",  "Training Accuracy",      "Accuracy (%)"),
        (ax_val_acc,    "val_accuracy_pct",    "Validation Accuracy",    "Accuracy (%)"),
    ]:
        if tnn_data is not None:
            _plot(ax, tnn_data, col, COLOR_TNN, args.tnn_label)
        if torch_data is not None:
            _plot(ax, torch_data, col, COLOR_TORCH, args.torch_label, linestyle="--")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
