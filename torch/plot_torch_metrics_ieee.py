import argparse
import pandas as pd
import matplotlib.pyplot as plt


COLS = [
    "timestamp",
    "phase",
    "epoch",
    "step",
    "batch_size",
    "micro_bs",
    "lr",
    "loss_sum",
    "samples",
    "avg_loss",
    "correct",
    "accuracy_percent",
    "epoch_time_sec",
]


def read_metrics_csv(path):
    return pd.read_csv(path, skiprows=1, names=COLS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to *_rank0_metrics.csv")
    parser.add_argument("--out", required=True, help="Output figure path")
    parser.add_argument("--title", default="")
    parser.add_argument("--double-column", action="store_true")
    args = parser.parse_args()

    df = read_metrics_csv(args.csv)

    for col in [
        "timestamp", "epoch", "step", "batch_size", "micro_bs", "lr",
        "loss_sum", "samples", "avg_loss", "correct",
        "accuracy_percent", "epoch_time_sec",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    train_df = (
        df[df["phase"] == "train_batch"]
        .dropna(subset=["epoch", "step", "avg_loss"])
        .sort_values(["timestamp", "epoch", "step"])
        .groupby("epoch", as_index=False)
        .tail(1)
        .sort_values("epoch")
    )

    val_df = (
        df[df["phase"] == "epoch_summary"]
        .dropna(subset=["epoch", "avg_loss", "accuracy_percent"])
        .sort_values(["timestamp", "epoch"])
        .groupby("epoch", as_index=False)
        .tail(1)
        .sort_values("epoch")
    )

    if train_df.empty:
        raise ValueError("No train_batch rows found")

    if val_df.empty:
        raise ValueError("No epoch_summary rows found")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "savefig.dpi": 600,
    })

    figsize = (7.1, 4.8) if args.double_column else (3.5, 4.4)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # ===== Loss subplot =====
    axes[0].plot(
        train_df["epoch"],
        train_df["avg_loss"],
        marker="o",
        label="Training Loss",
    )
    axes[0].plot(
        val_df["epoch"],
        val_df["avg_loss"],
        marker="s",
        label="Validation Loss",
    )

    axes[0].set_ylabel("Loss")
    axes[0].set_xlim(0, 20)
    axes[0].set_ylim(0, 4)
    axes[0].set_yticks([0, 1, 2, 3, 4])
    axes[0].grid(True, linestyle="--", alpha=0.45)
    axes[0].legend(frameon=True)

    # ===== Validation accuracy subplot =====
    axes[1].plot(
        val_df["epoch"],
        val_df["accuracy_percent"],
        marker="o",
        label="Validation Accuracy",
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlim(0, 20)
    axes[1].set_ylim(0, 70)
    axes[1].set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    axes[1].grid(True, linestyle="--", alpha=0.45)
    axes[1].legend(frameon=True)

    if args.title:
        fig.suptitle(args.title, y=0.995)

    plt.tight_layout()

    if args.title:
        plt.subplots_adjust(top=0.92)

    plt.savefig(args.out, bbox_inches="tight", dpi=600)
    print(f"Saved figure to {args.out}")

    plt.show()


if __name__ == "__main__":
    main()