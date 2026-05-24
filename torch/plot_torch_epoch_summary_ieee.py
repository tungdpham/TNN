import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--double-column", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    for col in ["epoch", "train_loss", "val_loss", "val_accuracy_percent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["epoch", "train_loss", "val_loss", "val_accuracy_percent"])
    df = df.sort_values("epoch")

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
    })

    figsize = (7.1, 4.8) if args.double_column else (3.5, 4.4)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="Training Loss")
    axes[0].plot(df["epoch"], df["val_loss"], marker="s", label="Validation Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlim(0, 20)
    axes[0].set_ylim(0, 4)
    axes[0].grid(True, linestyle="--", alpha=0.45)
    axes[0].legend(frameon=True)

    axes[1].plot(df["epoch"], df["val_accuracy_percent"], marker="o", label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlim(0, 20)
    axes[1].set_ylim(0, 70)
    axes[1].grid(True, linestyle="--", alpha=0.45)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=600, bbox_inches="tight")
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()