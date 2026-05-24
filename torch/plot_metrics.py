import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    numeric_cols = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy_pct",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ===== Loss =====
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="Training Loss")
    axes[0].plot(df["epoch"], df["val_loss"], marker="o", label="Validation Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # ===== ONLY Validation Accuracy =====
    axes[1].plot(df["epoch"], df["val_accuracy_pct"], marker="o", label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=300)
        print(f"Saved plot to {args.out}")

    plt.show()


if __name__ == "__main__":
    main()