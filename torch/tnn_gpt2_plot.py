import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot TNN GPT-2 loss (step)")
    parser.add_argument("--csv", required=True, help="Path to CSV log")
    parser.add_argument("--out", default=None, help="Output image path")
    parser.add_argument("--start_step", type=int, default=None)
    parser.add_argument("--end_step", type=int, default=None)
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # ===== Validate =====
    if "step" not in df.columns or "loss" not in df.columns:
        raise ValueError("CSV must contain 'step' and 'loss' columns")

    # ===== Convert numeric =====
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")

    df = df.dropna(subset=["step", "loss"])
    df = df.sort_values("step")

    # ===== Filter =====
    if args.start_step is not None:
        df = df[df["step"] >= args.start_step]

    if args.end_step is not None:
        df = df[df["step"] <= args.end_step]

    if df.empty:
        raise ValueError("No data after filtering")

    # ===== Smoothing =====
    smooth = max(1, args.smooth)
    df["loss_plot"] = df["loss"].rolling(window=smooth, min_periods=1).mean()

    # ===== Plot =====
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["loss_plot"], linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("TNN GPT-2 Training Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=300)
        print(f"Saved plot to {args.out}")

    plt.show()


if __name__ == "__main__":
    main()