import argparse
import pandas as pd
import matplotlib.pyplot as plt


def smooth(series, k):
    if k <= 1:
        return series
    return series.rolling(window=k, min_periods=1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--double-column", action="store_true")
    parser.add_argument("--xmin", type=int, default=0)
    parser.add_argument("--xmax", type=int, default=20000)
    parser.add_argument("--ymin", type=float, default=0)
    parser.add_argument("--ymax", type=float, default=11)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["tnn_loss"] = pd.to_numeric(df["tnn_loss"], errors="coerce")
    df["torch_loss"] = pd.to_numeric(df["torch_loss"], errors="coerce")
    df = df.dropna(subset=["step"]).sort_values("step")

    df["tnn_plot"] = smooth(df["tnn_loss"], args.smooth)
    df["torch_plot"] = smooth(df["torch_loss"], args.smooth)

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
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
    })

    figsize = (7.1, 3.0) if args.double_column else (3.5, 2.6)

    plt.figure(figsize=figsize)
    plt.plot(df["step"], df["tnn_plot"], label="TNN", linestyle="-")
    plt.plot(df["step"], df["torch_plot"], label="PyTorch", linestyle="--")

    plt.xlabel("Training Step")
    plt.ylabel("Training Loss")
    plt.xlim(args.xmin, args.xmax)
    plt.ylim(args.ymin, args.ymax)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig(args.out, dpi=600, bbox_inches="tight")
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()