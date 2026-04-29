import argparse
import pandas as pd
import matplotlib.pyplot as plt


def smooth(s, k):
    if k <= 1:
        return s
    return s.rolling(k, min_periods=1).mean()


# ===== TNN =====
def load_tnn(path):
    df = pd.read_csv(path)

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")

    df = df.dropna(subset=["step", "loss"])
    return df.sort_values("step")


# ===== PyTorch (convert avg_loss -> batch loss) =====
def load_torch(path):
    COLS = [
        "timestamp","phase","epoch","epoch_step","step",
        "micro_bs","lr","loss_sum","samples","avg_loss",
        "correct","accuracy_percent","epoch_time_sec"
    ]

    df = pd.read_csv(path, names=COLS, header=0, engine="python", on_bad_lines="skip")

    df = df[df["phase"] == "train_batch"].copy()

    df["loss_sum"] = pd.to_numeric(df["loss_sum"], errors="coerce")
    df["samples"] = pd.to_numeric(df["samples"], errors="coerce")

    df = df.dropna(subset=["loss_sum", "samples"])
    df = df.sort_values(["epoch", "epoch_step"])

    # ===== compute batch loss =====
    df["loss_diff"] = df["loss_sum"].diff()
    df["sample_diff"] = df["samples"].diff()

    df["batch_loss"] = df["loss_diff"] / df["sample_diff"]

    df = df.dropna(subset=["batch_loss"])

    # tạo global step
    df["step"] = range(1, len(df) + 1)

    return df[["step", "batch_loss"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tnn", required=True)
    parser.add_argument("--torch", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--double-column", action="store_true")
    args = parser.parse_args()

    tnn = load_tnn(args.tnn)
    torch = load_torch(args.torch)

    tnn["loss"] = smooth(tnn["loss"], args.smooth)
    torch["batch_loss"] = smooth(torch["batch_loss"], args.smooth)

    # ===== IEEE style =====
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "lines.linewidth": 1.8,
    })

    figsize = (7.1, 3.0) if args.double_column else (3.5, 2.6)
    plt.figure(figsize=figsize)

    plt.plot(tnn["step"], tnn["loss"], label="TNN")
    plt.plot(torch["step"], torch["batch_loss"], label="PyTorch")

    plt.xlabel("Step")
    plt.ylabel("Training Loss")

    plt.xlim(0, 20000)
    plt.ylim(0, 11)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.out, dpi=600, bbox_inches="tight")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()