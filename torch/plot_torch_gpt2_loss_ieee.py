#!/usr/bin/env python3
import argparse
import csv
import matplotlib.pyplot as plt


def moving_average(values, window):
    if window <= 1:
        return values

    out = []

    for i in range(len(values)):
        s = max(0, i - window + 1)
        out.append(sum(values[s:i + 1]) / (i - s + 1))

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--smooth", type=int, default=1)

    parser.add_argument(
        "--plot",
        choices=["batch", "avg", "both"],
        default="both",
        help="What to plot"
    )

    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--double-column", action="store_true")

    args = parser.parse_args()

    rows = []

    with open(args.csv, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:

            if r.get("phase") != "train_batch":
                continue

            try:
                step = int(float(r["step"]))

                batch_loss = float(r["batch_loss"])
                avg_loss = float(r["avg_loss"])

            except Exception:
                continue

            rows.append({
                "step": step,
                "batch_loss": batch_loss,
                "avg_loss": avg_loss,
            })

    if not rows:
        raise RuntimeError("No valid train_batch rows found")

    # ==========================================================
    # Split appended runs
    # ==========================================================

    runs = []

    cur = []

    prev_step = None

    for row in rows:

        step = row["step"]

        if prev_step is not None and step <= prev_step and cur:
            runs.append(cur)
            cur = []

        cur.append(row)

        prev_step = step

    if cur:
        runs.append(cur)

    selected_runs = runs if args.all_runs else [runs[-1]]

    print(f"Detected runs: {len(runs)}")

    # ==========================================================
    # Plot style
    # ==========================================================

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

    # ==========================================================
    # Plot data
    # ==========================================================

    for i, run in enumerate(selected_runs):

        steps = [x["step"] for x in run]

        batch_losses = moving_average(
            [x["batch_loss"] for x in run],
            args.smooth
        )

        avg_losses = moving_average(
            [x["avg_loss"] for x in run],
            args.smooth
        )

        if len(set(steps)) <= 1:
            steps = list(range(len(batch_losses)))

        prefix = f"Run {i + 1} " if args.all_runs else ""

        if args.plot in ["batch", "both"]:
            plt.plot(
                steps,
                batch_losses,
                label=prefix + "Batch Loss"
            )

        if args.plot in ["avg", "both"]:
            plt.plot(
                steps,
                avg_losses,
                linestyle="--",
                label=prefix + "Avg Loss"
            )

    # ==========================================================
    # Labels
    # ==========================================================

    plt.xlabel("Step")

    if args.plot == "batch":
        plt.ylabel("Batch Loss")

    elif args.plot == "avg":
        plt.ylabel("Avg Loss")

    else:
        plt.ylabel("Loss")

    plt.grid(True, linestyle="--", alpha=0.45)

    plt.legend(frameon=True)

    plt.tight_layout()

    plt.savefig(args.out, dpi=600, bbox_inches="tight")

    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()