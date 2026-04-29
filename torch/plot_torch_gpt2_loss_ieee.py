import argparse
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
    parser.add_argument("--double-column", action="store_true")
    args = parser.parse_args()

    rows = []

    with open(args.csv, "r", encoding="utf-8", errors="ignore") as f:
        next(f)

        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue

            try:
                phase = parts[1]
                step = float(parts[4])      # column 5
                loss = float(parts[9])      # column 10
            except Exception:
                continue

            if phase != "train_batch":
                continue

            rows.append((step, loss))

    if not rows:
        raise RuntimeError("No valid train_batch rows found")

    # ===== split appended runs =====
    runs = []
    cur = []

    prev_step = None
    prev_loss = None

    for step, loss in rows:
        new_run = False

        if prev_step is not None:
            # step reset hoặc loss nhảy lên mạnh => run mới
            if step < prev_step or loss > prev_loss + 1.0:
                new_run = True

        if new_run and cur:
            runs.append(cur)
            cur = []

        cur.append((step, loss))
        prev_step = step
        prev_loss = loss

    if cur:
        runs.append(cur)

    # lấy run cuối cùng
    selected = runs[-1]

    steps = [x[0] for x in selected]
    losses = [x[1] for x in selected]

    # nếu step bị lỗi/constant thì dùng index
    if len(set(steps)) <= 1:
        steps = list(range(1, len(losses) + 1))

    losses = moving_average(losses, args.smooth)

    print(f"Detected runs: {len(runs)}")
    print(f"Plotting last run with {len(losses)} points")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
    })

    figsize = (7.1, 3.0) if args.double_column else (3.5, 2.6)

    plt.figure(figsize=figsize)
    plt.plot(steps, losses, label="Training Avg Loss")

    plt.xlabel("Step")
    plt.ylabel("Avg Loss")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig(args.out, dpi=600, bbox_inches="tight")
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()