#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


GROUPS = {
    "roce": ["roce_rx_Gbps", "roce_tx_Gbps"],
    "gpu": ["gpu_util_percent"],
    "tcp": ["net_rx_Gbps", "net_tx_Gbps"],
    "mem": ["gpu_mem_used_mb", "gpu_mem_total_mb"],
}


def parse_groups(groups_arg):
    groups = []
    for g in groups_arg.split(","):
        g = g.strip()
        if g:
            groups.append(g)
    return groups


def main():
    parser = argparse.ArgumentParser(description="Plot grouped monitor metrics.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="logs/monitor_grouped.png")
    parser.add_argument("--time-col", default="elapsed_sec")
    parser.add_argument("--groups", default="roce,gpu,tcp,mem",
                        help="Groups to plot: roce,gpu,tcp,mem")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--title", default="Network / RoCE / GPU Monitor")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)

    if df.empty:
        raise ValueError("CSV is empty")

    if args.time_col not in df.columns:
        raise ValueError(f"Missing time column: {args.time_col}")

    start = args.start
    if args.end is not None:
        end = args.end
    elif args.duration is not None:
        end = start + args.duration
    else:
        end = df[args.time_col].max()

    df = df[(df[args.time_col] >= start) & (df[args.time_col] <= end)]

    if df.empty:
        raise ValueError(f"No data in selected range: start={start}, end={end}")

    selected_groups = parse_groups(args.groups)

    plot_groups = []
    for group in selected_groups:
        if group not in GROUPS:
            print(f"[WARN] Unknown group skipped: {group}")
            continue

        cols = [c for c in GROUPS[group] if c in df.columns]
        if not cols:
            print(f"[WARN] Group skipped because columns not found: {group}")
            continue

        plot_groups.append((group, cols))

    if not plot_groups:
        print("[ERROR] No valid groups to plot.")
        print("[INFO] Available columns:")
        for c in df.columns:
            print(f"  - {c}")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    x = df[args.time_col]
    n = len(plot_groups)

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(14, max(3.2 * n, 6)),
        sharex=True,
    )

    if n == 1:
        axes = [axes]

    fig.suptitle(f"{args.title} | {start:.2f}s → {end:.2f}s")

    for ax, (group, cols) in zip(axes, plot_groups):
        for col in cols:
            ax.plot(x, df[col], linewidth=2, label=col)

        if group == "roce":
            ax.set_title("RoCE Throughput")
            ax.set_ylabel("Gbps")
        elif group == "tcp":
            ax.set_title("TCP / Netdev Throughput")
            ax.set_ylabel("Gbps")
        elif group == "gpu":
            ax.set_title("GPU Utilization")
            ax.set_ylabel("Percent (%)")
        elif group == "mem":
            ax.set_title("GPU Memory Usage")
            ax.set_ylabel("MB")
        else:
            ax.set_title(group)
            ax.set_ylabel("Value")

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig(args.out, dpi=160, bbox_inches="tight")

    print(f"[OK] Saved plot: {args.out}")
    print("[OK] Plotted groups:")
    for group, cols in plot_groups:
        print(f"  - {group}: {cols}")
    print(f"[OK] Time range: {start:.2f}s -> {end:.2f}s")

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()