#!/usr/bin/env python3
# plot_cpu_three.py
# Vẽ %CPU theo thời gian cho 3 container từ các CSV trong ./logs
# Usage:
#   python3 plot_cpu_three.py --logs ./logs --out cpu_usage.png
#   python3 plot_cpu_three.py --logs ./logs --out cpu_usage_0_5.png --tmin 0 --tmax 5
#   (tùy chọn) --smooth 3  # trung bình trượt cửa sổ 3 điểm

import os, glob, csv, argparse
import matplotlib.pyplot as plt

def find_latest_csv(logdir: str, prefix: str):
    pattern = os.path.join(logdir, f"{prefix}_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Không thấy file CSV cho prefix '{prefix}' trong {logdir}")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_series(path: str):
    t, y, tag = [], [], None
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row.get("t_sec", "0")))
            y.append(float(row.get("cpu_percent", "0")))
            if tag is None:
                tag = row.get("tag", None)
    if tag is None:
        base = os.path.basename(path)
        tag = base.split("_")[0]
    return tag, t, y

def moving_avg(vals, k):
    if k <= 1: return vals
    out = []
    s = 0.0
    from collections import deque
    q = deque()
    for v in vals:
        q.append(v); s += v
        if len(q) > k:
            s -= q.popleft()
        out.append(s / len(q))
    return out

def clip_range(t, y, tmin, tmax):
    if tmin is None and tmax is None:
        return t, y
    tt, yy = [], []
    for ti, yi in zip(t, y):
        if tmin is not None and ti < tmin:
            continue
        if tmax is not None and ti > tmax:
            continue
        tt.append(ti); yy.append(yi)
    return tt, yy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=str, default="./logs", help="Thư mục chứa các CSV")
    ap.add_argument("--out", type=str, default="cpu_usage.png", help="Tên file ảnh đầu ra")
    ap.add_argument("--smooth", type=int, default=1, help="Cửa sổ trung bình trượt (>=1, 1 là không làm mượt)")
    ap.add_argument("--tmin", type=float, default=None, help="Giới hạn dưới thời gian (s)")
    ap.add_argument("--tmax", type=float, default=None, help="Giới hạn trên thời gian (s)")
    args = ap.parse_args()

    prefixes = ["coordinator", "worker-8001", "worker-8002"]
    series = []
    picked_files = []

    for p in prefixes:
        f = find_latest_csv(args.logs, p)
        picked_files.append(f)
        tag, t, y = load_series(f)
        # lọc theo khoảng thời gian nếu có
        t, y = clip_range(t, y, args.tmin, args.tmax)
        # nếu sau khi cắt rỗng thì bỏ qua
        if not t:
            print(f"[Cảnh báo] Dãy '{tag}' trống sau khi áp dụng tmin/tmax. Bỏ qua.")
            continue
        # làm mượt nếu cần
        if args.smooth > 1:
            y = moving_avg(y, args.smooth)
        series.append((tag, t, y, f))

    if not series:
        raise SystemExit("Không có dữ liệu hợp lệ để vẽ (có thể tmin/tmax quá chặt).")

    plt.figure(figsize=(18, 10))
    for tag, t, y, f in series:
        plt.plot(t, y, label=f"{tag}  (from: {os.path.basename(f)})")

    plt.xlabel("Time (s)")
    plt.ylabel("CPU Utilization (%)")
    title = "CPU usage per container"
    if args.tmin is not None or args.tmax is not None:
        title += f"  [window: {args.tmin if args.tmin is not None else '-'}–{args.tmax if args.tmax is not None else '-'} s]"
    plt.title(title)
    plt.grid(True, alpha=0.3, linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # cố định khung nhìn theo tmin/tmax nếu có (không bắt buộc, nhưng trực quan)
    if args.tmin is not None or args.tmax is not None:
        xmin = args.tmin if args.tmin is not None else min(min(t for _, t, _, _ in series))
        xmax = args.tmax if args.tmax is not None else max(max(t for _, t, _, _ in series))
        if xmin < xmax:
            plt.xlim(xmin, xmax)

    plt.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")
    print("Sources:")
    for f in picked_files:
        print(" -", f)

if __name__ == "__main__":
    main()
