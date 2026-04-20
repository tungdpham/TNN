#!/usr/bin/env python3
import csv
import os
import sys
import time
import signal
import argparse
import subprocess
from datetime import datetime

RUNNING = True


def handle_signal(signum, frame):
    global RUNNING
    RUNNING = False


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def read_int(path: str) -> int:
    with open(path, "r") as f:
        return int(f.read().strip())


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_net_counters(interface: str):
    base = f"/sys/class/net/{interface}/statistics"
    rx_bytes = read_int(os.path.join(base, "rx_bytes"))
    tx_bytes = read_int(os.path.join(base, "tx_bytes"))
    rx_packets = read_int(os.path.join(base, "rx_packets"))
    tx_packets = read_int(os.path.join(base, "tx_packets"))
    return rx_bytes, tx_bytes, rx_packets, tx_packets


def read_rdma_counters(rdma_dev: str, port: int):
    base = f"/sys/class/infiniband/{rdma_dev}/ports/{port}/counters"
    if not os.path.isdir(base):
        raise FileNotFoundError(f"RDMA counter path not found: {base}")

    rcv_data = read_int(os.path.join(base, "port_rcv_data"))
    xmit_data = read_int(os.path.join(base, "port_xmit_data"))
    rcv_packets = read_int(os.path.join(base, "port_rcv_packets"))
    xmit_packets = read_int(os.path.join(base, "port_xmit_packets"))
    return rcv_data, xmit_data, rcv_packets, xmit_packets


def query_gpu(gpu_index: int):
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        parts = [x.strip() for x in out.split(",")]
        if len(parts) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi output: {out}")
        return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    except Exception as e:
        print(f"[WARN] Failed to query GPU: {e}", file=sys.stderr)
        return -1.0, -1.0, -1.0, -1.0


def main():
    parser = argparse.ArgumentParser(
        description="Monitor GPU and network/RDMA throughput over time."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval (seconds)")
    parser.add_argument("--csv", required=True, help="Output CSV path")

    parser.add_argument("--iface", type=str, default=None, help="Netdev interface, e.g. enp131s0f0np0")
    parser.add_argument("--rdma-dev", type=str, default=None, help="RDMA device, e.g. mlx5_0")
    parser.add_argument("--rdma-port", type=int, default=1, help="RDMA port number")
    parser.add_argument("--append", action="store_true", help="Append to CSV")
    args = parser.parse_args()

    if args.iface is None and args.rdma_dev is None:
        print("[ERROR] You must provide at least one of --iface or --rdma-dev", file=sys.stderr)
        sys.exit(1)

    if args.iface is not None and not os.path.exists(f"/sys/class/net/{args.iface}"):
        print(f"[ERROR] Interface not found: {args.iface}", file=sys.stderr)
        sys.exit(1)

    if args.rdma_dev is not None and not os.path.exists(f"/sys/class/infiniband/{args.rdma_dev}"):
        print(f"[ERROR] RDMA device not found: {args.rdma_dev}", file=sys.stderr)
        sys.exit(1)

    ensure_parent_dir(args.csv)

    fields = [
        "wall_time",
        "elapsed_sec",
        "gpu_index",
        "gpu_util_percent",
        "gpu_mem_util_percent",
        "gpu_mem_used_mb",
        "gpu_mem_total_mb",
    ]

    if args.iface is not None:
        fields += [
            "iface",
            "net_rx_bytes",
            "net_tx_bytes",
            "net_rx_packets",
            "net_tx_packets",
            "net_rx_Bps",
            "net_tx_Bps",
            "net_rx_Gbps",
            "net_tx_Gbps",
        ]

    if args.rdma_dev is not None:
        fields += [
            "rdma_dev",
            "rdma_port",
            "roce_rcv_words",
            "roce_xmit_words",
            "roce_rcv_packets",
            "roce_xmit_packets",
            "roce_rx_Bps",
            "roce_tx_Bps",
            "roce_rx_Gbps",
            "roce_tx_Gbps",
        ]

    prev_net = None
    prev_rdma = None
    start_time = time.time()
    prev_t = start_time

    if args.iface is not None:
        prev_net = read_net_counters(args.iface)

    if args.rdma_dev is not None:
        prev_rdma = read_rdma_counters(args.rdma_dev, args.rdma_port)

    file_exists = os.path.exists(args.csv)
    mode = "a" if args.append else "w"

    with open(args.csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if (not args.append) or (not file_exists):
            writer.writeheader()

        print("[INFO] Start monitoring")
        print(f"[INFO] GPU      : {args.gpu}")
        print(f"[INFO] Interval : {args.interval}s")
        if args.iface:
            print(f"[INFO] NET IF   : {args.iface}")
        if args.rdma_dev:
            print(f"[INFO] RDMA DEV : {args.rdma_dev} port {args.rdma_port}")
        print(f"[INFO] CSV      : {args.csv}")
        print("[INFO] Stop with Ctrl+C")

        while RUNNING:
            time.sleep(args.interval)
            now = time.time()
            dt = now - prev_t
            if dt <= 0:
                continue

            gpu_util, gpu_mem_util, gpu_mem_used, gpu_mem_total = query_gpu(args.gpu)

            row = {
                "wall_time": datetime.now().isoformat(timespec="seconds"),
                "elapsed_sec": round(now - start_time, 6),
                "gpu_index": args.gpu,
                "gpu_util_percent": gpu_util,
                "gpu_mem_util_percent": gpu_mem_util,
                "gpu_mem_used_mb": gpu_mem_used,
                "gpu_mem_total_mb": gpu_mem_total,
            }

            msg_parts = []

            if args.iface is not None:
                net_now = read_net_counters(args.iface)
                rx, tx, rxp, txp = net_now
                prev_rx, prev_tx, prev_rxp, prev_txp = prev_net

                net_rx_Bps = (rx - prev_rx) / dt
                net_tx_Bps = (tx - prev_tx) / dt
                net_rx_Gbps = (net_rx_Bps * 8.0) / 1e9
                net_tx_Gbps = (net_tx_Bps * 8.0) / 1e9

                row.update({
                    "iface": args.iface,
                    "net_rx_bytes": rx,
                    "net_tx_bytes": tx,
                    "net_rx_packets": rxp,
                    "net_tx_packets": txp,
                    "net_rx_Bps": round(net_rx_Bps, 6),
                    "net_tx_Bps": round(net_tx_Bps, 6),
                    "net_rx_Gbps": round(net_rx_Gbps, 6),
                    "net_tx_Gbps": round(net_tx_Gbps, 6),
                })

                msg_parts.append(f"NET_RX={net_rx_Gbps:.6f} Gbps")
                msg_parts.append(f"NET_TX={net_tx_Gbps:.6f} Gbps")
                prev_net = net_now

            if args.rdma_dev is not None:
                rdma_now = read_rdma_counters(args.rdma_dev, args.rdma_port)
                rcv_words, xmit_words, rcv_pkts, xmit_pkts = rdma_now
                prev_rcv_words, prev_xmit_words, prev_rcv_pkts, prev_xmit_pkts = prev_rdma

                # IB/RDMA counters are commonly in 32-bit words
                roce_rx_bytes = (rcv_words - prev_rcv_words) * 4
                roce_tx_bytes = (xmit_words - prev_xmit_words) * 4

                roce_rx_Bps = roce_rx_bytes / dt
                roce_tx_Bps = roce_tx_bytes / dt
                roce_rx_Gbps = (roce_rx_Bps * 8.0) / 1e9
                roce_tx_Gbps = (roce_tx_Bps * 8.0) / 1e9

                row.update({
                    "rdma_dev": args.rdma_dev,
                    "rdma_port": args.rdma_port,
                    "roce_rcv_words": rcv_words,
                    "roce_xmit_words": xmit_words,
                    "roce_rcv_packets": rcv_pkts,
                    "roce_xmit_packets": xmit_pkts,
                    "roce_rx_Bps": round(roce_rx_Bps, 6),
                    "roce_tx_Bps": round(roce_tx_Bps, 6),
                    "roce_rx_Gbps": round(roce_rx_Gbps, 6),
                    "roce_tx_Gbps": round(roce_tx_Gbps, 6),
                })

                msg_parts.append(f"ROCE_RX={roce_rx_Gbps:.6f} Gbps")
                msg_parts.append(f"ROCE_TX={roce_tx_Gbps:.6f} Gbps")
                prev_rdma = rdma_now

            msg_parts.append(f"GPU={gpu_util:.1f}%")
            msg_parts.append(f"GPU_MEM={gpu_mem_used:.0f}/{gpu_mem_total:.0f} MB")

            writer.writerow(row)
            f.flush()

            print(f"[{row['wall_time']}] " + " | ".join(msg_parts))

            prev_t = now

    print("[INFO] Monitoring stopped.")


if __name__ == "__main__":
    main()