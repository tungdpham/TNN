import os
import argparse
import numpy as np


def sample_random_chunks(src_path, dst_path, pct, chunk_size, dtype=np.uint16, seed=42):
    rng = np.random.default_rng(seed)

    src = np.memmap(src_path, dtype=dtype, mode="r")
    total_tokens = len(src)

    target_tokens = int(total_tokens * pct / 100.0)

    num_chunks = target_tokens // chunk_size

    print(f"[INFO] Total tokens: {total_tokens:,}")
    print(f"[INFO] Target tokens: {target_tokens:,} ({pct}%)")
    print(f"[INFO] Chunk size: {chunk_size}")
    print(f"[INFO] Number of chunks: {num_chunks}")

    dst = np.memmap(dst_path, dtype=dtype, mode="w+", shape=(num_chunks * chunk_size,))

    used_ranges = []

    idx = 0
    attempts = 0

    while idx < num_chunks:
        start = int(rng.integers(0, total_tokens - chunk_size))

        # tránh overlap (không bắt buộc nhưng tốt)
        overlap = False
        for s, e in used_ranges:
            if not (start + chunk_size <= s or start >= e):
                overlap = True
                break

        if overlap:
            attempts += 1
            if attempts > 1000:
                # fallback nếu khó tìm
                overlap = False
            else:
                continue

        end = start + chunk_size

        dst[idx * chunk_size:(idx + 1) * chunk_size] = src[start:end]

        used_ranges.append((start, end))
        idx += 1

        if idx % 100 == 0:
            print(f"[INFO] Sampled {idx}/{num_chunks} chunks")

    dst.flush()
    print(f"[OK] wrote {dst_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--pct", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)

    sample_random_chunks(
        args.src,
        args.dst,
        pct=args.pct,
        chunk_size=args.chunk_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()