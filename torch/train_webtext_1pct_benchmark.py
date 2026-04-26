import os
import math
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel


def detect_dtype(device: torch.device):
    return torch.float32


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m}m {s}s"


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: torch.device):
    n = len(data)
    if n <= block_size + 1:
        raise ValueError(f"Dataset too small: len={n}, block_size={block_size}")

    ix = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(np.array(data[i:i + block_size], dtype=np.int64))
        for i in ix.tolist()
    ])
    y = torch.stack([
        torch.from_numpy(np.array(data[i + 1:i + 1 + block_size], dtype=np.int64))
        for i in ix.tolist()
    ])

    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, eval_iters, device, amp_dtype):
    model.eval()
    losses = []
    token_accs = []

    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            out = model(input_ids=x, labels=y)
            loss = out.loss
            logits = out.logits

        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()

        losses.append(loss.item())
        token_accs.append(acc)

    model.train()
    return float(np.mean(losses)), float(np.mean(token_accs))


@dataclass
class ModelSpec:
    n_layer: int
    n_head: int
    n_embd: int


MODEL_SPECS = {
    "gpt2_tiny": ModelSpec(n_layer=6, n_head=6, n_embd=384),
    "gpt2_small": ModelSpec(n_layer=12, n_head=12, n_embd=768),
}


def build_model(model_name: str, block_size: int, vocab_size: int = 50257):
    spec = MODEL_SPECS[model_name]
    cfg = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bos_token_id=50256,
        eos_token_id=50256,
        )
    return GPT2LMHeadModel(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/open-web-text")
    parser.add_argument("--model-name", type=str, default="gpt2_small", choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Total optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="Steps ignored in throughput averaging")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-on-train", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out-dir", type=str, default="out_webtext_1pct_benchmark")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = detect_dtype(device)

    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "val.bin")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}")
    if not args.eval_on_train and not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing {val_path}. Use --eval-on-train if needed.")

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    eval_data = train_data if args.eval_on_train else np.memmap(val_path, dtype=np.uint16, mode="r")

    total_tokens = len(train_data)
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    seqs_per_step = args.batch_size * args.grad_accum

    steps_per_epoch = total_tokens / tokens_per_step

    print(f"Using device           : {device}")
    print(f"AMP dtype              : {amp_dtype}")
    print(f"Dataset path           : {args.data_dir}")
    print(f"Train tokens           : {total_tokens:,}")
    print(f"Eval tokens            : {len(eval_data):,}")
    print(f"Model                  : {args.model_name}")
    print(f"Block size             : {args.block_size}")
    print(f"Batch size             : {args.batch_size}")
    print(f"Grad accum             : {args.grad_accum}")
    print(f"Tokens per step        : {tokens_per_step:,}")
    print(f"Sequences per step     : {seqs_per_step:,}")
    print(f"Approx steps / epoch   : {steps_per_epoch:.2f}")
    print(f"Benchmark max steps    : {args.max_steps}")
    print(f"Warmup steps excluded  : {args.warmup_steps}")

    model = build_model(args.model_name, args.block_size)
    model.to(device)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    model.train()

    step_times = []
    running_loss = 0.0
    running_acc = 0.0
    start_time = time.time()

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_acc = 0.0

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                out = model(input_ids=x, labels=y)
                loss = out.loss / args.grad_accum
                logits = out.logits

            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean().item()

            step_loss += loss.item() * args.grad_accum
            step_acc += acc

            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        step_time = t1 - t0
        if step > args.warmup_steps:
            step_times.append(step_time)

        step_acc /= args.grad_accum
        running_loss += step_loss
        running_acc += step_acc

        if step % args.log_interval == 0 or step == 1:
            avg_loss = running_loss / (args.log_interval if step > 1 else 1)
            avg_acc = running_acc / (args.log_interval if step > 1 else 1)
            elapsed = time.time() - start_time
            print(
                f"Step {step:5d}/{args.max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"token_acc {avg_acc*100:.2f}% | "
                f"step_time {step_time*1000:.1f} ms | "
                f"elapsed {format_seconds(elapsed)}"
            )
            running_loss = 0.0
            running_acc = 0.0

    total_elapsed = time.time() - start_time

    if len(step_times) == 0:
        raise RuntimeError("No measured steps left after warmup. Reduce warmup-steps.")

    avg_step_time = float(np.mean(step_times))
    std_step_time = float(np.std(step_times))
    tokens_per_sec = tokens_per_step / avg_step_time
    seqs_per_sec = seqs_per_step / avg_step_time
    est_epoch_time = steps_per_epoch * avg_step_time

    eval_loss, eval_acc = evaluate(
        model=model,
        data=eval_data,
        block_size=args.block_size,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        device=device,
        amp_dtype=amp_dtype,
    )
    ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")

    print("\n" + "=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"Measured steps (after warmup): {len(step_times)}")
    print(f"Average step time (s)        : {avg_step_time:.6f}")
    print(f"Std step time (s)            : {std_step_time:.6f}")
    print(f"Throughput (seq/s)           : {seqs_per_sec:.2f}")
    print(f"Throughput (tok/s)           : {tokens_per_sec:.2f}")
    print(f"Estimated steps/epoch        : {steps_per_epoch:.2f}")
    print(f"Estimated epoch time (s)     : {est_epoch_time:.2f}")
    print(f"Estimated epoch time (min)   : {est_epoch_time/60.0:.2f}")
    print(f"Final eval loss              : {eval_loss:.4f}")
    print(f"Final eval ppl               : {ppl:.2f}")
    print(f"Final eval token acc (%)     : {eval_acc*100:.2f}")
    print(f"Total wall-clock run (s)     : {total_elapsed:.2f}")

    summary_path = os.path.join(args.out_dir, "benchmark_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"model_name={args.model_name}\n")
        f.write(f"data_dir={args.data_dir}\n")
        f.write(f"train_tokens={total_tokens}\n")
        f.write(f"block_size={args.block_size}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"grad_accum={args.grad_accum}\n")
        f.write(f"max_steps={args.max_steps}\n")
        f.write(f"warmup_steps={args.warmup_steps}\n")
        f.write(f"avg_step_time={avg_step_time:.8f}\n")
        f.write(f"std_step_time={std_step_time:.8f}\n")
        f.write(f"throughput_seq_per_sec={seqs_per_sec:.8f}\n")
        f.write(f"throughput_tok_per_sec={tokens_per_sec:.8f}\n")
        f.write(f"steps_per_epoch={steps_per_epoch:.8f}\n")
        f.write(f"estimated_epoch_time_sec={est_epoch_time:.8f}\n")
        f.write(f"eval_loss={eval_loss:.8f}\n")
        f.write(f"eval_ppl={ppl:.8f}\n")
        f.write(f"eval_token_acc={eval_acc:.8f}\n")

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()