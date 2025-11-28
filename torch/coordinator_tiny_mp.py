# coordinator_tiny_mp.py
import time
import socket
import torch
from torch.utils.data import DataLoader

from common_mp import send_msg, recv_msg
from torch_tiny_imagenet_trainer import (
    TinyImageNetDataset,
    get_data_augmentation,
    get_val_transform,
    TrainingConfig,
)

TINY_ROOT = "../data/tiny-imagenet-200"


def get_loaders(config: TrainingConfig):
    train_transform = get_data_augmentation()
    val_transform = get_val_transform()

    train_dataset = TinyImageNetDataset(
        config.dataset_path, split="train", transform=train_transform
    )
    val_dataset = TinyImageNetDataset(
        config.dataset_path, split="val", transform=val_transform
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def main():
    WORKER_A_IP = "192.168.78.179"
    WORKER_B_IP = "192.168.78.32"
    PORT_A = 6000
    PORT_B = 7000

    config = TrainingConfig()
    # ép train 2 epoch cho dễ so sánh
    config.epochs = 5
    config.device_type = "GPU"  # chỉ để print cho vui
    config.print_config()

    # Kết nối tới Worker A
    connA = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Coordinator: connecting to Worker A at", WORKER_A_IP, PORT_A)
    connA.connect((WORKER_A_IP, PORT_A))
    print("Coordinator: connected to Worker A.")

    # Kết nối tới Worker B
    connB = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Coordinator: connecting to Worker B at", WORKER_B_IP, PORT_B)
    connB.connect((WORKER_B_IP, PORT_B))
    print("Coordinator: connected to Worker B.")

    print("Coordinator: loading Tiny ImageNet from", config.dataset_path)
    train_loader, val_loader = get_loaders(config)

    for epoch in range(1, config.epochs + 1):
        print(f"\n========== Epoch {epoch}/{config.epochs} ==========")
        epoch_start = time.time()

        # ---------- TRAIN ----------
        batch_idx = 0
        for imgs, labels in train_loader:
            batch_idx += 1

            # Gửi batch training sang Worker A
            send_msg(connA, {"type": "train_batch", "inputs": imgs})

            # Nhận activation từ Worker A
            act_msg = recv_msg(connA, map_location="cpu")
            if act_msg is None or act_msg.get("type") != "activation":
                print("Coordinator: unexpected activation msg at train.")
                break
            act = act_msg["tensor"]

            # Gửi activation + labels sang Worker B
            send_msg(connB, {"type": "train_batch", "act": act, "labels": labels})

            if batch_idx % 100 == 0:
                print(f"Coordinator: train batch {batch_idx}/{len(train_loader)}")

        # ---------- VALIDATION ----------
        print("Coordinator: starting validation...")
        correct = 0
        total = 0

        for imgs, labels in val_loader:
            # Gửi batch val sang Worker A
            send_msg(connA, {"type": "val_batch", "inputs": imgs})

            # Nhận activation val từ Worker A
            act_msg = recv_msg(connA, map_location="cpu")
            if act_msg is None or act_msg.get("type") != "activation_val":
                print("Coordinator: unexpected activation_val msg.")
                break
            act = act_msg["tensor"]

            # Gửi activation sang Worker B để lấy logits
            send_msg(connB, {"type": "val_batch", "act": act})

            logits_msg = recv_msg(connB, map_location="cpu")
            if logits_msg is None or logits_msg.get("type") != "logits":
                print("Coordinator: unexpected logits msg.")
                break
            logits = logits_msg["logits"]  # [N, 200]

            preds = torch.argmax(logits, dim=1)
            labels_tensor = labels  # CPU

            correct += (preds == labels_tensor).sum().item()
            total += labels_tensor.size(0)

        val_acc = 100.0 * correct / total if total > 0 else 0.0
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}: time = {epoch_time:.2f} s, "
            f"validation accuracy = {val_acc:.2f}%"
        )

    # Gửi tín hiệu shutdown
    send_msg(connA, {"type": "shutdown"})
    send_msg(connB, {"type": "shutdown"})

    connA.close()
    connB.close()
    print("Coordinator: closed.")


if __name__ == "__main__":
    main()
