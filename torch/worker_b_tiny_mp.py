# worker_b_tiny_mp.py
import socket
import torch
import torch.nn as nn
import torch.optim as optim

from common_mp import send_msg, recv_msg
from torch_tiny_imagenet_trainer import ResNet18TinyImageNet, TrainingConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_stage2(config: TrainingConfig):
    base = ResNet18TinyImageNet(num_classes=config.num_classes)
    stage2 = nn.Sequential(
        base.layer3,
        base.layer4,
        base.avgpool,
        base.flatten,
        base.fc,
    ).to(DEVICE)
    return stage2


def main():
    WORKER_A_IP = "192.168.78.179"
    PORT_ACT = 7000  # Coordinator -> Worker B
    PORT_GRAD = 6001  # Worker B -> Worker A

    config = TrainingConfig()
    print("Worker B: using Adam LR =", config.lr_initial)

    stage2 = build_stage2(config)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    optimizer = optim.Adam(
        stage2.parameters(),
        lr=config.lr_initial,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )

    torch.backends.cudnn.benchmark = True

    # Socket nhận activation / lệnh từ Coordinator
    sock_act = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_act.bind(("0.0.0.0", PORT_ACT))
    sock_act.listen(1)
    print("Worker B: waiting for Coordinator on port", PORT_ACT)
    coord_conn, _ = sock_act.accept()
    print("Worker B: Coordinator connected.")

    # Socket gửi gradient cho Worker A
    grad_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Worker B: connecting to Worker A at", WORKER_A_IP, PORT_GRAD)
    grad_conn.connect((WORKER_A_IP, PORT_GRAD))
    print("Worker B: connected to Worker A.")

    batch_count = 0

    while True:
        msg = recv_msg(coord_conn, map_location="cpu")
        if msg is None:
            break

        msg_type = msg.get("type", None)

        if msg_type == "shutdown":
            print("Worker B: shutdown signal received.")
            break

        elif msg_type == "train_batch":
            stage2.train()
            act = msg["act"].to(DEVICE, non_blocking=True)
            labels = msg["labels"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            act.requires_grad_(True)

            outputs = stage2(act)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Gửi gradient của activation về Worker A
            send_msg(grad_conn, {"type": "grad", "grad": act.grad.detach()})

            batch_count += 1
            if batch_count % 50 == 0:
                print(f"Worker B: train batch {batch_count}, loss={loss.item():.4f}")

        elif msg_type == "val_batch":
            stage2.eval()
            with torch.no_grad():
                act = msg["act"].to(DEVICE, non_blocking=True)
                logits = stage2(act)

            send_msg(coord_conn, {"type": "logits", "logits": logits.detach().cpu()})

        else:
            print("Worker B: unknown message type:", msg_type)
            break

    coord_conn.close()
    grad_conn.close()
    sock_act.close()
    print("Worker B: closed.")


if __name__ == "__main__":
    main()
