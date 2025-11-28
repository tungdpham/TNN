# worker_a_tiny_mp.py
import socket
import torch
import torch.nn as nn
import torch.optim as optim

from common_mp import send_msg, recv_msg
from torch_tiny_imagenet_trainer import ResNet18TinyImageNet, TrainingConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_stage1(config: TrainingConfig):
    base = ResNet18TinyImageNet(num_classes=config.num_classes)
    stage1 = nn.Sequential(
        base.conv1,
        base.bn1,
        base.relu,
        base.maxpool,
        base.layer1,
        base.layer2,
    ).to(DEVICE)
    return stage1


def main():
    HOST = "0.0.0.0"
    PORT_FWD = 6000  # Coordinator -> Worker A
    PORT_BWD = 6001  # Worker B -> Worker A (gradient)

    config = TrainingConfig()
    # epochs ở đây không quan trọng, chỉ lấy LR/Adam hyperparams
    print("Worker A: using Adam LR =", config.lr_initial)

    stage1 = build_stage1(config)
    optimizer = optim.Adam(
        stage1.parameters(),
        lr=config.lr_initial,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )

    torch.backends.cudnn.benchmark = True

    # Socket nhận batch / lệnh từ coordinator
    sock_fwd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_fwd.bind((HOST, PORT_FWD))
    sock_fwd.listen(1)
    print("Worker A: waiting for Coordinator on port", PORT_FWD)
    coord_conn, _ = sock_fwd.accept()
    print("Worker A: Coordinator connected.")

    # Socket nhận gradient từ Worker B
    sock_bwd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_bwd.bind((HOST, PORT_BWD))
    sock_bwd.listen(1)
    print("Worker A: waiting for Worker B on port", PORT_BWD)
    wb_conn, _ = sock_bwd.accept()
    print("Worker A: Worker B connected.")

    while True:
        msg = recv_msg(coord_conn, map_location="cpu")
        if msg is None:
            break

        msg_type = msg.get("type", None)

        if msg_type == "shutdown":
            print("Worker A: shutdown signal received.")
            break

        elif msg_type == "train_batch":
            stage1.train()
            inputs = msg["inputs"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            inputs.requires_grad_(True)

            act = stage1(inputs)

            # Gửi activation cho Coordinator
            send_msg(coord_conn, {"type": "activation", "tensor": act.detach().cpu()})

            # Nhận gradient từ Worker B
            grad_msg = recv_msg(wb_conn, map_location=DEVICE)
            if grad_msg is None or grad_msg.get("type") != "grad":
                print("Worker A: unexpected grad message or connection closed.")
                break

            grad = grad_msg["grad"]
            act.backward(grad)

            optimizer.step()

        elif msg_type == "val_batch":
            stage1.eval()
            with torch.no_grad():
                inputs = msg["inputs"].to(DEVICE, non_blocking=True)
                act = stage1(inputs)

            send_msg(
                coord_conn,
                {"type": "activation_val", "tensor": act.detach().cpu()},
            )

        else:
            print("Worker A: unknown message type:", msg_type)
            break

    coord_conn.close()
    wb_conn.close()
    sock_fwd.close()
    sock_bwd.close()
    print("Worker A: closed.")


if __name__ == "__main__":
    main()
