import os
import socket
import struct
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =====================================================
#  GIỚI HẠN CPU = 6 THREADS
# =====================================================
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
torch.set_num_threads(6)
torch.set_num_interop_threads(6)
print(">>> [WORKER] Using CPU threads:", torch.get_num_threads())


# =====================================================
#  Mô hình phần 2 của ResNet-9 (conv3, res3, res4, avgpool, fc)
# =====================================================
class BasicResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class ResNet9Part2(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2,
                               padding=1, bias=True)
        self.bn3   = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)

        self.res3 = BasicResidualBlock(256)
        self.res4 = BasicResidualBlock(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(256, num_classes, bias=True)

    def forward(self, x):
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =====================================================
#  Hàm send/recv object qua socket
# =====================================================
def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed unexpectedly")
        buf += chunk
    return buf


def send_obj(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!I", len(data))
    sock.sendall(header)
    sock.sendall(data)


def recv_obj(sock):
    header = recvall(sock, 4)
    if not header:
        return None
    (length,) = struct.unpack("!I", header)
    data = recvall(sock, length)
    return pickle.loads(data)


# =====================================================
#  Vòng loop worker
# =====================================================
def worker_loop(listen_ip="0.0.0.0", listen_port=5000):
    device = torch.device("cpu")
    print(f">>> [WORKER] Listening on {listen_ip}:{listen_port}")

    model_part2 = ResNet9Part2(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model_part2.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=1e-3,
        amsgrad=True,
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((listen_ip, listen_port))
    server.listen(1)

    conn, addr = server.accept()
    print(">>> [WORKER] Connected by", addr)

    try:
        while True:
            msg = recv_obj(conn)
            if msg is None:
                print(">>> [WORKER] No message, closing.")
                break

            cmd = msg.get("cmd", "")
            if cmd == "train_batch":
                h_cpu = msg["h"]
                y_cpu = msg["y"]

                h = h_cpu.to(device)
                y = y_cpu.to(device)

                h.requires_grad_(True)

                model_part2.train()
                optimizer.zero_grad()

                logits = model_part2(h)
                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    _, pred = logits.max(1)
                    correct = pred.eq(y).sum().item()
                    total = y.size(0)
                    grad_h = h.grad.detach().cpu()

                reply = {
                    "cmd": "train_result",
                    "grad_h": grad_h,
                    "batch_loss": float(loss.item()),
                    "batch_correct": int(correct),
                    "batch_total": int(total),
                }
                send_obj(conn, reply)

            elif cmd == "eval_batch":
                h_cpu = msg["h"]
                y_cpu = msg["y"]
                h = h_cpu.to(device)
                y = y_cpu.to(device)

                model_part2.eval()
                with torch.no_grad():
                    logits = model_part2(h)
                    loss = criterion(logits, y)
                    _, pred = logits.max(1)
                    correct = pred.eq(y).sum().item()
                    total = y.size(0)

                reply = {
                    "cmd": "eval_result",
                    "batch_loss": float(loss.item()),
                    "batch_correct": int(correct),
                    "batch_total": int(total),
                }
                send_obj(conn, reply)

            elif cmd == "shutdown":
                print(">>> [WORKER] Received shutdown command. Exiting.")
                break

            else:
                print(f">>> [WORKER] Unknown cmd: {cmd}")
                break

    finally:
        conn.close()
        server.close()
        print(">>> [WORKER] Closed connection and server.")


if __name__ == "__main__":
    worker_loop(listen_ip="0.0.0.0", listen_port=5000)
