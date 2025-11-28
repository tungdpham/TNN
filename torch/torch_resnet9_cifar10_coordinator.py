import os
import random
import socket
import struct
import pickle
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
load_dotenv()    # tự tìm và load file .env trong thư mục hiện tại
# =====================================================
#  GIỚI HẠN CPU = 6 THREADS
# =====================================================
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
torch.set_num_threads(6)
torch.set_num_interop_threads(6)
print(">>> [COORD] Using CPU threads:", torch.get_num_threads())


# =====================================================
#  CIFAR-10 .bin loader
# =====================================================
class CIFAR10Bin(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.data = []
        self.targets = []

        if train:
            batch_files = [f"data_batch_{i}.bin" for i in range(1, 6)]
        else:
            batch_files = ["test_batch.bin"]

        for fname in batch_files:
            path = os.path.join(root, fname)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Không tìm thấy file: {path}")
            with open(path, "rb") as f:
                arr = np.frombuffer(f.read(), dtype=np.uint8)
                arr = arr.reshape(-1, 3073)

                labels = arr[:, 0]
                images = arr[:, 1:].reshape(-1, 3, 32, 32)

                self.data.append(images)
                self.targets.append(labels)

        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        label = int(self.targets[idx])

        if self.transform:
            img = self.transform(img)

        return img, label


# =====================================================
#  Augmentation giống TNN
# =====================================================
CIFAR10_MEAN = torch.tensor([0.49139968, 0.48215827, 0.44653124]).view(3, 1, 1)
CIFAR10_STD  = torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(3, 1, 1)


def normalize(img: torch.Tensor) -> torch.Tensor:
    return (img - CIFAR10_MEAN) / CIFAR10_STD


def random_horizontal_flip(img: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if random.random() < p:
        img = torch.flip(img, dims=[2])
    return img


def random_crop_with_padding(img: torch.Tensor, padding: int = 4) -> torch.Tensor:
    c, h, w = img.shape
    padded = torch.zeros((c, h + 2 * padding, w + 2 * padding), dtype=img.dtype)
    padded[:, padding:padding + h, padding:padding + w] = img
    max_offset = 2 * padding
    x = random.randint(0, max_offset)
    y = random.randint(0, max_offset)
    return padded[:, y:y + h, x:x + w]


def train_transform(img: torch.Tensor) -> torch.Tensor:
    img = random_crop_with_padding(img, padding=4)
    img = random_horizontal_flip(img, p=0.5)
    img = normalize(img)
    return img


def test_transform(img: torch.Tensor) -> torch.Tensor:
    img = normalize(img)
    return img


# =====================================================
#  ResNet-9 Part 1 (conv1, conv2, res1, res2)
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


class ResNet9Part1(nn.Module):
    """
    The first part of the ResNet-9 model (Initial Conv layers and first residual blocks).
    The output is the hidden representation 'h' sent to the worker.
    """
    def __init__(self):
        super().__init__()
        # 3x32x32 -> 64x32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)

        # 64x32x32 -> 128x32x32 (Note: stride=1 to prepare for MaxPool)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)

        # 128x32x32 -> 128x16x16 (The added MaxPool layer)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # Two residual blocks on 128 channels, maintaining 16x16 size
        self.res1 = BasicResidualBlock(128)
        self.res2 = BasicResidualBlock(128)

    def forward(self, x):
        # Conv1 -> BN -> ReLU
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # Conv2 -> BN -> ReLU
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        # MaxPool Downsampling (32x32 -> 16x16)
        x = self.pool1(x) 
        # Residual Blocks
        x = self.res1(x)
        x = self.res2(x)
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
#  Training + Validation phân tán (có đo thời gian)
# =====================================================
def main():
    device = torch.device("cpu")
    print(">>> [COORD] Running on device:", device)

    epochs = int(os.getenv("EPOCHS", "5"))
    batch_size = int(os.getenv("BATCH_SIZE", "128"))
    lr_initial = float(os.getenv("LR_INITIAL", "0.001"))
    data_root = os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin")

    print(f">>> [COORD] CIFAR-10 bin at: {data_root}")
    print(f">>> [COORD] Epochs: {epochs}, Batch size: {batch_size}, LR: {lr_initial}")

    train_set = CIFAR10Bin(root=data_root, train=True,  transform=train_transform)
    test_set  = CIFAR10Bin(root=data_root, train=False, transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model_part1 = ResNet9Part1().to(device)
    optimizer1 = optim.Adam(
        model_part1.parameters(),
        lr=lr_initial,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=1e-3,
        amsgrad=True,
    )

    worker_ip = "192.168.78.212"
    worker_port = 5000

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f">>> [COORD] Connecting to WORKER at {worker_ip}:{worker_port} ...")
    sock.connect((worker_ip, worker_port))
    print(">>> [COORD] Connected to WORKER.")

    try:
        for epoch in range(1, epochs + 1):
            print(f"\n===== Epoch {epoch}/{epochs} =====")
            epoch_start = time.time()
            last_100_start = time.time()

            # ---------------- TRAIN ----------------
            model_part1.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer1.zero_grad()
                h = model_part1(inputs)  # (N,128,H,W)

                h_cpu = h.detach().cpu()
                y_cpu = targets.detach().cpu()

                msg = {
                    "cmd": "train_batch",
                    "h": h_cpu,
                    "y": y_cpu,
                }
                send_obj(sock, msg)

                reply = recv_obj(sock)
                if reply is None or reply.get("cmd") != "train_result":
                    raise RuntimeError(">>> [COORD] Invalid reply from WORKER during train_batch")

                grad_h_cpu = reply["grad_h"]
                batch_loss = reply["batch_loss"]
                batch_correct = reply["batch_correct"]
                batch_total = reply["batch_total"]

                grad_h = grad_h_cpu.to(device)
                h.backward(grad_h)
                optimizer1.step()

                running_loss += batch_loss * batch_total
                running_correct += batch_correct
                running_total += batch_total

                if (batch_idx + 1) % 100 == 0:
                    batch_acc = 100.0 * batch_correct / batch_total
                    elapsed_100 = time.time() - last_100_start
                    last_100_start = time.time()
                    print(
                        f"[Train Batch {batch_idx+1}/{len(train_loader)}] "
                        f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}% | "
                        f"100-batch time: {elapsed_100:.2f}s"
                    )

            train_loss = running_loss / running_total
            train_acc = 100.0 * running_correct / running_total

            # ---------------- VALIDATION ----------------
            model_part1.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    h = model_part1(inputs)
                    h_cpu = h.detach().cpu()
                    y_cpu = targets.detach().cpu()

                    msg = {
                        "cmd": "eval_batch",
                        "h": h_cpu,
                        "y": y_cpu,
                    }
                    send_obj(sock, msg)

                    reply = recv_obj(sock)
                    if reply is None or reply.get("cmd") != "eval_result":
                        raise RuntimeError(">>> [COORD] Invalid reply from WORKER during eval_batch")

                    batch_loss = reply["batch_loss"]
                    batch_correct = reply["batch_correct"]
                    batch_total = reply["batch_total"]

                    val_loss_sum += batch_loss * batch_total
                    val_correct += batch_correct
                    val_total += batch_total

            val_loss = val_loss_sum / val_total
            val_acc = 100.0 * val_correct / val_total

            epoch_time = time.time() - epoch_start  # train + val

            print(
                f"Epoch {epoch}/{epochs} Completed in {epoch_time:.2f}s\n"
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
                f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%"
            )

        print("\n>>> [COORD] Training done, sending shutdown to WORKER...")
        send_obj(sock, {"cmd": "shutdown"})

    finally:
        sock.close()
        print(">>> [COORD] Closed socket to WORKER.")


if __name__ == "__main__":
    main()
