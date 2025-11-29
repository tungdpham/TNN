import os

# =====================================================
# RPC Configuration - SET ENV VARS BEFORE IMPORTS
# =====================================================
COORDINATOR_ID = "coordinator"
WORKER_ID = "worker"
RANK = 1 
WORLD_SIZE = 2

# IP of the Coordinator (The Worker connects TO this IP)
MASTER_ADDR = "192.168.78.32" 
MASTER_PORT = "5000"

# Worker's network interface (change if different from coordinator)
WORKER_INTERFACE = "eth0"  # Change this to your worker's interface name

# Set environment variables BEFORE importing torch.distributed
os.environ['GLOO_SOCKET_IFNAME'] = WORKER_INTERFACE
os.environ['TP_SOCKET_IFNAME'] = WORKER_INTERFACE
os.environ['MASTER_ADDR'] = MASTER_ADDR
os.environ['MASTER_PORT'] = MASTER_PORT

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.rpc as rpc
from resnet_modules import ResNet9Part2

# --- GPU SETUP ---
if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print(f">>> [WORKER] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print(">>> [WORKER] Falling back to CPU.")

# Global Model & Optimizer
model_part2 = None
optimizer = None
criterion = None

def init_worker():
    global model_part2, optimizer, criterion
    os.environ["OMP_NUM_THREADS"] = "6"
    torch.set_num_threads(6)
    
    model_part2 = ResNet9Part2(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_part2.parameters(), lr=0.001)
    print(f">>> [WORKER] Model Part 2 initialized on {device}.")

@torch.no_grad()
def run_forward_only(h, y):
    h, y = h.to(device), y.to(device) # Ensure device
    model_part2.eval()
    logits = model_part2(h)
    loss = criterion(logits, y)
    _, pred = logits.max(1)
    return float(loss.item()), int(pred.eq(y).sum().item()), int(y.size(0))

def run_train_step(h, y):
    h, y = h.to(device), y.to(device) # Ensure device
    h.requires_grad_(True)  # Enable gradient computation for h
    model_part2.train()
    optimizer.zero_grad()
    
    logits = model_part2(h)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    
    # Get gradient w.r.t. input h to send back to coordinator
    h_grad = h.grad.cpu()
    
    with torch.no_grad():
        _, pred = logits.max(1)
        correct = int(pred.eq(y).sum().item())
        
    return float(loss.item()), correct, int(y.size(0)), h_grad

def worker_loop():
    print(f">>> [WORKER] Connecting to Coordinator at {MASTER_ADDR}:{MASTER_PORT}...")
    
    # Configure TensorPipe RPC options with explicit init_method
    options = rpc.TensorPipeRpcBackendOptions(
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        rpc_timeout=60.0
    )

    rpc.init_rpc(
        WORKER_ID,
        rank=RANK,
        world_size=WORLD_SIZE,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options
    )

    init_worker()
    print(">>> [WORKER] Ready. Waiting for commands...")
    rpc.shutdown()
    print(">>> [WORKER] Shutdown.")

if __name__ == "__main__":
    worker_loop()