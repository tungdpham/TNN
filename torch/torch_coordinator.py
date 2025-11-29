import os

# ==========================================
# CONFIGURATION - SET ENV VARS BEFORE IMPORTS
# ==========================================
MASTER_ADDR = "192.168.78.32"
MASTER_PORT = "5000"
COORD_INTERFACE = "enp130s0"

# Set environment variables BEFORE importing torch.distributed
os.environ['GLOO_SOCKET_IFNAME'] = COORD_INTERFACE
os.environ['TP_SOCKET_IFNAME'] = COORD_INTERFACE
os.environ['MASTER_ADDR'] = MASTER_ADDR
os.environ['MASTER_PORT'] = MASTER_PORT

import time
import torch
import torch.optim as optim
import torch.distributed.rpc as rpc
from resnet_modules import ResNet9Part1, CIFAR10Bin, train_transform, test_transform, DataLoader
import trainer_lib 

def main():
    torch.set_num_threads(6)
    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f">>> [COORD] Local Device: {device}")
    else:
        device = torch.device("cpu")
        print(">>> [COORD] Local Device: CPU")

    print(f">>> [COORD] Initializing RPC on {MASTER_ADDR} via {COORD_INTERFACE}...")
    
    # Configure TensorPipe RPC options
    options = rpc.TensorPipeRpcBackendOptions(
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        rpc_timeout=60.0
    )
    # Force specific transports to avoid issues
    options._transports = ["uv"]

    rpc.init_rpc(
        "coordinator", 
        rank=0, 
        world_size=2, 
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options
    )

    # 2. Create Remote Trainer on Worker
    print(">>> [COORD] Creating remote trainer instance...")
    # We ask the worker to initialize on its own cuda:0
    worker_rref = rpc.remote("worker", trainer_lib.RemoteTrainer, args=("cuda:0",))

    # 3. Local Model Setup
    model_part1 = ResNet9Part1().to(device)
    optimizer1 = optim.Adam(model_part1.parameters(), lr=0.001)
    
    data_root = os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin")
    train_loader = DataLoader(CIFAR10Bin(data_root, True, train_transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(CIFAR10Bin(data_root, False, test_transform), batch_size=64, shuffle=False)

    print(">>> [COORD] Starting Training Loop...")
    
    try:
        for epoch in range(1, 6):
            print(f"\n===== Epoch {epoch} =====")
            epoch_start = time.perf_counter()
            model_part1.train()
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer1.zero_grad()
                
                # Part 1 Forward (Calculated on Coordinator GPU)
                h = model_part1(inputs)
                
                # Detach for RPC but keep h for backward pass
                # RPC cannot send CUDA tensors directly without a device map.
                h_send = h.detach().cpu()
                targets_send = targets.cpu()

                # Part 2 Remote Call - now returns gradient w.r.t. h
                loss, correct, total, h_grad = worker_rref.rpc_sync().run_train_step(h_send, targets_send)
                
                # Backprop through Part 1 using the received gradient
                h.backward(h_grad.to(device))
                optimizer1.step()
                
                epoch_loss += loss * total
                epoch_correct += correct
                epoch_total += total
                
                if i % 50 == 0:
                    print(f"Batch {i}: Loss {loss:.4f}")
            
            train_time = time.perf_counter() - epoch_start

            # Validation
            val_start = time.perf_counter()
            model_part1.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    h = model_part1(inputs)
                    
                    # --- FIX: Move to CPU for validation RPC too ---
                    h_send = h.cpu()
                    targets_send = targets.cpu()
                    
                    fut = worker_rref.rpc_sync().run_eval_step(h_send, targets_send)
                    _, c, t = fut
                    val_correct += c
                    val_total += t
            
            val_time = time.perf_counter() - val_start
            epoch_time = train_time + val_time
            
            avg_loss = epoch_loss / epoch_total
            train_acc = 100.0 * epoch_correct / epoch_total
            val_acc = 100.0 * val_correct / val_total
            
            print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}%")
            print(f"Time: {epoch_time:.2f}s (train: {train_time:.2f}s, val: {val_time:.2f}s)")

    except Exception as e:
        print(f">>> [COORD] ERROR: {e}")
    finally:
        rpc.shutdown()

if __name__ == "__main__":
    main()