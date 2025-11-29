import torch
import torch.nn as nn
import torch.optim as optim
from resnet_modules import ResNet9Part2


class RemoteTrainer:
    """
    Remote trainer class that runs on the worker node.
    Handles Part 2 of ResNet-9 and returns gradients for Part 1.
    """
    def __init__(self, device_str: str = "cuda:0"):
        if torch.cuda.is_available() and device_str.startswith("cuda"):
            self.device = torch.device(device_str)
            print(f">>> [RemoteTrainer] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print(">>> [RemoteTrainer] Falling back to CPU.")
        
        self.model = ResNet9Part2(num_classes=10).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        print(f">>> [RemoteTrainer] Model Part 2 initialized on {self.device}.")

    def run_train_step(self, h, y):
        """
        Run a training step on Part 2.
        Returns: (loss, correct, total, h_grad)
        h_grad is the gradient w.r.t. h for backprop through Part 1.
        """
        h, y = h.to(self.device), y.to(self.device)
        h.requires_grad_(True)  # Enable gradient computation for h
        
        self.model.train()
        self.optimizer.zero_grad()
        
        logits = self.model(h)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        
        # Get gradient w.r.t. input h to send back to coordinator
        h_grad = h.grad.cpu()
        
        with torch.no_grad():
            _, pred = logits.max(1)
            correct = int(pred.eq(y).sum().item())
        
        return float(loss.item()), correct, int(y.size(0)), h_grad

    def run_eval_step(self, h, y):
        """
        Run an evaluation step on Part 2 (no gradients).
        Returns: (loss, correct, total)
        """
        h, y = h.to(self.device), y.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(h)
            loss = self.criterion(logits, y)
            _, pred = logits.max(1)
            correct = int(pred.eq(y).sum().item())
        
        return float(loss.item()), correct, int(y.size(0))
