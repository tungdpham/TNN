import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.distributed.rpc as rpc
from torch.distributed.pipeline.sync import Pipe
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


#############################################
# 1. CHIA RESNET-18 THÀNH 2 STAGE
#############################################
def make_stage1():
    m = models.resnet18(weights=None)
    return nn.Sequential(
        m.conv1, m.bn1, m.relu, m.maxpool,
        m.layer1, m.layer2,
    )


def make_stage2():
    m = models.resnet18(weights=None)
    return nn.Sequential(
        m.layer3, m.layer4,
        m.avgpool, nn.Flatten(), m.fc
    )


#############################################
# 2. MODULE REMOTE CHO STAGE 2 (NODE B)
#############################################
class RemoteStage2(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = make_stage2().to(device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(next(self.model.parameters()).device)
        return self.model(x)


#############################################
# 3. PIPELINE MODEL GỒM 2 STAGE
#############################################
def build_pipeline(rank, world_size):
    if rank == 0:
        # Node A: Stage 1 local, Stage 2 remote
        stage1 = make_stage1().to("cuda:0")

        # Tạo RRef đến node B (rank 1)
        stage2_rref = rpc.remote("worker1", RemoteStage2, args=("cuda:0",))

        # Xây pipeline: stage1 local, stage2 remote
        model = nn.Sequential(stage1, stage2_rref)

        pipe = Pipe(model, chunks=4, checkpoint="never")
        return pipe

    return None


#############################################
# 4. TRAINING LOOP (chạy trên node A)
#############################################
def train(pipe):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    dataset = datasets.FakeData(
        size=500, image_size=(3,224,224), num_classes=1000,
        transform=transform
    )  # Thay bằng ImageNet thật nếu có

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optim_pipe = optim.SGD(pipe.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 3):
        for imgs, labels in loader:
            imgs = imgs.cuda(0)
            labels = labels.cuda(0)

            out = pipe(imgs)
            loss = criterion(out.local_value(), labels)

            optim_pipe.zero_grad()
            loss.backward()
            optim_pipe.step()

        print(f"[Epoch {epoch}] Loss = {loss.item():.4f}")


#############################################
# 5. MAIN ENTRY FOR EACH NODE
#############################################
def run_worker(rank, world_size):
    if rank == 0:
        rpc.init_rpc(
            "worker0", rank=0, world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method="tcp://192.168.78.179:29500"
            )
        )

        pipe = build_pipeline(rank, world_size)
        train(pipe)

    else:
        rpc.init_rpc(
            "worker1", rank=1, world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method="tcp://192.168.78.179:29500"
            )
        )

    rpc.shutdown()


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    run_worker(rank, world_size)
