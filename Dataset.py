from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.transforms import BaseTransform
from torch_geometric.datasets import ModelNet
import torch

class SelectClass:
    def __init__(self, id):
        self.id = id

    def __call__(self, data):
        return int(data.y.item()) == self.id

    def __repr__(self):
        return f"SelectClass({self.id})"

class Voxelize(BaseTransform):
    def __init__(self, R=32):
        self.R = R

    def __call__(self, data):
        pos = data.pos - data.pos.mean(0, keepdim=True)
        pos = pos / pos.abs().max()
        idx = ((pos + 1) / 2 * (self.R - 1)).round().long()
        grid = torch.zeros(self.R, self.R, self.R)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        data.x = grid.unsqueeze(0) # [1, R, R, R] -> batch: [B, 1, R, R, R]
        return data

    def __repr__(self):
        return f"Voxelize({self.R})"

class ModelNet10:
    def __init__(self, class_id=2, R=32, batch_size=32):
        self.batch_size = batch_size
        self.train_dataset = ModelNet(
            root="data/ModelNet10",
            name="10",
            train=True,
            pre_filter=SelectClass(class_id),
            pre_transform=T.Compose([T.SamplePoints(2048), Voxelize(R)]),
        )
        self.loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
