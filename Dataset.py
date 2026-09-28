from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.transforms import BaseTransform
from torch_geometric.datasets import ModelNet
import torch

class Voxelize(BaseTransform):
    def __init__(self, R=32) -> None:
        self.R = R

    def forward(self, data):
        pos = data.pos - data.pos.mean(0, keepdim=True)
        pos = pos / pos.abs().max()
        idx = ((pos + 1) / 2 * (self.R - 1)).round().long()
        grid = torch.zeros(self.R, self.R, self.R)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        data.x = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, R, R, R]
        return data

class SelectClass(BaseTransform):
    def __init__(self, id) -> None:
        super().__init__()
        self.id = id

    def forward(self, data):
        idx = (data.y == self.id).nonzero(as_tuple=True)[0]
        iddata = data[idx]
        return iddata

class ModelNet10:
    def __init__(self) -> None:
        self.pre_transform = T.Compose([
            T.SamplePoints(num=2048),
            Voxelize()
        ]) 
                
        self.train_dataset = ModelNet(
            root="data/ModelNet10", 
            name="10", 
            train=True, 
            pre_filter=SelectClass(2),
            transform=self.pre_transform
        )
        
        self.loader = DataLoader(self.train_dataset, batch_size=32)
    