from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn.fx import Sequential
from dsets.SolidWood import SolidWood
from utils import fdGetNoisy

class TrainProcess:
    def __init__(self, vm) -> None:
        self.loss = MSELoss()
        self.model = Sequential()
        self.ds = SolidWood()
        self.vm = vm
        self.adamw = AdamW(self.model.parameters(), lr=0.001)
        # Todo, add proper cosine annealing
        # self.scheduler = CosineAnnealingLR(self.adamw)
        
    def mbgd(self):
        for batch in self.ds.loader:
            batch
        ...

    def train(self):
        ...