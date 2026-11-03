from torch.nn import MSELoss
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn.fx import Sequential
from dsets.SolidWood import SolidWood
from utils import fdGetNoisy
from utils.Scheduler import Scheduler
import numpy as np

from utils.VersionManager import VersionManager

class TrainProcess:
    def __init__(self, vm) -> None:
        self.criterion = MSELoss()
        self.model = Sequential()
        self.ds = SolidWood()
        self.vm: VersionManager = vm
        self.start = self.vm.startepoch
        self.ns = Scheduler(1000)
        self.adamw = AdamW(self.model.parameters(), lr=0.001)
        # Todo, add proper cosine annealing
        # self.scheduler = CosineAnnealingLR(self.adamw)
        
    def mbgd(self):
        avg = []
        for batch in self.ds.loader:
            t = torch.randint(0, 1000, (batch.shape[0],))
            xt, eps = fdGetNoisy(batch, self.ns, t)
            xp = self.model(xt, t)
            loss = self.criterion(xp, eps)
            loss.backward()
            lsv = loss.item()
            avg.append(lsv)
            self.adamw.step()
            self.adamw.zero_grad()
            
        return np.mean(avg) 

    def train(self, epochs):
        self.model.train()

        for e in range(self.start, epochs):
            loss = self.mbgd()
            self.vm.setEpoch(e)
            
            self.vm.save()
            ...
