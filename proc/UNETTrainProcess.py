from torch.nn import MSELoss
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn.fx import Sequential
from dsets.SolidWood import SolidWood
from utils import fdGetNoisy
from utils.Scheduler import Scheduler
import numpy as np

from utils.VersionManager import VersionManager

class TrainProcesss:
    def __init__(self, model, vm, epochs=1000,) -> None:
        self.criterion = MSELoss()
        self.model = model
        self.epochs = epochs
        self.ds = SolidWood()
        self.vm: VersionManager = vm
        self.start = self.vm.startepoch
        self.ns = Scheduler(1000)
        self.adamw = AdamW(self.model.parameters(), lr=0.001)
        self.writer = SummaryWriter()
        # Todo, add proper cosine annealing
        # self.scheduler = CosineAnnealingLR(self.adamw)
        
    def mbgd(self):
        avg = []
        for batch in self.ds.loader:
            t = torch.randint(0, 1000, (batch.shape[0],))
            xt, eps = fdGetNoisy(batch, t, self.ns)
            xp = self.model(xt, t)
            loss = self.criterion(xp, eps)
            loss.backward()
            lsv = loss.item()
            avg.append(lsv)
            self.adamw.step()
            self.adamw.zero_grad()
            
        return np.mean(avg) 

    def train(self):
        self.model.train()

        for e in range(self.start, self.epochs):
            loss = self.mbgd()
            self.vm.setEpoch(e)
            self.writer.add_scalar('UNET/Loss', loss)
            self.vm.save()
