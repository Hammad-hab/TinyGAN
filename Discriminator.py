from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv3d(1, 16, 3, 2) # 32 -> 16 
        self.c2 = nn.Conv3d(16, 32, 3, 2) # 16 -> 8
        self.c23 = nn.Conv3d(32, 32, 3, 1) # 8 -> 8
        self.c3 = nn.Conv3d(32, 64, 3, 2) # 8 -> 4
        self.c4 = nn.Conv3d(64, 64, 3, 1) # 4 -> 4 
        self.flat = nn.Flatten() # [64, 4, 4, 4] -> 4096
        self.l5 = nn.Linear(4096, 1)
        self.s4 = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c23(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = self.flat(x)
        x = F.relu(self.l5(x))
        return self.s4(x)
