from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv3d(1, 16, 3, 2, 1) # 32 -> 16 
        self.c2 = nn.Conv3d(16, 32, 3, 2, 1) # 16 -> 8
        self.c23 = nn.Conv3d(32, 32, 3, 1, 1) # 8 -> 8
        self.c3 = nn.Conv3d(32, 64, 3, 2, 1) # 8 -> 4
        self.c4 = nn.Conv3d(64, 64, 3, 1, 1) # 4 -> 4 
        self.flat = nn.Flatten() # [64, 4, 4, 4] -> 4096
        self.l5 = nn.Linear(4096, 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.c1(x))
        x = F.leaky_relu(self.c2(x))
        x = F.leaky_relu(self.c23(x))
        x = F.leaky_relu(self.c3(x))
        x = F.leaky_relu(self.c4(x))
        x = self.flat(x)
        x = self.l5(x)
        return x
