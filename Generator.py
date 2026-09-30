from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self) -> None:
        self.c1 = nn.Conv3d(1, 16, 3, 2) # 32 -> 16 
        self.c2 = nn.Conv3d(16, 32, 3, 2) # 16 -> 8
        self.c23 = nn.Conv3d(32, 32, 3, 1) # 8 -> 8
        self.c3 = nn.Conv3d(32, 64, 3, 2) # 8 -> 4
        self.c4 = nn.Conv3d(64, 64, 3, 1) # 4 -> 4 
        
        self.c5 = nn.ConvTranspose3d(64, 32, 3, 2, 1, output_padding=1) # 4 -> 8
        self.c6 = nn.ConvTranspose3d(32, 16, 3, 2, 1, output_padding=1) # 8 -> 16
        self.c7 = nn.ConvTranspose3d(16, 1, 3, 2, 1, output_padding=1) # 16 -> 32

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c23(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        
        x = F.relu(self.c5(x))
        x = F.relu(self.c6(x))
        x = F.relu(self.c7(x))

        return x