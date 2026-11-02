from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.c5  = nn.ConvTranspose3d(latent_dim, 64, 4, 1, 0)  # 1 -> 4
        self.c56 = nn.ConvTranspose3d(64, 64, 3, 1, 1)          # 4 -> 4
        self.c6  = nn.ConvTranspose3d(64, 32, 3, 2, 1,
                                      output_padding=1)          # 4 -> 8
        self.c67 = nn.ConvTranspose3d(32, 32, 3, 1, 1)          # 8 -> 8
        self.c7  = nn.ConvTranspose3d(32, 16, 3, 2, 1,
                                      output_padding=1)          # 8 -> 16
        self.c78 = nn.ConvTranspose3d(16, 16, 3, 1, 1)          # 16 -> 16
        self.c8  = nn.ConvTranspose3d(16, 1, 3, 2, 1,
                                      output_padding=1)          # 16 -> 32
        
    def forward(self, z):
        x = F.relu(self.c5(z))
        x = F.relu(self.c56(x))
        x = F.relu(self.c6(x))
        x = F.relu(self.c67(x))
        x = F.relu(self.c7(x))
        x = F.relu(self.c78(x))
        x = self.c8(x)
        return x

        