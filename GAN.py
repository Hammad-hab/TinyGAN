from Dataset import ModelNet10
from torch import nn
import torch
from Discriminator import Discriminator
from Generator import Generator


class GAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generator = Generator()
        self.bsize = 32
        self.discriminator = Discriminator()

    def compute_d_loss(self, criterion, real, fake):
        d_real = criterion(self.discriminator(real), torch.ones(self.bsize, 1))
        d_fake = criterion(self.discriminator(fake.detach()), torch.zeros(self.bsize, 1))
        d_loss = d_real + d_fake

        return d_loss

    def compute_g_loss(self, criterion, fake):
        return criterion(self.discriminator(fake), torch.ones(self.bsize, 1))