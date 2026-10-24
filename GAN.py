from Dataset import ModelNet10
from torch import nn
import torch
from Discriminator import Discriminator
from Generator import Generator


def compute_norm(model):
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    return total_norm

class GAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generator = Generator()
        self.bsize = 32
        self.discriminator = Discriminator()

    def compute_d_loss(self, criterion, real, fake):
        pred = self.discriminator(real)
        # print('real, prediction', real.shape, pred.shape)
        d_real = criterion(pred, torch.ones(self.bsize, 1))
        d_fake = criterion(self.discriminator(fake.detach()), torch.zeros(self.bsize, 1))
        d_loss = d_real + d_fake

        return d_loss

    def gradient_penalty(self,real, fake):
       # real.size(0) is the batch size
       a = torch.rand(real.size(0), 1, 1, 1, 1, device=real.device)
       x = (a * real + (1 - a) * fake).requires_grad_(True)
       g, = torch.autograd.grad(self.discriminator(x).sum(), x, create_graph=True)
       return ((g.flatten(1).norm(dim=1) - 1) ** 2).mean()
    
    def compute_detailed_d_loss(self, criterion, real, fake):
        rl = self.discriminator(real)
        fk = self.discriminator(fake.detach())
        # print('real, prediction', real.shape, pred.shape)
        d_real = criterion(rl, torch.ones(self.bsize, 1))
        d_fake = criterion(fk, torch.zeros(self.bsize, 1))
        d_loss = d_real + d_fake

        return (d_loss, d_real, d_fake, rl, fk)

    def compute_g_loss(self, criterion, fake):
        return criterion(self.discriminator(fake), torch.ones(self.bsize, 1))
