from torch.nn.modules.loss import BCELoss

from Dataset import ModelNet10
from GAN import GAN
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from VersionManager import VersionManager
import torch
import numpy as np

class TrainProcesss:
    def __init__(self, model: GAN, vm: VersionManager, epochs=1000, device=torch.device('cpu')) -> None:
        torch.autograd.set_detect_anomaly(True)
        self.gan = model
        self.ds = ModelNet10()
        self.vm = vm
        self.writer = SummaryWriter(f'runs/{self.vm.name}')
        self.cepoch = 0
        self.device = device
        self.epochs = epochs
        self.gstep = 0
        self.doptim = AdamW(self.gan.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.goptim = AdamW(self.gan.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        
        self.dscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.doptim, T_max=self.epochs, eta_min=1e-7
        )
        self.gscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.goptim, T_max=self.epochs, eta_min=1e-7
        )
        
        self.lossess = []
        self.criterion = BCELoss()
        

    def mbgd_step(self, step_fn):
        avgd_loss = []
        avgg_loss = []
        mbgd_epoch = 1
        for x_batch in self.ds.loader:
   
            g, d = step_fn(x_batch)
            
            avgg_loss.append(g.detach().numpy())
            avgd_loss.append(d.detach().numpy())
            mbgd_epoch += 1
            self.gstep += 1
            
        return np.mean(avgg_loss), np.mean(avgd_loss)

    def step_fn(self, x_batch):
        real = x_batch.x
        real = real.detach().unsqueeze(1)
        latent_vector = torch.randn(self.ds.batch_size, 100, 1, 1, 1, device=self.device)
        fake = self.gan.generator(latent_vector)
        
        d_loss = self.gan.compute_d_loss(self.criterion, real, fake)
        
        d_loss.backward()
        self.doptim.step()
        self.doptim.zero_grad()
        
        g_loss = self.gan.compute_g_loss(self.criterion, fake)
        
        g_loss.backward()
        self.goptim.step()
        self.goptim.zero_grad()
        
        self.writer.add_scalar("MBGD/Generator", g_loss.item(), self.gstep)
        self.writer.add_scalar("MBGD/Discriminator", d_loss.item(), self.gstep)
        self.writer.add_scalars('MBGD/Generator-Discriminator', {'generator': g_loss.item(), 'discriminator': d_loss.item()}, self.gstep)
        
        return (g_loss, d_loss)
        
    def train(self):
        self.gan.discriminator.train()
        self.gan.generator.train()
        
        for e in range(self.epochs):
            self.vm.setEpoch(self.cepoch)
            
            ag, ad = self.mbgd_step(self.step_fn)
            self.writer.add_scalar("AVG/Generator", ag.item(), self.cepoch)
            self.writer.add_scalar("AVG/Discriminator", ad.item(), self.cepoch)
            self.writer.add_scalars('AVG/Generator-Discriminator', {'generator': ag.item(), 'discriminator': ad.item()}, self.cepoch)

            self.cepoch += 1
            self.lossess.append((ag, ad))

            self.dscheduler.step()
            self.gscheduler.step()
            self.vm.save()