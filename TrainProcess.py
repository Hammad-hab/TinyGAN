from torch.nn.modules.loss import BCEWithLogitsLoss

from Dataset import ModelNet10
from GAN import GAN, compute_norm
from torch.optim import AdamW, Adam
from torch.utils.tensorboard import SummaryWriter
from Generator import Generator
from VersionManager import VersionManager
import torch
import numpy as np

class TrainProcesss:
    def __init__(self, model: GAN, vm: VersionManager, epochs=1000, device=torch.device('cpu')) -> None:
        self.gan = model
        self.ds = ModelNet10()
        self.vm = vm
        self.writer = SummaryWriter(f'runs/{self.vm.name}')
        self.cepoch = 0
        self.device = device
        self.epochs = epochs
        self.gstep = 0
        self.doptim = Adam(self.gan.discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))
        self.goptim = Adam(self.gan.generator.parameters(),     lr=1e-4, betas=(0.0, 0.9))
        self.real_iter = self._infinite(self.ds.loader)
        
        self.dscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.doptim, T_max=self.epochs, eta_min=1e-8
        )
        self.gscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.goptim, T_max=self.epochs, eta_min=1e-7
        )
        
        self.lossess = []
        self.criterion = BCEWithLogitsLoss()


    @staticmethod
    def _infinite(loader):
        while True:
            yield from loader

    def mbgd_step(self, step_fn):
         gl, dl = [], []
         for _ in range(len(self.ds.loader)):          # one "epoch" = len(loader) G steps
             g, d = step_fn()
             gl.append(g); dl.append(d)
             self.gstep += 1
         return np.mean(gl), np.mean(dl)

    
        
    def step_fn(self):
        G, D = self.gan.generator, self.gan.discriminator
        B = self.ds.batch_size
        w_est = []
    
        for _ in range(5):
            real = next(self.real_iter).x.detach().unsqueeze(1).to(self.device)
            z = torch.randn(real.size(0), 100, 1, 1, 1, device=self.device)
            fake = G(z).detach()
    
            d_real, d_fake = D(real).mean(), D(fake).mean()
            d_loss = d_fake - d_real + 10 * self.gan.gradient_penalty(real, fake)
    
            self.doptim.zero_grad()
            d_loss.backward()
            self.doptim.step()
            w_est.append((d_real - d_fake).item())
    
        z = torch.randn(B, 100, 1, 1, 1, device=self.device)
        g_loss = -D(G(z)).mean()
        self.goptim.zero_grad()
        g_loss.backward()
        gnorm = compute_norm(G)
        self.goptim.step()
    
        w = float(np.mean(w_est))
        self.writer.add_scalar("Batch/Generator/Loss", g_loss.item(), self.gstep)
        self.writer.add_scalar("Batch/Generator/Gradient Norm", gnorm.item(), self.gstep)
        self.writer.add_scalar("Batch/Critic/Wasserstein Estimate", w, self.gstep)
        return g_loss.item(), w
        
    def train(self):
        self.gan.discriminator.train()
        self.gan.generator.train()
        
        for e in range(self.epochs):
            self.vm.setEpoch(self.cepoch)
            
            ag, ad = self.mbgd_step(self.step_fn)
            self.writer.add_scalar("Epoch/Generator/Loss", ag.item(), self.cepoch)
            self.writer.add_scalar("Epoch/Discriminator/Loss", ad.item(), self.cepoch)
            self.writer.add_scalars('Epoch/Loss Comparison', {'generator': ag.item(), 'discriminator': ad.item()}, self.cepoch)

            self.cepoch += 1
            self.lossess.append((ag, ad))

            self.dscheduler.step()
            self.gscheduler.step()
            self.vm.save()