import torch
import math

class Scheduler:
    def __init__(self, time) -> None:
        self.T = time
        self.s = 0.008
        self._tms = torch.linspace(0, self.T, self.T+1)
        self._num = torch.square(torch.cos(((self._tms/self.T + self.s)/(1+self.s))*(torch.pi/2)))
        self._denom = math.pow(math.cos((self.s/(1+self.s))*(torch.pi/2)), 2)

        self._alphab = self._num / self._denom
        
        self._betas = 1 - (self._alphab[1:] / self._alphab[:-1])
        self._betas = torch.clamp(self._betas, max=0.999)
        
        self._alphas = 1 - self._betas
        