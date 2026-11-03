import torch
from torch import nn


class UNET(nn.Module):
    def __init__(self) -> None:
        # input 256x256
        # self.c1 = nn.Conv2d(3, )