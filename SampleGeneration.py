from GAN import GAN
from VersionManager import VersionManager
import torch
import numpy as np
from util import get_device

gan = GAN()
vm = VersionManager(gan, 'tiny-gan')
vm.load_latest(True, True)

lvector = torch.randn(1, 100, 1, 1, 1)
output = gan.generator(lvector)
