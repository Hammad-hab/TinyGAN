from GAN import GAN
from VersionManager import VersionManager
import torch

from util import get_device

gan = GAN()
device = get_device()
vm = VersionManager(gan, 'tiny-gan')
vm.load_latest(True, True)

lvector = torch.randn(1, 100, 1, 1, 1, device=device)
gan.generator(lvector)
# TODO: implement the rest of the generation