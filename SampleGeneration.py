from GAN import GAN
from VersionManager import VersionManager
import torch
import numpy as np
from util import get_device
import time



gan = GAN()
vm = VersionManager(gan, "tiny-gan")
start = time.time()

result = vm.load_latest(True, True)

lvector = torch.randn(
    1, 100, 1, 1, 1
)
output = gan.generator(lvector)

arr = output.squeeze(0).cpu().detach().numpy()  # drop the batch dim
np.savez_compressed("output.npz", voxels=arr)