from models.GAN import GAN
from proc.GANTrainProcess import TrainProcesss
from utils.VersionManager import VersionManager
from utils import get_device

model = GAN()
device = get_device()
vm = VersionManager(model, 'tiny-gan')
tp = TrainProcesss(model, vm, 1000)
vm.load_latest(True, True)

@vm.save_on_fail
def main():
    tp.train()

main()