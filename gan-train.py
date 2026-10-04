from GAN import GAN
from TrainProcess import TrainProcesss
from VersionManager import VersionManager
from util import get_device

model = GAN()
vm = VersionManager(model, 'tiny-gan')
tp = TrainProcesss(model, vm, 1000)
vm.load_latest(True, True)


@vm.save_on_fail
def main():
    tp.train()

main()