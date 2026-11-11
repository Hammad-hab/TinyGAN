from models.UNET import UNET
from proc.UNETTrainProcess import TrainProcesss
from utils.VersionManager import VersionManager
from utils import get_device

model = UNET()
device = get_device()
vm = VersionManager(model, 'tiny-texture-net')
tp = TrainProcesss(model, vm, 1000)
vm.load_latest(True, True)
print('Loaded Model Succesfully')

@vm.save_on_fail
def main():
    tp.train()

main()