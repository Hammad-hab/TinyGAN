from Dataset import ModelNet10
from GAN import GAN


class TrainProcesss:
    def __init__(self) -> None:
        self.gan = GAN()
        self.ds = ModelNet10()
        pass
    ...