from torch.utils.data import DataLoader, Dataset
import os 
from PIL import Image

class SolidWoodDS(Dataset):
    def __init__(self, dir="/data/SolidWood", transform=None) -> None:
        super().__init__()
        self._paths = list(filter(lambda f: f.endswith(".jpg"), os.listdir(dir)))
        self.transform = transform

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index):
        img = Image.open(self._paths[index])
        if (self.transform):
            img = self.transform(img)
        return img


class SolidWood:
    def __init__(self, class_id=2, R=32, batch_size=32):
        self.batch_size = batch_size
        self.train_dataset = SolidWoodDS()
        self.loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
