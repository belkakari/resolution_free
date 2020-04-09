from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

from resolution_free.utils import get_grid


class OneImageSet(Dataset):
    def __init__(self,
                 img_path,
                 max_crop_size,
                 len_dloader
                 ):
        trans = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()])
        self.img = trans(Image.open(img_path))
        c, h, w = self.img.shape
        self.grid = get_grid(h, w, norm=True).permute(2, 0, 1) * 2 - 1
        self.len_dloader = len_dloader
        self.max_crop_size = max_crop_size

    def __len__(self):
        return self.len_dloader

    def __getitem__(self, idx):
        crop_size = 64 #np.random.randint(32, self.max_crop_size, 1)[0]
        crop_tl = np.random.randint(0, self.img.shape[1] - crop_size, 2)
        crop_br = crop_tl + crop_size
        img_crop = self.img[:, crop_tl[0]:crop_br[0], crop_tl[1]:crop_br[1]]
        grid_crop = self.grid[:, crop_tl[0]:crop_br[0], crop_tl[1]:crop_br[1]]
        return img_crop, grid_crop
