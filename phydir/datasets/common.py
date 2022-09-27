import os
import json
import torch
import torch.utils.data
import torchvision.transforms as tfs
import numpy as np
from PIL import Image


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


## simple image dataset ##
def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = {}
    for root, _, fnames in sorted(os.walk(os.path.join(dir, 'datalist'))):
        for fname in fnames:
            dlist = json.load(open(os.path.join(root, fname)))
            if len(dlist) >= 6:
                subject, _ = fname.split('.')
                images[subject] = dlist
    return images


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False):
        super(ImageDataset, self).__init__()
        self.root = data_dir
        self.image_size = image_size
        self.ids = make_dataset(data_dir) # {subject: [fpath, ...]}
        self.size = len(self.ids)
        self.crop = crop
        self.is_validation = is_validation

    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def __getitem__(self, index):
        K = np.random.randint(1, 7) # random 1~6
        random_ind = np.random.permutation(len(self.ids[index]))[:K]
        imgs = []
        for fpath in random_ind:
            if is_image_file(fpath):
                img = Image.open(fpath).convert('RGB')
                hflip = not self.is_validation and np.random.rand() > 0.5
                img = self.transform(img, hflip)
                imgs.append(img)
        imgs = torch.stack(imgs, dim=0) # [K, 3, 256, 256]
        return imgs

    def __len__(self):
        return self.size

    def name(self):
        return 'ImageDataset'


