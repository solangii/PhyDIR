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


def make_dataset(dir, prev_idx):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = {}
    for root, _, fnames in sorted(os.walk(os.path.join(dir, 'datalist'))):
        jfile = os.path.join(root, 'combined.json')
        if os.path.isfile(jfile):
            with open(jfile, 'r') as f:
                jdata = json.load(f)
            idx = prev_idx
            for k, v in jdata.items():
                if len(v) >= 6:
                    images[idx] = v
                    idx += 1
    return images

def make_paired_dataset(dir, used_data, AB_dnames=None, AB_fnames=None, partition='test'):
    A_dname, B_dname = AB_dnames or ('A', 'B')

    dir_A = os.path.join(dir, used_data[0], partition, A_dname) #todo list 제거
    dir_B = os.path.join(dir, used_data[0], partition, B_dname)
    assert os.path.isdir(dir_A), '%s is not a valid directory' % dir_A
    assert os.path.isdir(dir_B), '%s is not a valid directory' % dir_B

    images = []
    for root_A, _, fnames_A in sorted(os.walk(dir_A)):
        for fname_A in sorted(fnames_A):
            if is_image_file(fname_A):
                path_A = os.path.join(root_A, fname_A)
                root_B = root_A.replace(dir_A, dir_B, 1)
                if AB_fnames is not None:
                    fname_B = fname_A.replace(*AB_fnames)
                else:
                    fname_B = fname_A
                path_B = os.path.join(root_B, fname_B)
                if os.path.isfile(path_B):
                    images.append((path_A, path_B))
    return images

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False, K=None, idx=None):
        super(ImageDataset, self).__init__()
        self.root = data_dir
        self.image_size = image_size
        self.ids = make_dataset(data_dir, idx) # {subject: [fpath, ...]}
        self.size = len(self.ids)
        self.crop = crop
        self.is_validation = is_validation
        self.K = K

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
        # data : [(3, H, W), .... ]  K images, list of tensors
        data = []
        if self.K is None:
            self.K = np.random.randint(1, 7) # random 1~6
        random_ind = np.random.permutation(len(self.ids[index]))[:self.K]
        for i in random_ind:
            fpath = self.ids[index][i]
            if is_image_file(fpath):
                img = Image.open(os.path.join(self.root, fpath)).convert('RGB')
                hflip = not self.is_validation and np.random.rand() > 0.5
                img = self.transform(img, hflip)
                data.append(img)
        data = torch.stack(data, dim=0)
        return data

    def __len__(self):
        return self.size

    def name(self):
        return 'ImageDataset'

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, used_data, image_size=256, crop=None, is_validation=False, AB_dnames=None, AB_fnames=None, partition='test'):
        super(PairedDataset, self).__init__()
        self.root = data_dir
        self.paths = make_paired_dataset(data_dir, used_data=used_data ,AB_dnames=AB_dnames, AB_fnames=AB_fnames)
        self.size = len(self.paths)
        self.image_size = image_size
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
        path_A, path_B = self.paths[index % self.size]
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        hflip = not self.is_validation and np.random.rand()>0.5
        return self.transform(img_A, hflip=hflip), self.transform(img_B, hflip=hflip)

    def __len__(self):
        return self.size

    def name(self):
        return 'PairedDataset'



