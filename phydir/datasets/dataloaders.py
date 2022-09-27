import json
import os
import torchvision.transforms as tfs
import torch.utils.data
import numpy as np
from PIL import Image


def get_data_loaders(cfgs):
    batch_size = cfgs.get('batch_size', 8)
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 256)
    crop = cfgs.get('crop', None)

    run_train = cfgs.get('run_train', False)
    train_val_data_dir = cfgs.get('train_val_data_dir', './data')
    run_test = cfgs.get('run_test', False)
    test_data_dir = cfgs.get('test_data_dir', './data/test')

    load_gt_depth = cfgs.get('load_gt_depth', False)
    AB_dnames = cfgs.get('paired_data_dir_names', ['A', 'B'])
    AB_fnames = cfgs.get('paired_data_filename_diff', None)

    train_loader = val_loader = test_loader = None

    get_loader = lambda **kargs: get_image_loader(**kargs, batch_size=batch_size, image_size=image_size, crop=crop)

    if run_train:
        train_data_dir = os.path.join(train_val_data_dir, "train")
        val_data_dir = os.path.join(train_val_data_dir, "val")
        assert os.path.isdir(train_data_dir), "Training data directory does not exist: %s" %train_data_dir
        assert os.path.isdir(val_data_dir), "Validation data directory does not exist: %s" %val_data_dir
        print(f"Loading training data from {train_data_dir}")
        train_loader = get_loader(data_dir=train_data_dir, is_validation=False)
        print(f"Loading validation data from {val_data_dir}")
        val_loader = get_loader(data_dir=val_data_dir, is_validation=True)
    if run_test:
        assert os.path.isdir(test_data_dir), "Testing data directory does not exist: %s" %test_data_dir
        print(f"Loading testing data from {test_data_dir}")
        test_loader = get_loader(data_dir=test_data_dir, is_validation=True)

    return train_loader, val_loader, test_loader


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


## simple image dataset ##
def make_dataset(dir):
    dir = '../data/img_celeba'
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = {}
    for root, _, fnames in sorted(os.walk(os.path.join(dir, 'datalist'))):
        for fname in fnames:
            dlist = json.load(open(os.path.join(root, fname)))
            if len(dlist) >= 6:
                n_id, _ = fname.split('.')
                images[n_id] = dlist
    return images


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False):
        super(ImageDataset, self).__init__()
        self.root = data_dir
        self.image_size = image_size
        self.ids = make_dataset(data_dir)
        self.size = len(self.ids) # len(self.paths.keys()) # todo check size
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


def get_image_loader(data_dir, is_validation=False,
    batch_size=8, num_workers=4, image_size=256, crop=None):

    dataset = ImageDataset(data_dir, image_size=image_size, crop=crop, is_validation=is_validation)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


if __name__ == '__main__':
    import argparse
    from phydir import setup_runtime
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', default='../configs/debug.yml', type=str, help='Specify a config file path')
    parser.add_argument('--gpu', default=0, type=int, help='Specify a GPU device')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Specify the number of worker threads for data loaders')
    parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
    args = parser.parse_args()
    cfgs = setup_runtime(args)

    train_loader, val_loader, test_loader = get_data_loaders(cfgs)

    print("Training data:")
    for i, data in enumerate(train_loader):
        print(data.size())
        if i > 0:
            break

    print("Validation data:")
    for i, data in enumerate(val_loader):
        print(data.size())
        if i > 0:
            break

    print("Testing data:")
    for i, data in enumerate(test_loader):
        print(data.size())
        if i > 0:
            break

