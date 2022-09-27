import json
import os
import torchvision.transforms as tfs
import torch.utils.data
import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset
from common import ImageDataset


def get_data_loaders(cfgs):
    batch_size = cfgs.get('batch_size', 8)
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 256)
    crop = cfgs.get('crop', None)

    training_data = cfgs.get('training_data', 'celeba')
    run_train = cfgs.get('run_train', False)
    train_val_data_dir = cfgs.get('train_val_data_dir', './data')
    run_test = cfgs.get('run_test', False)
    test_data_dir = cfgs.get('test_data_dir', './data/test')

    train_loader = val_loader = test_loader = None

    get_loader = lambda **kargs: get_image_loader(**kargs, datasets=training_data, batch_size=batch_size, image_size=image_size, crop=crop)

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

def get_image_loader(data_dir, is_validation=False, datasets='celeba',
    batch_size=8, num_workers=4, image_size=256, crop=None):
    # data_list = []

    # todo
    if 'celeba' in datasets:
        pass
    # if 'casia' in cfg.trainig_data:
        # data_list.append()
        # pass
    # dataset = ConcatDataset(data_list)

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