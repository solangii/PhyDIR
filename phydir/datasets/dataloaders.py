import os
import torch.utils.data
from torch.utils.data import ConcatDataset
from .common import ImageDataset
from .collate_fn import make_batch


def get_data_loaders(cfgs):
    batch_size = cfgs.get('batch_size', 8)
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 256)
    crop = cfgs.get('crop', None)
    K = cfgs.get('K', None)

    training_data = cfgs.get('training_data', 'celeba')
    run_train = cfgs.get('run_train', False)
    train_val_data_dir = cfgs.get('train_val_data_dir', './data')
    run_test = cfgs.get('run_test', False)
    test_data_dir = cfgs.get('test_data_dir', './data/test')

    train_loader = val_loader = test_loader = None

    get_loader = lambda **kargs: get_image_loader(**kargs, datasets=training_data, batch_size=batch_size,
                                                  image_size=image_size, crop=crop, K=K)

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
    batch_size=8, num_workers=4, image_size=256, crop=None, K=None):
    data_list = []

    if 'celeba' in datasets:
        celeba_dataset = ImageDataset(data_dir, image_size=image_size, crop=crop, is_validation=is_validation, K=K)
        data_list.append(celeba_dataset)
    if 'casia' in datasets:
        pass # todo
    dataset = ConcatDataset(data_list)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=make_batch
    )
    return loader

if __name__ == '__main__':
    import argparse
    from phydir import setup_runtime
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', default='configs/debug.yml', type=str, help='Specify a config file path')
    parser.add_argument('--gpu', default=0, type=int, help='Specify a GPU device')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Specify the number of worker threads for data loaders')
    parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
    args = parser.parse_args()
    cfgs = setup_runtime(args)

    train_loader, val_loader, test_loader = get_data_loaders(cfgs)
    if train_loader is not None:
        print(train_loader)
        print("Training data:")
        for i, data in enumerate(train_loader):
            print(data)
            if i > 0:
                break

    if val_loader is not None:
        print("Validation data:")
        for i, data in enumerate(val_loader):
            print(data)
            if i > 0:
                break

    if test_loader is not None:
        print("Testing data:")
        for i, data in enumerate(test_loader):
            print(data)
            if i > 0:
                break