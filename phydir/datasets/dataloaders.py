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

    run_train = cfgs.get('run_train', False)
    run_test = cfgs.get('run_test', False)

    training_data = cfgs.get('training_data', ['celeba'])
    data_root = cfgs.get('data_root', None)

    train_loader = val_loader = test_loader = None
    get_loader = lambda **kargs: get_image_loader(**kargs, training_data=training_data, batch_size=batch_size,
                                                  num_workers=num_workers, image_size=image_size, crop=crop, K=K)

    if run_train:
        train_loader = get_loader(data_dir=data_root, partition='train')
        val_loader = get_loader(data_dir=data_root, partition='val')
    if run_test:
        test_loader = get_loader(data_dir=data_root, partition='test')

    return train_loader, val_loader, test_loader

def get_dataset(data_dir, training_data, partition='train', image_size=256, crop=None, K=None, is_validation=False):
    data_list = []
    for dataset in training_data:
        dataset_dir = os.path.join(data_dir, dataset, partition)
        assert os.path.isdir(dataset_dir), "Training data directory does not exist: %s" %dataset_dir
        print(f"Loading {partition} data from {dataset_dir}")
        dataset = ImageDataset(dataset_dir, image_size=image_size, crop=crop, is_validation=is_validation, K=K)
        data_list.append(dataset)

    return data_list

def get_image_loader(data_dir, training_data, partition='train', batch_size=8, num_workers=4, image_size=256, crop=None, K=None):
    is_validation = False if partition == 'train' else False
    data_list = get_dataset(data_dir, training_data, partition, image_size, crop, K, is_validation)
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
    parser.add_argument('--config', default='configs/train_stage1.yml', type=str, help='Specify a config file path')
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