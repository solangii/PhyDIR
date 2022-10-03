import torch

def make_batch(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return batch

if __name__ == '__main__':
    batch = []
    import numpy as np
    for i in range(8):
        sample = []
        idx = np.random.randint(1, 7)
        for j in range(idx):
            sample.append(torch.rand(3, 256, 256))
        batch.append(sample)
    batch = (make_batch(batch))
    print(batch)