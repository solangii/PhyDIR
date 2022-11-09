import argparse
from phydir import setup_runtime, Trainer, PhyDIR

## runtime arguments
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
parser.add_argument('--num_workers', default=0, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
args = parser.parse_args()

## set up
cfgs = setup_runtime(args)
trainer = Trainer(cfgs, PhyDIR)
run_train = cfgs.get('run_train', False)
run_test = cfgs.get('run_test', False)
debug = cfgs.get('debug', False)

## run
if debug:
    trainer.debug()
elif run_train:
    trainer.train()
elif run_test:
    trainer.test()
