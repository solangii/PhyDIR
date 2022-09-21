import os
import glob
from datetime import datetime
import numpy as np
import torch
from . import utils

class Trainer():
    def __init__(self, cfgs, model):
        self.device = cfgs.get('device', 'cpu')
        self.num_epochs = cfgs.get('num_epochs', 60)
        self.batch_size = cfgs.get('batch_size', 8)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.resume = cfgs.get('resume', True)
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.archive_code = cfgs.get('archive_code', True)
        self.checkpoint_name = cfgs.get('checkpoint_name', None)
        self.test_result_dir = cfgs.get('test_result_dir', None)
        self.cfgs = cfgs

        self.model = model(cfgs)
        self.model.trainer = self
        self.train_loader, self.val_loader, self.test_loader = None, None, None #todo

    def load_checkpoint(self, optim=True):
        pass

    def save_checkpoint(self, epoch, optim=True):
        pass

    def save_clean_checkpoint(self, path):
        pass

    def test(self):
        pass

    def train(self):
        pass

    def run_epoch(self, loader, epoch=0, is_validation=False, is_test=False):
        pass
