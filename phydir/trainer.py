import os
import glob
import numpy as np
import torch
from . import meters
from . import utils
from .datasets.dataloaders import get_data_loaders

class Trainer:
    def __init__(self, cfgs, model):
        self.device = cfgs.get('device', 'cpu')
        self.num_epochs = cfgs.get('num_epochs', 60)
        self.batch_size = cfgs.get('batch_size', 8)
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.resume = cfgs.get('resume', False)
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.archive_code = cfgs.get('archive_code', True)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', None)
        self.checkpoint_name = cfgs.get('checkpoint_name', None)
        self.result_dir = cfgs.get('result_dir', None)
        self.test_result_dir = cfgs.get('test_result_dir', None)
        self.stage = cfgs.get('stage', None) # 1. unet, 2. 3d, 3. joint
        self.pretrain_dir = cfgs.get('pretrain_dir', None) # for stage 2 and 3 (prev stage dir)
        self.test_name = cfgs.get('test_name', None) # for testing
        self.cfgs = cfgs

        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)
        self.model = model(cfgs)
        self.model.trainer = self
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(cfgs)
        self.set_result_dir()

    def set_result_dir(self):
        """Set result_dir based on model name"""
        if self.result_dir is None:
            self.result_dir = os.path.join('results', self.model.exp_name, f'stage{self.stage}')
        print(f"Saving results to {self.result_dir}")

    def load_checkpoint(self, optim=True, metrics=True, epoch=True):
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        checkpoint_name = self.get_checkpoint_name(self.pretrain_dir, self.checkpoint_dir, self.resume)
        if checkpoint_name is None:
            return 0 # from scratch

        if self.test_name is None:
            self.checkpoint_name = checkpoint_name.split('/')[-1].split('.')[0]
        else:
            self.checkpoint_name = self.test_name
        print(f"Loading checkpoint from {checkpoint_name}")

        cp = torch.load(checkpoint_name, map_location=self.device)
        self.model.load_model_state(cp)
        if self.pretrain_dir is None:
            if optim:
                self.model.load_optimizer_state(cp)
            if metrics:
                self.metrics_trace = cp['metrics_trace']
            if epoch:
                epoch = cp['epoch']
        else:
            epoch = 0
        return epoch

    def get_checkpoint_name(self, pretrain_dir=None, checkpoint_dir=None, resume=False):
        self.stage_change = False
        if pretrain_dir is not None:
            checkpoint_path = utils.get_ckpt(pretrain_dir, ext='pth')
        elif checkpoint_dir is not None:
            checkpoint_path = utils.get_ckpt(checkpoint_dir, ext='pth')
        elif resume:
            checkpoint_path = utils.get_latest_checkpoint(self.result_dir, ext='pth')
            if checkpoint_path is None and self.stage > 1:
                self.checkpoint_dir = self.result_dir.split('stage')[0] + f'stage{self.stage - 1}'
                checkpoint_path = utils.get_latest_checkpoint(self.checkpoint_dir, ext='pth')
                self.stage_change = True
        else:
            checkpoint_path = None
        return checkpoint_path

    def save_checkpoint(self, epoch, optim=True):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
        utils.xmkdir(self.result_dir)
        result_path = os.path.join(self.result_dir, f'checkpoint{epoch:03}.pth')
        state_dict = self.model.get_model_state()
        if optim:
            optimizer_state = self.model.get_optimizer_state()
            state_dict = {**state_dict, **optimizer_state}
        state_dict['metrics_trace'] = self.metrics_trace
        state_dict['epoch'] = epoch
        print(f"Saving checkpoint to {result_path}")
        torch.save(state_dict, result_path)
        if self.keep_num_checkpoint > 0:
            utils.clean_checkpoint(self.result_dir, keep_num=self.keep_num_checkpoint)

    def save_clean_checkpoint(self, path):
        """Save model state only to specified path."""
        torch.save(self.model.get_model_state(), path)

    def test(self):
        """Perform testing."""
        self.model.to_device(self.device)
        self.current_epoch = self.load_checkpoint(optim=False)
        if self.test_result_dir is None:
            self.test_result_dir = os.path.join(self.result_dir,
                                                f'test_results_{self.checkpoint_name}'.replace('.pth', ''))
        print(f"Saving testing results to {self.test_result_dir}")

        with torch.no_grad():
            m = self.run_epoch(self.test_loader, epoch=self.current_epoch, is_test=True)

        score_path = os.path.join(self.test_result_dir, 'eval_scores.txt')
        self.model.save_scores(score_path)

    def train(self):
        """Perform training."""
        ## archive code and configs
        if self.archive_code:
            utils.archive_code(os.path.join(self.result_dir, 'archived_code.zip'), filetypes=['.py', '.yml'])
        utils.dump_yaml(os.path.join(self.result_dir, 'configs.yml'), self.cfgs)

        ## initialize
        self.metrics_trace.reset()
        self.train_iter_per_epoch = len(self.train_loader)
        self.model.to_device(self.device)
        self.model.init_optimizers(self.stage)

        ## load ckpt
        start_epoch = self.load_checkpoint(optim=True, metrics=True, epoch=self.resume)
        start_epoch = 0 if self.stage_change else start_epoch

        ## initialize tensorboardX logger
        if self.use_logger:
            from tensorboardX import SummaryWriter
            from datetime import datetime, timezone, timedelta

            kst = timezone(timedelta(hours=9))
            self.logger = SummaryWriter(
                os.path.join(self.result_dir, 'logs', datetime.now(kst).strftime("%Y%m%d-%H%M%S"))) #timezone
            print(f"Saving logs to {self.logger.logdir}")
            ## cache one batch for visualization
            self.viz_input = self.val_loader.__iter__().__next__()

        ## run epochs
        print(f"{self.model.exp_name}: optimizing to {self.num_epochs} epochs")
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            metrics = self.run_epoch(self.train_loader, epoch)
            self.metrics_trace.append("train", metrics)

            with torch.no_grad():
                metrics = self.run_epoch(self.val_loader, epoch, is_validation=True)
                self.metrics_trace.append("val", metrics)

            if (epoch+1) % self.save_checkpoint_freq == 0:
                self.save_checkpoint(epoch+1, optim=True)
            self.metrics_trace.plot(pdf_path=os.path.join(self.result_dir, 'metrics.pdf'))
            self.metrics_trace.save(os.path.join(self.result_dir, 'metrics.json'))

        print(f"Training completed after {epoch+1} epochs.")

    def run_epoch(self, loader, epoch=0, is_validation=False, is_test=False):
        is_train = not is_validation and not is_test
        metrics = self.make_metrics()

        if is_train:
            print(f"Starting training epoch {epoch}")
            self.model.set_train()
        else:
            print(f"Starting validation epoch {epoch}")
            self.model.set_eval()

        for iter, input in enumerate(loader):
            if self.cfgs['loss_adv']:
                # discriminator update
                loss_d = self.model.forward(input, mode='discriminator')
                if self.stage is not 2 and is_train:
                    self.model.backward(mode='discriminator') # update discriminator

                # generator update
                loss_g = self.model.forward(input, mode='generator')
                if self.stage is not 2 and is_train:
                    self.model.backward(mode='generator') # update generator

            m = self.model.forward(input)
            if is_train:
                self.model.backward()
            elif is_test:
                if iter<3:
                    self.model.save_results(self.test_result_dir)
                else:
                    break

            if self.cfgs['loss_adv']:
                m['loss'] += (loss_d + loss_g) * self.model.lam_adv
            metrics.update(m, self.batch_size)
            print(f"{'T' if is_train else 'V'}{epoch:02}/{iter:05}/{metrics}")

            if self.use_logger and is_train:
                total_iter = iter + epoch * self.train_iter_per_epoch
                if total_iter % self.log_freq == 0:
                    self.model.forward(self.viz_input)
                    self.model.visualize(self.logger, total_iter=total_iter, max_bs=25, stage=self.stage)
            torch.cuda.empty_cache()
        return metrics

    def debug(self):
        ## initialize
        self.metrics_trace.reset()
        self.train_iter_per_epoch = len(self.train_loader)
        self.model.to_device(self.device)
        self.model.init_optimizers(self.stage)

        ## load ckpt
        start_epoch = self.load_checkpoint(optim=True, metrics=True, epoch=True)

        ## initialize tensorboardX logger
        if self.use_logger:
            from tensorboardX import SummaryWriter
            from datetime import datetime, timezone, timedelta

            kst = timezone(timedelta(hours=9))
            self.logger = SummaryWriter(
                os.path.join(self.result_dir, 'logs', datetime.now(kst).strftime("%Y%m%d-%H%M%S"))) #timezone
            print(f"Saving logs to {self.logger.logdir}")
            ## cache one batch for visualization
            self.viz_input = self.val_loader.__iter__().__next__()

        ## run epochs
        print(f"{self.model.exp_name}: optimizing to {self.num_epochs} epochs")
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            metrics = self.run_epoch_debug(self.train_loader, epoch)

        print("debug end")
        print(metrics)

    def run_epoch_debug(self, loader, epoch=0, is_validation=False, is_test=False):
        is_train = not is_validation and not is_test
        metrics = self.make_metrics()

        if is_train:
            print(f"Starting training epoch {epoch}")
            self.model.set_train()
        else:
            print(f"Starting validation epoch {epoch}")
            self.model.set_eval()

        for iter, input in enumerate(loader):
            m, v = self.model.calc_view_range(input)
            print("iter: ", iter, "mean value of view: ", m)
            print("iter: ", iter, "variance of view: ", v)
        return m, v
