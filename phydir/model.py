import os
import math
import glob
import torch
import torch.nn as nn
import torchvision
from .models import EDDeconv, Encoder, UNet, ConvLayer
from . import utils
from .renderer import Renderer


EPS = 1e-7


class PhyDIR():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 256)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.min_amb_light = cfgs.get('min_amb_light', 0.)
        self.max_amb_light = cfgs.get('max_amb_light', 1.)
        self.min_diff_light = cfgs.get('min_diff_light', 0.)
        self.max_diff_light = cfgs.get('max_diff_light', 1.)
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)

        self.lr = cfgs.get('lr', 1e-4)
        self.renderer = Renderer(cfgs)

        ## networks and optimizers
        self.netD = EDDeconv(cin=3, cout=1, nf=256, zdim=512, activation=None)
        self.netL = Encoder(cin=3, cout=2, nf=32)
        self.netV = Encoder(cin=3, cout=6, nf=32)
        self.netT = UNet(n_channels=3, n_classes=32)
        self.netC = ConvLayer(cin=32, cout=32)

        self.network_names = [k for k in vars(self) if 'net' in k]
        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4) # todo weight decay?

        ## other parameters

        ## depth rescaler: -1~1 -> min_depth~max_depth
        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth
        self.amb_light_rescaler = lambda x : (1+x)/2 *self.max_amb_light + (1-x)/2 *self.min_amb_light
        self.diff_light_rescaler = lambda x : (1+x)/2 *self.max_diff_light + (1-x)/2 *self.min_diff_light

    def init_optimizers(self):
        self.optimizer_names = []
        for net_name in self.network_names:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net','optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                getattr(self, k).load_state_dict(cp[k])

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names:
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
        # if self.other_param_names:
        #     for param_name in self.other_param_names:
        #         setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        loss = (im1-im2).abs()
        if conf_sigma is not None:
            loss = loss *2**0.5 / (conf_sigma +EPS) + (conf_sigma +EPS).log()
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def backward(self):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        self.loss_total.backward()
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()

    def forward(self, batch):
        # [data, data, data, ...., data]
        loss_total = 0
        for input in batch:
            # data: K, 3, 256, 256 (K is random number with 1~6)
            input = torch.stack(input, dim=0)
            self.input_im = input.to(self.device) *2.-1.
            k, c, h, w = self.input_im.shape

            # 3D Networks
            # 1. predict depth
            self.canon_depth_raw = self.netD(self.input_im).squeeze(1)
            self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(k, -1).mean(1).view(k, 1, 1)
            self.canon_depth = self.canon_depth.tanh()
            self.canon_depth = self.depth_rescaler(self.canon_depth)

            # 2. predict light
            canon_light = self.netL(self.input_im)  # Bx2
            self.canon_light_a = self.amb_light_rescaler(canon_light[:, :1])  # ambience term
            self.canon_light_b = self.diff_light_rescaler(canon_light[:, 1:])  # diffuse term
            canon_light_dxy = canon_light[:, 2:]
            self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(k * 2, 1).to(self.input_im.device)], 1)
            self.canon_light_d = self.canon_light_d / (
                (self.canon_light_d ** 2).sum(1, keepdim=True)) ** 0.5  # diffuse light direction

            # 3. predict viewpoint
            self.view = self.netV(self.input_im).repeat(2, 1)
            self.view = torch.cat([
                self.view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
                self.view[:, 3:5] * self.xy_translation_range,
                self.view[:, 5:] * self.z_translation_range], 1)

            # shading (light + shape, lambertian function)
            self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth)
            t1 = self.canon_light_d.view(-1, 1, 1, 3)
            t2 = self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)
            t3 = (self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)).sum(3)
            t4 = (self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(
                min=0)
            t5 = (self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(
                min=0).unsqueeze(1)
            self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(
                min=0).unsqueeze(1)
            canon_shading = self.canon_light_a.view(-1, 1, 1, 1) + self.canon_light_b.view(-1, 1, 1,
                                                                                           1) * self.canon_diffuse_shading

            # predict texture
            self.textures = None
            self.cannon_texture = None

            # multi-image-shading
            self.shaded_texture = None
            self.shaded_cannon_texture = None

            # reconstruction
            self.renderer.set_transform_matrices(self.view)
            self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
            self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)
            grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
            self.recon_im = nn.functional.grid_sample(self.canon_im, grid_2d_from_canon, mode='bilinear') # todo cannon_im -> texture

            # rendering

            # loss function
            # self.loss_total = self.loss_depth + self.loss_view + self.loss_light + self.loss_texture + self.loss_rendering
            self.loss_recon = None
            self.loss_adv = None
            self.loss_shape = None
            self.loss_tex = None
            self.loss_l1 = None
            loss_total += self.loss_recon + self.loss_adv + self.loss_shape + self.loss_tex + self.loss_l1
        self.loss_total = loss_total / len(batch)
        metrics = {'loss': self.loss_total}
        return metrics






