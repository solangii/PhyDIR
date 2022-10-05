import os
import math
import glob
import torch
import torch.nn as nn
import torchvision
from .models import EDDeconv, Encoder, UNet, ConvLayer, ConfNet
from . import utils
from .renderer import Renderer

EPS = 1e-7


class PhyDIR():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.batch_size = cfgs.get('batch_size', 1)
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
        self.use_conf_map = cfgs.get('use_conf_map', True)
        self.lam_adv = cfgs.get('lam_adv', 0.5)
        self.lam_shape = cfgs.get('lam_shape', 0.3)
        self.lam_tex = cfgs.get('lam_tex', 0.3)
        self.lam_l1 = cfgs.get('lam_l1', 0.3)
        self.lr = cfgs.get('lr', 1e-4)
        self.renderer = Renderer(cfgs)

        ## networks and optimizers
        self.netD = EDDeconv(cin=3, cout=1, nf=256, zdim=512, activation=None)
        self.netL = Encoder(cin=3, cout=4, nf=32)
        self.netV = Encoder(cin=3, cout=6, nf=32)
        self.netT = UNet(n_channels=3, n_classes=32)
        self.netN = UNet(n_channels=32, n_classes=3)
        # self.netDis = Discriminator(size=512)
        self.netT_conv = ConvLayer(cin=32, cout=32)
        self.netF_conv = ConvLayer(cin=32, cout=32)
        if self.use_conf_map:
            self.netC = ConfNet(cin=3, cout=1, nf=256, zdim=512)

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

    def adversarial_loss(self, pred, target):
        if target:
            return -torch.mean(torch.log(pred + EPS))
        else:
            return -torch.mean(torch.log(1 - pred + EPS))

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

            ## 3D Networks
            # 1. predict depth
            self.canon_depth_raw = self.netD(self.input_im).squeeze(1)
            self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(k, -1).mean(1).view(k, 1, 1)
            self.canon_depth = self.canon_depth.tanh()
            self.canon_depth = self.depth_rescaler(self.canon_depth)

            # 2. predict light
            canon_light = self.netL(self.input_im) # Bx4
            self.canon_light_a = self.amb_light_rescaler(canon_light[:, :1])  # ambience term, Bx1
            self.canon_light_b = self.diff_light_rescaler(canon_light[:, 1:2])  # diffuse term, Bx1
            canon_light_dxy = canon_light[:, 2:]
            self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(k, 1).to(self.input_im.device)], 1)
            self.canon_light_d = self.canon_light_d / (
                (self.canon_light_d ** 2).sum(1, keepdim=True)) ** 0.5  # diffuse light direction, Bx3

            # 3. predict viewpoint
            self.view = self.netV(self.input_im)
            self.view = torch.cat([
                self.view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
                self.view[:, 3:5] * self.xy_translation_range,
                self.view[:, 5:] * self.z_translation_range], 1) # Bx6


            ## Texture Networks
            self.im_tex = self.netT(self.input_im) # Bx32xHxW
            self.cannon_im_tex = self.im_tex.mean(0, keepdim=True) # 1x32xHxW
            self.canon_im_tex = self.netT_conv(self.cannon_im_tex) # 1x32xHxW


            ## 3D Physical Process
            # multi-image-shading (shading, texture)
            self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth) # BxHxWx3
            self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(
                min=0).unsqueeze(1)
            canon_shading = self.canon_light_a.view(-1, 1, 1, 1) + self.canon_light_b.view(-1, 1, 1,
                                                                                           1) * self.canon_diffuse_shading
            self.shaded_texture = canon_shading * self.im_tex # Bx32xHxW
            self.shaded_cannon_texture = canon_shading * self.canon_im_tex # Bx32xHxW

            # grid sampling
            self.renderer.set_transform_matrices(self.view)
            self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
            grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
            self.recon_im_tex = nn.functional.grid_sample(self.shaded_texture, grid_2d_from_canon, mode='bilinear').unsqueeze(0)
            self.recon_cannon_im_tex = nn.functional.grid_sample(self.shaded_cannon_texture, grid_2d_from_canon, mode='bilinear').unsqueeze(0)

            self.fused_im_tex = torch.cat([self.recon_im_tex, self.recon_cannon_im_tex]).mean(0)
            self.fused_im_tex = self.netF_conv(self.fused_im_tex)


            ## Neural Appearance Renderer
            self.recon_im = self.netN(self.fused_im_tex)

            ## predict confidence map
            if self.use_conf_map:
                self.conf_sigma_l1 = self.netC(self.input_im)  # Bx1xHxW
            else:
                self.conf_sigma_l1 = None

            ## rotated image (not sure..)
            self.random_view = torch.rand(1, 6).to(self.input_im.device)
            self.random_view = torch.cat([
                self.random_view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
                self.random_view[:, 3:5] * self.xy_translation_range,
                self.random_view[:, 5:] * self.z_translation_range], 1) # Bx6
            self.renderer.set_transform_matrices(self.random_view)
            self.recon_depth_rotate = self.renderer.warp_canon_depth(self.canon_depth)
            grid_2d_from_canon_rotate = self.renderer.get_inv_warped_2d_grid(self.recon_depth_rotate)
            self.recon_im_tex_rotate = nn.functional.grid_sample(self.shaded_texture, grid_2d_from_canon_rotate, mode='bilinear').unsqueeze(0)
            self.recon_cannon_im_tex_rotate = nn.functional.grid_sample(self.shaded_cannon_texture, grid_2d_from_canon_rotate, mode='bilinear').unsqueeze(0)

            self.fused_im_tex_rotate = torch.cat([self.recon_im_tex_rotate, self.recon_cannon_im_tex_rotate]).mean(0)
            self.fused_im_tex_rotate = self.netF_conv(self.fused_im_tex_rotate)
            self.recon_im_rotate = self.netN(self.fused_im_tex_rotate)

            ## loss function
            self.loss_recon = self.photometric_loss(self.recon_im, self.input_im, self.conf_sigma_l1)
            # self.loss_adv =
            self.loss_tex = (self.netT(self.recon_im_rotate) - self.netT(self.input_im)).abs().mean()
            self.loss_shape = (self.netD(self.recon_im_rotate) - self.netD(self.input_im)).abs().mean()
            self.loss_light = (self.netL(self.recon_im_rotate) - self.netL(self.input_im)).abs().mean()
            loss_total += self.loss_recon + self.lam_shape * self.loss_shape +\
                          self.lam_tex * self.loss_tex + self.lam_l1 * self.loss_light
        self.loss_total = loss_total / self.batch_size
        metrics = {'loss': self.loss_total}
        return metrics
