import os
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .models import EDDeconv, Encoder, UNet, ConvLayer, ConfNet, PerceptualLoss
from . import utils
from .renderer import Renderer
from .models.stylegan2 import Discriminator, DiscriminatorLoss, GeneratorLoss
# from .models.resnet import ResNet, BasicBlock
from .meters import TotalAverage

EPS = 1e-7


class PhyDIR():
    def __init__(self, cfgs):
        self.exp_name = cfgs.get('exp_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.batch_size = cfgs.get('batch_size', 1)
        self.image_size = cfgs.get('image_size', 256)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.border_depth = cfgs.get('border_depth', (0.7*self.max_depth + 0.3*self.min_depth)) # temporary
        self.min_amb_light = cfgs.get('min_amb_light', 0.)
        self.max_amb_light = cfgs.get('max_amb_light', 1.)
        self.min_diff_light = cfgs.get('min_diff_light', 0.)
        self.max_diff_light = cfgs.get('max_diff_light', 1.)
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)
        self.use_conf_map = cfgs.get('use_conf_map', True)
        self.lam_flip = cfgs.get('lam_f', 0.5)
        self.lam_flip_start_epoch = cfgs.get('lam_flip_start_epoch', 0)
        self.lam_perc = cfgs.get('lam_perc', 1)
        self.lam_adv = cfgs.get('lam_adv', 0.5)
        self.lam_shape = cfgs.get('lam_shape', 0.3)
        self.lam_tex = cfgs.get('lam_tex', 0.3)
        self.lam_light = cfgs.get('lam_light', 0.3)
        self.lr = cfgs.get('lr', 1e-4)
        self.K = cfgs.get('K', None)
        self.tex_channels = cfgs.get('tex_channels', 32)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)
        self.renderer = Renderer(cfgs)

        ## networks and optimizers
        self.netD = EDDeconv(cin=3, cout=1, nf=64, zdim=256, activation=None) # 3, 1, 256, 512, None
        self.netL = Encoder(cin=3, cout=4, nf=32)
        self.netV = Encoder(cin=3, cout=6, nf=32)
        self.netT = UNet(n_channels=3, n_classes=self.tex_channels)
        self.netN = UNet(n_channels=self.tex_channels, n_classes=3)
        self.discriminator = Discriminator(int(math.log2(self.image_size)), n_features = 512, max_features = 512).to(self.device)
        self.netT_conv = ConvLayer(cin=self.tex_channels, cout=self.tex_channels)
        self.netF_conv = ConvLayer(cin=self.tex_channels * 2, cout=self.tex_channels)
        if self.use_conf_map:
            self.netC = ConfNet(cin=3, cout=2, nf=64, zdim=128) # 3, 1, 256, 512

        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()
        self.perceptual_loss = PerceptualLoss(requires_grad=False)

        self.network_names = [k for k in vars(self) if 'net' in k]
        self.loss_names = [k for k in vars(self) if 'loss' in k]
        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, betas=(0.9, 0.999))

        ## depth rescaler: -1~1 -> min_depth~max_depth
        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth
        self.amb_light_rescaler = lambda x : (1+x)/2 *self.max_amb_light + (1-x)/2 *self.min_amb_light
        self.diff_light_rescaler = lambda x : (1+x)/2 *self.max_diff_light + (1-x)/2 *self.min_diff_light

        self.debug = cfgs.get('debug', False)
        if self.debug:
            self.view_m = TotalAverage()
            self.view_v = TotalAverage()

    def init_optimizers(self, stage=None):
        target_nets = {1: ['netC', 'netT', 'netN', 'netT_conv', 'netF_conv'],
                       2: ['netC', 'netD', 'netL', 'netV'],
                       3: ['netC', 'netD', 'netL', 'netV', 'netT', 'netN', 'netT_conv', 'netF_conv']}
        freeze_nets = {1: ['netD', 'netL', 'netV'],
                       2: ['netT', 'netN', 'netT_conv', 'netF_conv', 'discriminator'],
                       3: []}
        self.set_requires_grad(freeze_nets[stage], requires_grad=False)
        self.set_requires_grad(target_nets[stage], requires_grad=True)

        self.optimizer_names = []
        for net_name in target_nets[stage]:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net','optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]

        if stage is not 2:
            self.optimizer_discriminator = self.make_optimizer(self.discriminator)

    def set_requires_grad(self, net_names, requires_grad=True):
        for net_name in net_names:
            utils.set_requires_grad(getattr(self, net_name), requires_grad)

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
        for loss_name in self.loss_names:
            setattr(self, loss_name, getattr(self, loss_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        loss = (im1-im2).abs()
        if conf_sigma is not None:
            loss = loss *(2**0.5) / (conf_sigma +EPS) + (conf_sigma*(2**0.5) +EPS).log()
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

    def update_D(self):
        self.optimizer_discriminator.zero_grad()
        self.loss_d.backward()
        self.optimizer_discriminator.step()

    def forward(self, batch, mode=None):
        # [data, data, data, ...., data]
        self.loss_total = 0
        if self.load_gt_depth: # todo
            input, depth_gt = batch
        if self.K is None:
            for input in batch:
                # data: K, 3, 256, 256 (K is random number with 1~6)
                self.input_im = input.to(self.device) *2.-1.
                k, c, h, w = self.input_im.shape

                ## predict depth
                canon_depth_raw = self.netD(self.input_im).squeeze(1)
                self.canon_depth = canon_depth_raw - canon_depth_raw.view(k, -1).mean(1).view(k, 1, 1)
                self.canon_depth = self.canon_depth.tanh()
                self.canon_depth = self.depth_rescaler(self.canon_depth)

                ## predict light
                canon_light = self.netL(self.input_im) # Bx4
                self.canon_light_a = self.amb_light_rescaler(canon_light[:, :1])  # ambience term, Bx1
                self.canon_light_b = self.diff_light_rescaler(canon_light[:, 1:2])  # diffuse term, Bx1
                canon_light_dxy = canon_light[:, 2:]
                self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(k, 1).to(self.input_im.device)], 1)
                self.canon_light_d = self.canon_light_d / (
                    (self.canon_light_d ** 2).sum(1, keepdim=True)) ** 0.5  # diffuse light direction, Bx3

                ## predict viewpoint
                self.view = self.netV(self.input_im)
                self.view = torch.cat([
                    self.view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
                    self.view[:, 3:5] * self.xy_translation_range,
                    self.view[:, 5:] * self.z_translation_range], 1) # Bx6

                ## Texture Networks
                self.im_tex = self.netT(self.input_im) # Bx32xHxW
                self.canon_im_tex = self.im_tex.mean(0, keepdim=True) # 1x32xHxW
                self.canon_im_tex = self.netT_conv(self.canon_im_tex) # 1x32xHxW

                ## multi-image-shading (shading, texture)
                self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth) # BxHxWx3
                self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(
                    min=0).unsqueeze(1)
                canon_shading = self.canon_light_a.view(-1, 1, 1, 1) + self.canon_light_b.view(-1, 1, 1,
                                                                                               1) * self.canon_diffuse_shading
                self.shaded_texture = canon_shading * self.im_tex # Bx32xHxW
                self.shaded_canon_texture = canon_shading * self.canon_im_tex # Bx32xHxW

                ## grid sampling
                self.renderer.set_transform_matrices(self.view)
                self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
                self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)
                grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
                self.recon_im_tex = nn.functional.grid_sample(self.shaded_texture, grid_2d_from_canon, mode='bilinear').unsqueeze(0)
                self.recon_canon_im_tex = nn.functional.grid_sample(self.shaded_canon_texture, grid_2d_from_canon, mode='bilinear').unsqueeze(0)

                self.fused_im_tex = torch.cat([self.recon_im_tex, self.recon_canon_im_tex]).mean(0)
                self.fused_im_tex = self.netF_conv(self.fused_im_tex)

                ## Neural Appearance Renderer
                self.recon_im = self.netN(self.fused_im_tex)

                ## predict confidence map
                if self.use_conf_map:
                    self.conf_sigma_l1 = self.netC(self.input_im)  # Bx1xHxW
                else:
                    self.conf_sigma_l1 = None

                ## rotated image (not sure..)
                random_view = torch.rand(1, 6).to(self.input_im.device)
                random_view = torch.cat([
                    random_view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
                    random_view[:, 3:5] * self.xy_translation_range,
                    random_view[:, 5:] * self.z_translation_range], 1) # Bx6
                self.renderer.set_transform_matrices(random_view)
                self.recon_depth_rotate = self.renderer.warp_canon_depth(self.canon_depth)
                grid_2d_from_canon_rotate = self.renderer.get_inv_warped_2d_grid(self.recon_depth_rotate)
                self.recon_im_tex_rotate = nn.functional.grid_sample(self.shaded_texture, grid_2d_from_canon_rotate, mode='bilinear').unsqueeze(0)
                self.recon_canon_im_tex_rotate = nn.functional.grid_sample(self.shaded_canon_texture, grid_2d_from_canon_rotate, mode='bilinear').unsqueeze(0)

                self.fused_im_tex_rotate = torch.cat([self.recon_im_tex_rotate, self.recon_canon_im_tex_rotate]).mean(0)
                self.fused_im_tex_rotate = self.netF_conv(self.fused_im_tex_rotate)
                self.recon_im_rotate = self.netN(self.fused_im_tex_rotate)

                ## loss function
                loss_recon = self.photometric_loss(self.recon_im, self.input_im, self.conf_sigma_l1)
                # self.loss_adv =
                loss_tex = (self.netT(self.recon_im_rotate) - self.netT(self.input_im)).abs().mean()
                loss_shape = (self.netD(self.recon_im_rotate) - self.netD(self.input_im)).abs().mean()
                loss_light = (self.netL(self.recon_im_rotate) - self.netL(self.input_im)).abs().mean()
                self.loss_total += loss_recon + self.lam_shape * loss_shape +\
                              self.lam_tex * loss_tex + self.lam_light * loss_light
        else:
            self.input_im = torch.stack(batch, dim=0).to(self.device) *2.-1.  # b, k, 3, h, w
            b, k, c, h, w = self.input_im.shape
            self.input_im = self.input_im.view(b * k, c, h, w)

            ## 3D Networks
            # 1. predict depth
            self.canon_depth_raw = self.netD(self.input_im).squeeze(1) # b*k, 1, h, w
            self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(b*k, -1).mean(1).view(b*k, 1, 1)
            self.canon_depth = self.canon_depth.tanh()
            self.canon_depth = self.depth_rescaler(self.canon_depth)

            ## clamp border depth
            depth_border = torch.zeros(1, h, w - 4).to(self.input_im.device)
            depth_border = nn.functional.pad(depth_border, (2, 2), mode='constant', value=1)
            self.canon_depth = self.canon_depth * (1 - depth_border) + depth_border * self.border_depth
            self.canon_depth = torch.cat([self.canon_depth, self.canon_depth.flip(2)], 0)  # flip

            # 2. predict light
            canon_light = self.netL(self.input_im).repeat(2,1)   # b*k x4
            self.canon_light_a = self.amb_light_rescaler(canon_light[:, :1])  # ambience term, b*kx1
            self.canon_light_b = self.diff_light_rescaler(canon_light[:, 1:2])  # diffuse term, b*kx1
            canon_light_dxy = canon_light[:, 2:]
            self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(b*k*2, 1).to(self.input_im.device)], 1)
            self.canon_light_d = self.canon_light_d / (
                (self.canon_light_d ** 2).sum(1, keepdim=True)) ** 0.5  # diffuse light direction, b*kx3

            # 3. predict viewpoint
            self.view = self.netV(self.input_im).repeat(2,1)
            self.view = torch.cat([
                self.view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
                self.view[:, 3:5] * self.xy_translation_range,
                self.view[:, 5:] * self.z_translation_range], 1)  # b*kx6

            ## Texture Networks
            self.im_tex = self.netT(self.input_im)  # b*kx32xHxW
            self.canon_im_tex = self.im_tex.view(b, k, self.tex_channels, h, w).mean(1)  # bx32xHxW
            self.canon_im_tex = self.netT_conv(self.canon_im_tex).repeat_interleave(k, dim=0)  # b*kx32xHxW
            self.im_tex = torch.cat([self.im_tex, self.im_tex.flip(3)], 0)  # flip
            self.canon_im_tex = torch.cat([self.canon_im_tex, self.canon_im_tex.flip(3)], 0)  # flip

            ## 3D Physical Process
            # multi-image-shading (shading, texture)
            self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth)  # B*k xHxWx3
            self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(
                min=0).unsqueeze(1)
            canon_shading = self.canon_light_a.view(-1, 1, 1, 1) + self.canon_light_b.view(-1, 1, 1,
                                                                                 1) * self.canon_diffuse_shading
            self.shaded_texture = canon_shading * self.im_tex  # B*Kx32xHxW
            self.shaded_canon_texture = canon_shading * self.canon_im_tex  # B*kx32xHxW

            # grid sampling
            self.renderer.set_transform_matrices(self.view)
            self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
            self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)
            grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
            self.recon_im_tex = nn.functional.grid_sample(self.shaded_texture, grid_2d_from_canon, mode='bilinear').unsqueeze(0)
            self.recon_canon_im_tex = nn.functional.grid_sample(self.shaded_canon_texture, grid_2d_from_canon,
                                                            mode='bilinear').unsqueeze(0)

            self.fused_im_tex = torch.cat([self.recon_im_tex, self.recon_canon_im_tex], dim=2).squeeze(0)
            self.fused_im_tex = self.netF_conv(self.fused_im_tex)

            ## Neural Appearance Renderer
            self.recon_im = self.netN(self.fused_im_tex)
            margin = (self.max_depth - self.min_depth) / 2
            recon_im_mask = (self.recon_depth < self.max_depth + margin).float()  # invalid border pixels have been clamped at max_depth+margin

            recon_im_mask = recon_im_mask.unsqueeze(1).detach()
            self.recon_im = self.recon_im * recon_im_mask

            if mode == 'discriminator':
                generated_im = self.recon_im
                fake_output = self.discriminator(generated_im.detach())
                real_output = self.discriminator(self.input_im)

                # get discriminator loss
                real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
                self.loss_d = (real_loss + fake_loss) * self.lam_adv
                return
            else:
                ## predict confidence map
                if self.use_conf_map:
                    conf_sigma_l1, conf_sigma_percl = self.netC(self.input_im)  # B*kx1xHxW
                    self.conf_sigma_l1 = conf_sigma_l1[:, :1]
                    self.conf_sigma_l1_flip = conf_sigma_l1[:, 1:]
                    self.conf_sigma_percl = conf_sigma_percl[:, :1]
                    self.conf_sigma_percl_flip = conf_sigma_percl[:, 1:]
                else:
                    self.conf_sigma_l1 = None

                ## rotated image (not sure..)
                random_a = torch.normal(mean=1.39e-01, std=0.00317, size=(b*k, 1)).to(self.input_im.device) * math.pi / 180 * self.xyz_rotation_range
                random_b = torch.normal(mean=3.72e-03, std=0.0462, size=(b*k, 1)).to(self.input_im.device) * math.pi / 180 * self.xyz_rotation_range
                random_c = torch.normal(mean=4.781e-03, std=0.0149, size=(b*k, 1)).to(self.input_im.device) * math.pi / 180 * self.xyz_rotation_range
                random_d = torch.normal(mean=1.415e-03, std=0.0071, size=(b*k, 1)).to(self.input_im.device) * self.xyz_rotation_range
                random_e = torch.normal(mean=9.527e-02, std=0.0039, size=(b*k, 1)).to(self.input_im.device) * self.xyz_rotation_range
                random_f = torch.normal(mean=2.681e-41, std=0. , size=(b*k, 1)).to(self.input_im.device) * self.xyz_rotation_range

                self.view_rot = torch.cat([
                    random_a, random_b, random_c, random_d, random_e, random_f], 1
                ).repeat(2,1)

                # random_R = torch.rand(3).to(self.input_im.device)
                # random_R = random_R * math.pi / 180 * self.xyz_rotation_range
                # self.view_rot = torch.cat([
                #     self.view[:, :3] + random_R,
                #     self.view[:, 3:5],
                #     self.view[:, 5:]], 1)  # b*kx6
                self.renderer.set_transform_matrices(self.view_rot)
                self.recon_depth_rotate = self.renderer.warp_canon_depth(self.canon_depth)
                grid_2d_from_canon_rotate = self.renderer.get_inv_warped_2d_grid(self.recon_depth_rotate)
                self.recon_im_tex_rotate = nn.functional.grid_sample(self.shaded_texture, grid_2d_from_canon_rotate,
                                                                mode='bilinear').unsqueeze(0)
                self.recon_canon_im_tex_rotate = nn.functional.grid_sample(self.shaded_canon_texture, grid_2d_from_canon_rotate,
                                                                       mode='bilinear').unsqueeze(0)

                self.fused_im_tex_rotate = torch.cat([self.recon_im_tex_rotate, self.recon_canon_im_tex_rotate], dim=2).squeeze(0)
                self.fused_im_tex_rotate = self.netF_conv(self.fused_im_tex_rotate)
                self.recon_im_rotate = self.netN(self.fused_im_tex_rotate)
                recon_im_mask_rotate = (
                        self.recon_depth_rotate < self.max_depth + margin).float()  # invalid border pixels have been clamped at max_depth+margin
                recon_im_mask_rotate = recon_im_mask_rotate.unsqueeze(1).detach()
                self.recon_im_rotate = self.recon_im_rotate * recon_im_mask_rotate

                ## loss function
                self.loss_recon = self.photometric_loss(self.recon_im[:b*k], self.input_im, mask=recon_im_mask[:b*k], conf_sigma=self.conf_sigma_l1)
                self.loss_recon_flip = self.photometric_loss(self.recon_im[b*k:], self.input_im, mask=recon_im_mask[b*k:], conf_sigma=self.conf_sigma_l1_flip)
                # self.loss_perc_im = self.perceptual_loss(self.recon_im[:b*k], self.input_im, mask=recon_im_mask[:b*k], conf_sigma=self.conf_sigma_percl)
                # self.loss_perc_im_flip = self.perceptual_loss(self.recon_im[b*k:], self.input_im, mask=recon_im_mask[b*k:], conf_sigma=self.conf_sigma_percl_flip)
                # self.loss_g = self.generator_loss(self.discriminator(self.recon_im))
                # self.loss_adv = self.loss_g + self.loss_d
                self.loss_tex = (self.netT(self.recon_im_rotate[:b*k].detach()) - self.netT(self.input_im)).abs().mean()
                self.loss_shape = (self.netD(self.recon_im_rotate[:b*k].detach()) - self.netD(self.input_im)).abs().mean()
                self.loss_light = (self.netL(self.recon_im_rotate[:b*k].detach()) - self.netL(self.input_im)).abs().mean()
                # self.loss_total += self.loss_recon + self.lam_shape * self.loss_shape + self.lam_adv * self.loss_g \
                #                    + self.lam_tex * self.loss_tex + self.lam_light * self.loss_light
                # self.loss_total += self.loss_recon + self.lam_flip*self.loss_recon_flip + self.lam_perc*(self.loss_perc_im + self.lam_flip*self.loss_perc_im_flip) \
                #                    + self.lam_shape * self.loss_shape + self.lam_tex * self.loss_tex + self.lam_light * self.loss_light
                self.loss_total += self.loss_recon + self.lam_flip * self.loss_recon_flip + \
                                   self.lam_shape * self.loss_shape + self.lam_tex * self.loss_tex + self.lam_light * self.loss_light

        metrics = {'loss': self.loss_total}

        ## compute accuracy if gt depth is available
        if self.load_gt_depth:
            self.depth_gt = depth_gt[:,0,:,:].to(self.input_im.device)
            self.depth_gt = (1-self.depth_gt)*2-1
            self.depth_gt = self.depth_rescaler(self.depth_gt)
            self.normal_gt = self.renderer.get_normal_from_depth(self.depth_gt)

            # mask out background
            mask_gt = (self.depth_gt<self.depth_gt.max()).float()
            mask_gt = (nn.functional.avg_pool2d(mask_gt.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float()  # erode by 1 pixel
            mask_pred = (nn.functional.avg_pool2d(recon_im_mask[:b].unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float()  # erode by 1 pixel
            mask = mask_gt * mask_pred
            self.acc_mae_masked = ((self.recon_depth[:b] - self.depth_gt[:b]).abs() *mask).view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            self.acc_mse_masked = (((self.recon_depth[:b] - self.depth_gt[:b])**2) *mask).view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            self.sie_map_masked = utils.compute_sc_inv_err(self.recon_depth[:b].log(), self.depth_gt[:b].log(), mask=mask)
            self.acc_sie_masked = (self.sie_map_masked.view(b,-1).sum(1) / mask.view(b,-1).sum(1))**0.5
            self.norm_err_map_masked = utils.compute_angular_distance(self.recon_normal[:b], self.normal_gt[:b], mask=mask)
            self.acc_normal_masked = self.norm_err_map_masked.view(b,-1).sum(1) / mask.view(b,-1).sum(1)

            metrics['SIE_masked'] = self.acc_sie_masked.mean()
            metrics['NorErr_masked'] = self.acc_normal_masked.mean()

        return metrics

    def save_results(self, save_dir):
        if self.K is None:
            k, c, h, w = self.input_im.shape

            with torch.no_grad():
                v0 = torch.FloatTensor([-0.1*math.pi/180*60,0,0,0,0,0]).to(self.input_im.device).repeat(k,1)
                # canon_im_rotate = self.renderer.render_yaw(self.canon_im, self.canon_depth, v_before=v0,
                #                                            maxr=90, nsample=15)  # (B,T,C,H,W)
                # canon_im_rotate = canon_im_rotate.clamp(-1, 1).detach().cpu() / 2 + 0.5
                canon_normal_rotate = self.renderer.render_yaw(self.canon_normal.permute(0,3,1,2), self.canon_depth, v_before=v0, maxr=90, nsample=15)  # (B,T,C,H,W)
                canon_normal_rotate = canon_normal_rotate.clamp(-1,1).detach().cpu() /2+0.5

            input_im = self.input_im.detach().cpu().numpy() /2+0.5
            recon_im = self.recon_im.clamp(-1,1).detach().cpu().numpy() /2+0.5
            canon_depth = ((self.canon_depth -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
            recon_depth = ((self.recon_depth -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
            canon_diffuse_shading = self.canon_diffuse_shading.detach().cpu().numpy()
            canon_normal = self.canon_normal.permute(0,3,1,2).detach().cpu().numpy() /2+0.5
            recon_normal = self.recon_normal.permute(0,3,1,2).detach().cpu().numpy() /2+0.5
            im_tex = self.canon_im_tex.detach().cpu().numpy() /2+0.5
            recon_im_tex = self.recon_im_tex.clamp(-1,1).detach().cpu().numpy() /2+0.5


            if self.use_conf_map:
                conf_map_l1 = 1/(1+self.conf_sigma_l1.detach().cpu().numpy()+EPS)
            canon_light = torch.cat([self.canon_light_a, self.canon_light_b, self.canon_light_d], 1).detach().cpu().numpy()
            view = self.view.detach().cpu().numpy()

            # canon_im_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(k ** 0.5))) for img in
            #                         torch.unbind(canon_im_rotate, 1)]  # [(C,H,W)]*T
            # canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)
            canon_normal_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(k**0.5))) for img in torch.unbind(canon_normal_rotate,1)]  # [(C,H,W)]*T
            canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)

            sep_folder = True
            utils.save_images(save_dir, input_im, suffix='input_image', sep_folder=sep_folder)
            utils.save_images(save_dir, recon_im, suffix='recon_image', sep_folder=sep_folder)
            utils.save_images(save_dir, canon_depth, suffix='canonical_depth', sep_folder=sep_folder)
            utils.save_images(save_dir, recon_depth, suffix='recon_depth', sep_folder=sep_folder)
            utils.save_images(save_dir, canon_diffuse_shading, suffix='canonical_diffuse_shading', sep_folder=sep_folder)
            utils.save_images(save_dir, canon_normal, suffix='canonical_normal', sep_folder=sep_folder)
            utils.save_images(save_dir, recon_normal, suffix='recon_normal', sep_folder=sep_folder)
            if self.use_conf_map:
                utils.save_images(save_dir, conf_map_l1, suffix='conf_map_l1', sep_folder=sep_folder)
            utils.save_txt(save_dir, canon_light, suffix='canonical_light', sep_folder=sep_folder)
            utils.save_txt(save_dir, view, suffix='viewpoint', sep_folder=sep_folder)

            # utils.save_videos(save_dir, canon_im_rotate_grid, suffix='image_video', sep_folder=sep_folder, cycle=True)
            utils.save_videos(save_dir, canon_normal_rotate_grid, suffix='normal_video', sep_folder=sep_folder, cycle=True)
        else:
            n, c, h, w = self.input_im.shape

            with torch.no_grad():
                v0 = torch.FloatTensor([-0.1*math.pi/180*60,0,0,0,0,0]).to(self.input_im.device).repeat(n,1)
                # canon_im_rotate = self.renderer.render_yaw(self.canon_im, self.canon_depth, v_before=v0,
                #                                            maxr=90, nsample=15)  # (B,T,C,H,W)
                # canon_im_rotate = canon_im_rotate.clamp(-1, 1).detach().cpu() / 2 + 0.5
                canon_normal_rotate = self.renderer.render_yaw(self.canon_normal.permute(0,3,1,2), self.canon_depth, v_before=v0, maxr=90, nsample=15)  # (B,T,C,H,W)
                canon_normal_rotate = canon_normal_rotate.clamp(-1,1).detach().cpu() /2+0.5

            input_im = self.input_im.detach().cpu().numpy() /2+0.5
            recon_im = self.recon_im.clamp(-1,1).detach().cpu().numpy() /2+0.5
            canon_depth = ((self.canon_depth -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
            recon_depth = ((self.recon_depth -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
            canon_diffuse_shading = self.canon_diffuse_shading.detach().cpu().numpy()
            canon_normal = self.canon_normal.permute(0,3,1,2).detach().cpu().numpy() /2+0.5
            recon_normal = self.recon_normal.permute(0,3,1,2).detach().cpu().numpy() /2+0.5
            if self.use_conf_map:
                conf_map_l1 = 1/(1+self.conf_sigma_l1.detach().cpu().numpy()+EPS)
            canon_light = torch.cat([self.canon_light_a, self.canon_light_b, self.canon_light_d], 1).detach().cpu().numpy()
            view = self.view.detach().cpu().numpy()

            # canon_im_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(k ** 0.5))) for img in
            #                         torch.unbind(canon_im_rotate, 1)]  # [(C,H,W)]*T
            # canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)
            canon_normal_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(n**0.5))) for img in torch.unbind(canon_normal_rotate,1)]  # [(C,H,W)]*T
            canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)

            sep_folder = True
            utils.save_images(save_dir, input_im, suffix='input_image', sep_folder=sep_folder)
            utils.save_images(save_dir, recon_im, suffix='recon_image', sep_folder=sep_folder)
            utils.save_images(save_dir, canon_depth, suffix='canonical_depth', sep_folder=sep_folder)
            utils.save_images(save_dir, recon_depth, suffix='recon_depth', sep_folder=sep_folder)
            utils.save_images(save_dir, canon_diffuse_shading, suffix='canonical_diffuse_shading', sep_folder=sep_folder)
            utils.save_images(save_dir, canon_normal, suffix='canonical_normal', sep_folder=sep_folder)
            utils.save_images(save_dir, recon_normal, suffix='recon_normal', sep_folder=sep_folder)
            if self.use_conf_map:
                utils.save_images(save_dir, conf_map_l1, suffix='conf_map_l1', sep_folder=sep_folder)
            utils.save_txt(save_dir, canon_light, suffix='canonical_light', sep_folder=sep_folder)
            utils.save_txt(save_dir, view, suffix='viewpoint', sep_folder=sep_folder)

            # utils.save_videos(save_dir, canon_im_rotate_grid, suffix='image_video', sep_folder=sep_folder, cycle=True)
            utils.save_videos(save_dir, canon_normal_rotate_grid, suffix='normal_video', sep_folder=sep_folder, cycle=True)

    def save_scores(self, path):
        # save scores if gt is loaded
        if self.load_gt_depth:
            header = 'MAE_masked, \
                      MSE_masked, \
                      SIE_masked, \
                      NorErr_masked'
            mean = self.all_scores.mean(0)
            std = self.all_scores.std(0)
            header = header + '\nMean: ' + ',\t'.join(['%.8f'%x for x in mean])
            header = header + '\nStd: ' + ',\t'.join(['%.8f'%x for x in std])
            utils.save_scores(path, self.all_scores, header=header)

    def visualize(self, logger, total_iter, max_bs=25):
        k, c, h, w = self.input_im.shape
        b0 = min(max_bs, k)

        with torch.no_grad():
            v0 = torch.FloatTensor([-0.1*math.pi/180*60,0,0,0,0,0]).to(self.input_im.device).repeat(k,1)
            # canon_im_rotate = self.renderer.render_yaw(self.recon_im[:b0], self.canon_depth[:b0], v_before=v0, maxr=90).detach().cpu() /2.+0.5  # (B,T,C,H,W)
            canon_normal_rotate = self.renderer.render_yaw(self.canon_normal[:k].permute(0,3,1,2), self.canon_depth[:k], v_before=v0, maxr=90, nsample=15)  # (B,T,C,H,W)
            canon_normal_rotate = canon_normal_rotate.clamp(-1,1).detach().cpu() /2+0.5

        input_im = self.input_im.detach().cpu() /2+0.5
        recon_im = self.recon_im.clamp(-1,1).detach().cpu() /2+0.5
        recon_im_rotate = self.recon_im_rotate.clamp(-1,1).detach().cpu() /2+0.5

        canon_depth = ((self.canon_depth -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1)
        # canon_depth_raw_hist = self.canon_depth_raw.detach().unsqueeze(1).cpu()
        canon_depth_raw = self.canon_depth_raw.detach().unsqueeze(1).cpu() / 2. + 0.5
        recon_depth = ((self.recon_depth -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1)
        canon_diffuse_shading = self.canon_diffuse_shading.detach().cpu()
        canon_normal = self.canon_normal.permute(0,3,1,2).detach().cpu() /2+0.5
        recon_normal = self.recon_normal.permute(0,3,1,2).detach().cpu() /2+0.5

        recon_depth_rotate = ((self.recon_depth_rotate -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1)

        def texture_viz(x):
            from .utils import pca, tsne
            x = x.permute(0,2,3,1).contiguous().view(-1, self.tex_channels) # (BHW,C)
            x = pca(x, 3) # (BHW,3)
            x = x.view(-1, h, w, 3).permute(0,3,1,2) # (B,3,H,W)
            return x

        im_tex = texture_viz(self.im_tex.detach().cpu() /2+0.5)
        canon_im_tex = texture_viz(self.canon_im_tex.detach().cpu() /2+0.5)
        shaded_texture = texture_viz(self.shaded_texture.detach().cpu() /2+0.5)
        shaded_canon_texture = texture_viz(self.shaded_canon_texture.detach().cpu() /2+0.5)
        recon_im_tex = texture_viz(self.recon_im_tex.squeeze(0).clamp(-1,1).detach().cpu() /2+0.5)
        recon_canon_im_tex = texture_viz(self.recon_canon_im_tex.squeeze(0).clamp(-1,1).detach().cpu() /2+0.5)
        fused_im_tex = texture_viz(self.fused_im_tex.squeeze(0).clamp(-1,1).detach().cpu() /2+0.5)

        if self.use_conf_map:
            conf_map_l1 = 1/(1+self.conf_sigma_l1.detach().cpu()+EPS)
            conf_map_l1_flip = 1/(1+self.conf_sigma_l1_flip.detach().cpu()+EPS)
            conf_map_percl = 1/(1+self.conf_sigma_percl.detach().cpu()+EPS)
            conf_map_percl_flip = 1/(1+self.conf_sigma_percl_flip.detach().cpu()+EPS)
        # canon_light = torch.cat([self.canon_light_a, self.canon_light_b, self.canon_light_d], 1).detach().cpu()
        # view = self.view.detach().cpu()
        # canon_im_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0**0.5))) for img in torch.unbind(canon_im_rotate, 1)]  # [(C,H,W)]*T
        # canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)
        canon_normal_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(k**0.5))) for img in torch.unbind(canon_normal_rotate,1)]  # [(C,H,W)]*T
        canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)

        ## write summary
        logger.add_scalar('Loss/loss_total', self.loss_total, total_iter)
        logger.add_scalar('Loss/loss_recon', self.loss_recon, total_iter)
        logger.add_scalar('Loss/loss_recon_flip', self.loss_recon_flip, total_iter)
        logger.add_scalar('Loss/loss_tex', self.loss_tex, total_iter)
        logger.add_scalar('Loss/loss_shape', self.loss_shape, total_iter)
        logger.add_scalar('Loss/loss_light', self.loss_light, total_iter)
        # logger.add_scalar('Loss/loss_perc_im', self.loss_perc_im, total_iter)
        # logger.add_scalar('Loss/loss_perc_im_flip', self.loss_perc_im_flip, total_iter)
        # logger.add_scalar('Loss/loss_adv', self.loss_adv, total_iter)
        # logger.add_scalar('Loss/loss_g', self.loss_g, total_iter)
        # logger.add_scalar('Loss/loss_d', self.loss_d, total_iter)

        # logger.add_histogram('Depth/canon_depth_raw_hist', canon_depth_raw_hist, total_iter)
        vlist = ['view_rx', 'view_ry', 'view_rz', 'view_tx', 'view_ty', 'view_tz']
        for i in range(self.view.shape[1]):
            logger.add_histogram('View/' + vlist[i], self.view[:, i], total_iter)
        logger.add_histogram('Light/canon_light_a', self.canon_light_a, total_iter)
        logger.add_histogram('Light/canon_light_b', self.canon_light_b, total_iter)
        llist = ['canon_light_dx', 'canon_light_dy', 'canon_light_dz']
        for i in range(self.canon_light_d.shape[1]):
            logger.add_histogram('Light/' + llist[i], self.canon_light_d[:, i], total_iter)

        def log_grid_image(label, im, nrow=int(math.ceil(b0 ** 0.5)), iter=total_iter):
            im_grid = torchvision.utils.make_grid(im, nrow=nrow)
            logger.add_image(label, im_grid, iter)

        log_grid_image('Image/input_image', input_im)
        log_grid_image('Image/recon_image', recon_im[:b0])
        log_grid_image('Image/recon_image_rotate', recon_im_rotate[:b0])
        # log_grid_image('Image/recon_side', canon_im_rotate[:,0,:,:,:])

        log_grid_image('Depth/canonical_depth_raw', canon_depth_raw[:b0])
        log_grid_image('Depth/canonical_depth', canon_depth[:b0])
        log_grid_image('Depth/recon_depth', recon_depth[:b0])
        log_grid_image('Depth/canonical_diffuse_shading', canon_diffuse_shading[:b0])
        log_grid_image('Depth/canonical_normal', canon_normal[:b0])
        log_grid_image('Depth/recon_normal', recon_normal[:b0])
        log_grid_image('Depth/recon_depth_rotate', recon_depth_rotate[:b0])

        log_grid_image('Texture/im_tex', im_tex[:b0])
        log_grid_image('Texture/recon_im_tex', recon_im_tex[:b0])
        log_grid_image('Texture/canonical_im_tex', canon_im_tex[:b0])
        log_grid_image('Texture/recon_canon_im_tex', recon_canon_im_tex[:b0])
        log_grid_image('Texture/shaded_texture', shaded_texture[:b0])
        log_grid_image('Texture/shaded_canon_texture', shaded_canon_texture[:b0])
        log_grid_image('Texture/fused_im_tex', fused_im_tex[:b0])

        # logger.add_histogram('Image/canonical_diffuse_shading_hist', canon_diffuse_shading, total_iter)

        if self.use_conf_map:
            log_grid_image('Conf/conf_map_l1', conf_map_l1)
            log_grid_image('Conf/conf_map_l1_flip', conf_map_l1_flip)
            log_grid_image('Conf/conf_map_perc_im', conf_map_percl)
            log_grid_image('Conf/conf_map_perc_im_flip', conf_map_percl_flip)

            logger.add_histogram('Conf/conf_sigma_l1_hist', self.conf_sigma_l1, total_iter)

        # logger.add_video('Image_rotate/recon_rotate', canon_im_rotate_grid, total_iter, fps=4)
        logger.add_video('Image_rotate/canon_normal_rotate', canon_normal_rotate_grid, total_iter, fps=4)

        if self.load_gt_depth:
            depth_gt = ((self.depth_gt[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
            normal_gt = self.normal_gt.permute(0,3,1,2)[:b0].detach().cpu() /2+0.5
            sie_map_masked = self.sie_map_masked[:b0].detach().unsqueeze(1).cpu() *1000
            norm_err_map_masked = self.norm_err_map_masked[:b0].detach().unsqueeze(1).cpu() /100

            logger.add_scalar('Acc_masked/MAE_masked', self.acc_mae_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/MSE_masked', self.acc_mse_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/SIE_masked', self.acc_sie_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/NorErr_masked', self.acc_normal_masked.mean(), total_iter)

            log_grid_image('Depth_gt/depth_gt', depth_gt)
            log_grid_image('Depth_gt/normal_gt', normal_gt)
            log_grid_image('Depth_gt/sie_map_masked', sie_map_masked)
            log_grid_image('Depth_gt/norm_err_map_masked', norm_err_map_masked)

        # for rendering at canoncial view
        self.fused_canon_tex = torch.cat([self.shaded_texture, self.shaded_canon_texture], dim=1).squeeze(0)
        self.fused_canon_tex = self.netF_conv(self.fused_canon_tex)

        fused_canon_tex = texture_viz(self.fused_canon_tex.squeeze(0).clamp(-1,1).detach().cpu() /2+0.5)
        log_grid_image('Debug/fused_canon_tex', fused_canon_tex[:b0])

        recon_canon_im = self.netN(self.fused_canon_tex)
        recon_canon_im = recon_canon_im.clamp(-1,1).detach().cpu() /2+0.5
        log_grid_image('Debug/recon_canon_im', recon_canon_im[:b0])

    def calc_view_range(self, batch):
        # calculate the view range
        input_im = torch.stack(batch, dim=0).to(self.device) * 2. - 1.  # b, k, 3, h, w
        b, k, c, h, w = input_im.shape
        input_im = input_im.view(b * k, c, h, w)

        v = self.netV(input_im)
        var, mean = torch.var_mean(v, dim=0)
        var, mean = var.cpu().detach().numpy(), mean.cpu().detach().numpy()

        self.view_m.update(value= mean,mass=b*k)
        self.view_v.update(value= var,mass=b*k)
        return self.view_m.get(), self.view_v.get()
