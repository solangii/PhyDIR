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
        self.K = cfgs.get('K', None)
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
            lr=self.lr, betas=(0.9, 0.999))

        ## other parameters

        ## depth rescaler: -1~1 -> min_depth~max_depth
        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth
        self.amb_light_rescaler = lambda x : (1+x)/2 *self.max_amb_light + (1-x)/2 *self.min_amb_light
        self.diff_light_rescaler = lambda x : (1+x)/2 *self.max_diff_light + (1-x)/2 *self.min_diff_light

    def init_optimizers(self, stage=None):
        if stage is 1:
            self.network_names = ['netC', 'netT', 'netN', 'netT_conv', 'netF_conv']
        elif stage is 2:
            self.network_names = ['netC', 'netD', 'netL', 'netV']
        elif stage is 3:
            self.network_names = ['netC', 'netD', 'netL', 'netV', 'netT', 'netN', 'netT_conv', 'netF_conv']
        self.network_names_not = [k for k in vars(self) if 'net' in k]
        for k in self.network_names:
            self.network_names_not.remove(k)
        self.network_names_not.remove('network_names')

        self.set_requires_grad(self.network_names_not, requires_grad=False)
        self.set_requires_grad(self.network_names, requires_grad=True)

        self.optimizer_names = []
        for net_name in self.network_names:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net','optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]

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
        # if self.other_param_names:
        #     for param_name in self.other_param_names:
        #         setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()
        for net_name in self.network_names_not:
            getattr(self, net_name).eval()

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
        self.loss_total = 0
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
                self.cannon_im_tex = self.im_tex.mean(0, keepdim=True) # 1x32xHxW
                self.canon_im_tex = self.netT_conv(self.cannon_im_tex) # 1x32xHxW

                ## multi-image-shading (shading, texture)
                self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth) # BxHxWx3
                self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(
                    min=0).unsqueeze(1)
                canon_shading = self.canon_light_a.view(-1, 1, 1, 1) + self.canon_light_b.view(-1, 1, 1,
                                                                                               1) * self.canon_diffuse_shading
                self.shaded_texture = canon_shading * self.im_tex # Bx32xHxW
                self.shaded_cannon_texture = canon_shading * self.canon_im_tex # Bx32xHxW

                ## grid sampling
                self.renderer.set_transform_matrices(self.view)
                self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
                self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)
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
                random_view = torch.rand(1, 6).to(self.input_im.device)
                random_view = torch.cat([
                    random_view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
                    random_view[:, 3:5] * self.xy_translation_range,
                    random_view[:, 5:] * self.z_translation_range], 1) # Bx6
                self.renderer.set_transform_matrices(random_view)
                self.recon_depth_rotate = self.renderer.warp_canon_depth(self.canon_depth)
                grid_2d_from_canon_rotate = self.renderer.get_inv_warped_2d_grid(self.recon_depth_rotate)
                self.recon_im_tex_rotate = nn.functional.grid_sample(self.shaded_texture, grid_2d_from_canon_rotate, mode='bilinear').unsqueeze(0)
                self.recon_cannon_im_tex_rotate = nn.functional.grid_sample(self.shaded_cannon_texture, grid_2d_from_canon_rotate, mode='bilinear').unsqueeze(0)

                self.fused_im_tex_rotate = torch.cat([self.recon_im_tex_rotate, self.recon_cannon_im_tex_rotate]).mean(0)
                self.fused_im_tex_rotate = self.netF_conv(self.fused_im_tex_rotate)
                self.recon_im_rotate = self.netN(self.fused_im_tex_rotate)

                ## loss function
                loss_recon = self.photometric_loss(self.recon_im, self.input_im, self.conf_sigma_l1)
                # self.loss_adv =
                loss_tex = (self.netT(self.recon_im_rotate) - self.netT(self.input_im)).abs().mean()
                loss_shape = (self.netD(self.recon_im_rotate) - self.netD(self.input_im)).abs().mean()
                loss_light = (self.netL(self.recon_im_rotate) - self.netL(self.input_im)).abs().mean()
                self.loss_total += loss_recon + self.lam_shape * loss_shape +\
                              self.lam_tex * loss_tex + self.lam_l1 * loss_light
        else:
            self.input_im = torch.stack(batch, dim=0).to(self.device) *2.-1.  # b, k, 3, h, w
            b, k, c, h, w = self.input_im.shape
            self.input_im = self.input_im.view(b * k, c, h, w)

            ## 3D Networks
            # 1. predict depth
            canon_depth_raw = self.netD(self.input_im).squeeze(1) # b*k, 1, h, w
            self.canon_depth = canon_depth_raw - canon_depth_raw.view(b*k, -1).mean(1).view(b*k, 1, 1)
            self.canon_depth = self.canon_depth.tanh()
            self.canon_depth = self.depth_rescaler(self.canon_depth)

            # 2. predict light
            canon_light = self.netL(self.input_im)  # b*k x4
            self.canon_light_a = self.amb_light_rescaler(canon_light[:, :1])  # ambience term, b*kx1
            self.canon_light_b = self.diff_light_rescaler(canon_light[:, 1:2])  # diffuse term, b*kx1
            canon_light_dxy = canon_light[:, 2:]
            self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(b*k, 1).to(self.input_im.device)], 1)
            self.canon_light_d = self.canon_light_d / (
                (self.canon_light_d ** 2).sum(1, keepdim=True)) ** 0.5  # diffuse light direction, b*kx3

            # 3. predict viewpoint
            self.view = self.netV(self.input_im)
            self.view = torch.cat([
                self.view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
                self.view[:, 3:5] * self.xy_translation_range,
                self.view[:, 5:] * self.z_translation_range], 1)  # b*kx6

            ## Texture Networks
            im_tex = self.netT(self.input_im)  # b*kx32xHxW
            canon_im_tex = im_tex.view(b, k, 32, h, w).mean(1)  # bx32xHxW
            canon_im_tex = self.netT_conv(canon_im_tex).repeat_interleave(k, dim=0)  # b*kx32xHxW

            ## 3D Physical Process
            # multi-image-shading (shading, texture)
            self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth)  # B*k xHxWx3
            self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(
                min=0).unsqueeze(1)
            canon_shading = self.canon_light_a.view(-1, 1, 1, 1) + self.canon_light_b.view(-1, 1, 1,
                                                                                 1) * self.canon_diffuse_shading
            shaded_texture = canon_shading * im_tex  # B*Kx32xHxW
            shaded_cannon_texture = canon_shading * canon_im_tex  # B*kx32xHxW

            # grid sampling
            self.renderer.set_transform_matrices(self.view)
            self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
            self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)

            grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
            self.recon_im_tex = nn.functional.grid_sample(shaded_texture, grid_2d_from_canon, mode='bilinear').unsqueeze(0)
            self.recon_cannon_im_tex = nn.functional.grid_sample(shaded_cannon_texture, grid_2d_from_canon,
                                                            mode='bilinear').unsqueeze(0)

            self.fused_im_tex = torch.cat([self.recon_im_tex, self.recon_cannon_im_tex]).mean(0)
            self.fused_im_tex = self.netF_conv(self.fused_im_tex)

            ## Neural Appearance Renderer
            self.recon_im = self.netN(self.fused_im_tex)

            ## predict confidence map
            if self.use_conf_map:
                self.conf_sigma_l1 = self.netC(self.input_im)  # B*kx1xHxW
            else:
                self.conf_sigma_l1 = None

            ## rotated image (not sure..)
            random_view = torch.rand(1, 6).to(self.input_im.device)
            random_view = torch.cat([
                random_view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
                random_view[:, 3:5] * self.xy_translation_range,
                random_view[:, 5:] * self.z_translation_range], 1)  # Bx6
            self.renderer.set_transform_matrices(random_view)
            self.recon_depth_rotate = self.renderer.warp_canon_depth(self.canon_depth)
            grid_2d_from_canon_rotate = self.renderer.get_inv_warped_2d_grid(self.recon_depth_rotate)
            self.recon_im_tex_rotate = nn.functional.grid_sample(shaded_texture, grid_2d_from_canon_rotate,
                                                            mode='bilinear').unsqueeze(0)
            self.recon_cannon_im_tex_rotate = nn.functional.grid_sample(shaded_cannon_texture, grid_2d_from_canon_rotate,
                                                                   mode='bilinear').unsqueeze(0)

            self.fused_im_tex_rotate = torch.cat([self.recon_im_tex_rotate, self.recon_cannon_im_tex_rotate]).mean(0)
            self.fused_im_tex_rotate = self.netF_conv(self.fused_im_tex_rotate)
            self.recon_im_rotate = self.netN(self.fused_im_tex_rotate)

            ## loss function
            loss_recon = self.photometric_loss(self.recon_im, self.input_im, self.conf_sigma_l1)
            # self.loss_adv =
            loss_tex = (self.netT(self.recon_im_rotate) - self.netT(self.input_im)).abs().mean()
            loss_shape = (self.netD(self.recon_im_rotate) - self.netD(self.input_im)).abs().mean()
            loss_light = (self.netL(self.recon_im_rotate) - self.netL(self.input_im)).abs().mean()
            self.loss_total += loss_recon + self.lam_shape * loss_shape + \
                               self.lam_tex * loss_tex + self.lam_l1 * loss_light

        metrics = {'loss': self.loss_total}
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

    def save_score(self):
        pass

    def visualize(self):
        pass

