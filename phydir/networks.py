import torch
import torch.nn as nn
import torchvision

EPS = 1e-7


class Encoder(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, cout, kernel_size=1, stride=1, padding=0, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0),-1)


class EDDeconv(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64, activation=nn.Tanh):
        super(EDDeconv, self).__init__()
        ## downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        ## upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class ConfNet(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64):
        super(ConfNet, self).__init__()
        ## downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        ## upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True)]
        self.network = nn.Sequential(*network)

        out_net1 = [
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 64x64
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 2, kernel_size=5, stride=1, padding=2, bias=False),  # 64x64
            nn.Softplus()]
        self.out_net1 = nn.Sequential(*out_net1)

        out_net2 = [nn.Conv2d(nf*2, 2, kernel_size=3, stride=1, padding=1, bias=False),  # 16x16
                    nn.Softplus()]
        self.out_net2 = nn.Sequential(*out_net2)

    def forward(self, input):
        out = self.network(input)
        return self.out_net1(out), self.out_net2(out)


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.randn(1, 3, 256, 256).to(device)

    # depth network
    netD = EDDeconv(cin=3, cout=1, nf=256, zdim=512, activation=None).to(device) # original params: nf->64, zdim->256
    summary(netD, (3, 256, 256))
    output = netD(input)
    print(output.shape)

    # view network
    netV = Encoder(cin=3, cout=6, nf=32).to(device)
    summary(netV, (3, 256, 256))
    output = netV(input)
    print(output.shape)

    # light network
    netL = Encoder(cin=3, cout=4, nf=32).to(device)
    summary(netL, (3, 256, 256))
    output = netL(input)
    print(output.shape)
