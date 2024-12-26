import torch
from mamba_ssm import Mamba
import numpy as np
from torch import nn
from thop import profile
from model.mamba_sys import SS2D
from model.Unet.vgg import VGG16
import math


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Encoder(nn.Module):
    """
    for input size of (B, 3, 512, 512)
    output size is: feat1, feat2, feat3, feat4, feat5

    torch.Size([1, 64, 512, 512])
    torch.Size([1, 128, 256, 256])
    torch.Size([1, 256, 128, 128])
    torch.Size([1, 512, 64, 64])
    torch.Size([1, 512, 32, 32])
    """

    def __init__(self, in_channel, pretrain=False):
        super(Encoder, self).__init__()
        self.backbone = VGG16(pretrained=pretrain, in_channels=in_channel)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.backbone(x)

        return feat1, feat2, feat3, feat4, feat5


class Adaptive_global_filter(nn.Module):
    def __init__(self, ratio=10, dim=32, H=512, W=512):
        super().__init__()
        self.ratio = ratio
        self.filter = nn.Parameter(torch.randn(dim, H, W, 2, dtype=torch.float32), requires_grad=True)
        self.mask_low = nn.Parameter(data=torch.zeros(size=(H, W)), requires_grad=False)
        self.mask_high = nn.Parameter(data=torch.ones(size=(H, W)), requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        crow, ccol = int(h / 2), int(w / 2)

        mask_lowpass = self.mask_low
        mask_lowpass[crow - self.ratio:crow + self.ratio, ccol - self.ratio:ccol + self.ratio] = 1

        mask_highpass = self.mask_high
        mask_highpass[crow - self.ratio:crow + self.ratio, ccol - self.ratio:ccol + self.ratio] = 0

        x_fre = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm='ortho'))
        weight = torch.view_as_complex(self.filter)

        x_fre_low = torch.mul(x_fre, mask_lowpass)
        x_fre_high = torch.mul(x_fre, mask_highpass)

        x_fre_low = torch.mul(x_fre_low, weight)
        x_fre_new = x_fre_low + x_fre_high
        x_out = torch.fft.ifft2(torch.fft.ifftshift(x_fre_new, dim=(-2, -1))).real
        return x_out


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class MPCA(nn.Module):
    def __init__(self, input_channel1=128, input_channel2=64, gamma=2, bias=1):
        super(MPCA, self).__init__()
        self.input_channel1 = input_channel1
        self.input_channel2 = input_channel2

        self.avg1 = nn.AdaptiveAvgPool2d(1)
        self.avg2 = nn.AdaptiveAvgPool2d(1)

        kernel_size1 = int(abs((math.log(input_channel1, 2) + bias) / gamma))
        kernel_size1 = kernel_size1 if kernel_size1 % 2 else kernel_size1 + 1

        kernel_size2 = int(abs((math.log(input_channel2, 2) + bias) / gamma))
        kernel_size2 = kernel_size2 if kernel_size2 % 2 else kernel_size2 + 1

        kernel_size3 = int(abs((math.log(input_channel1 + input_channel2, 2) + bias) / gamma))
        kernel_size3 = kernel_size3 if kernel_size3 % 2 else kernel_size3 + 1

        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size1, padding=(kernel_size1 - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size2, padding=(kernel_size2 - 1) // 2, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_size3, padding=(kernel_size3 - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.up = nn.ConvTranspose2d(in_channels=input_channel2, out_channels=input_channel1, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)

    def forward(self, x1, x2):
        x1_ = self.avg1(x1)
        x2_ = self.avg2(x2)

        x1_ = self.conv1(x1_.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x2_ = self.conv2(x2_.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        x_middle = torch.cat((x1_, x2_), dim=1)
        x_middle = self.conv3(x_middle.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x_middle = self.sigmoid(x_middle)

        x_1, x_2 = torch.split(x_middle, [self.input_channel1, self.input_channel2], dim=1)

        x1_out = x1 * x_1
        x2_out = x2 * x_2

        x2_out = self.up(x2_out)

        result = x1_out + x2_out
        return result


class SpatialAttention(nn.Module):  # Spatial Attention Module
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        result = x * out
        return result


class FSA(nn.Module):
    def __init__(self, ratio=10, input_channel=64, size=512):
        super(FSA, self).__init__()
        self.agf = Adaptive_global_filter(ratio=ratio, dim=input_channel, H=size, W=size)
        self.sa = SpatialAttention()

    def forward(self, x):
        f_out = self.agf(x)
        sa_out = self.sa(x)
        result = f_out + sa_out
        return result


class skip_connection(nn.Module):
    def __init__(self, ratio=10, input_channel1=64, input_channel2=64, size=512):
        super(skip_connection, self).__init__()
        self.MPCA = MPCA(input_channel1=input_channel1, input_channel2=input_channel2)
        self.FSA = FSA(ratio=ratio, input_channel=input_channel1, size=size)

    def forward(self, x1, x2):
        MPCA = self.MPCA(x1, x2)
        FSA = self.FSA(ASCA)
        return FSA


class segmentation_head(nn.Module):
    def __init__(self, in_channel=64, num_classes=2):
        super(segmentation_head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3,
                               padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out


class SF_UNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=2, dims=[64, 128, 256, 512], sizes=[512, 256, 128, 64], *args,
                 **kwargs):
        super(SF_UNet, self).__init__()
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(in_channel=in_channel)

        self.decoder4 = unetUp(in_size=dims[3] * 2, out_size=dims[3])
        self.decoder3 = unetUp(in_size=dims[2] + dims[3], out_size=dims[2])
        self.decoder2 = unetUp(in_size=dims[1] + dims[2], out_size=dims[1])
        self.decoder1 = unetUp(in_size=dims[0] + dims[1], out_size=dims[0])

        self.skip4 = skip_connection(input_channel1=dims[3], input_channel2=dims[3], size=sizes[3])
        self.skip3 = skip_connection(input_channel1=dims[2], input_channel2=dims[3], size=sizes[2])
        self.skip2 = skip_connection(input_channel1=dims[1], input_channel2=dims[2], size=sizes[1])
        self.skip1 = skip_connection(input_channel1=dims[0], input_channel2=dims[1], size=sizes[0])

        self.seg_head = segmentation_head(in_channel=dims[0], num_classes=num_classes)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.encoder(x)

        skip4 = self.skip4(feat4, feat5)
        decoder4 = self.decoder4(skip4, feat5)

        skip3 = self.skip3(feat3, feat4)
        decoder3 = self.decoder3(skip3, decoder4)

        skip2 = self.skip2(feat2, feat3)
        decoder2 = self.decoder2(skip2, decoder3)

        skip1 = self.skip1(feat1, feat2)
        decoder1 = self.decoder1(skip1, decoder2)

        out = self.seg_head(decoder1)

        return out


def SF_Unet_512_512_3(input_channel=3, num_classes=2):
    net = SF_UNet(
        in_channel=input_channel,
        num_classes=num_classes,
        dims=[64, 128, 256, 512],
        sizes=[224, 112, 56, 28])
    return net


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    a = torch.rand(1, 3, 224, 224).to(device)
    # c = torch.rand(8, 128, 256, 256).to(device)

    print('==> Building model..')
    model = SF_Unet_512_512_3(input_channel=3, num_classes=2)
    model.to(device)
    print('==> Down..')

    flops, params = profile(model, (a,))
    print('Gflops: %.2f G, params: %.2f M' % ((flops / 1000000.0) / 1024.0, params / 1000000.0))

    b = model(a)
    print(b.shape)
    # b, c, d, e, f = model(a)
    # print(b.shape)
    # print(c.shape)
    # print(d.shape)
    # print(e.shape)
    # print(f.shape)
