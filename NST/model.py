import torch
import torch.nn as nn


def gram(x):
    b, ch, h, w = x.shape
    x = x.view(b, ch, w * h)
    x_t = x.transpose(1, 2)

    return x.bmm(x_t) / (ch * h * w)


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=False, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.downsample = downsample
        if self.downsample:
            self.residual_layer = nn.Conv2d(inplanes, planes * self.expansion,
                                         kernel_size=1, stride=stride)

        self.conv_block = nn.Sequential(
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            ConvLayer(planes, planes, kernel_size=3, stride=stride),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)
        )

    def forward(self, x):
        residual = self.residual_layer(x) if self.downsample else x
        return residual + self.conv_block(x)


class UpBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpBottleneck, self).__init__()
        self.expansion = 4
        self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion,
                                                kernel_size=1, stride=1, upsample=stride)

        self.conv_block = nn.Sequential(
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return self.residual_layer(x) + self.conv_block(x)


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        self.reflection_padding = kernel_size // 2
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class Inspiration(nn.Module):

    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=False)
        self.G = torch.Tensor(B, C, C)
        self.C = C
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def set_style(self, target):
        self.G = target

    def forward(self, X):
        self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
        return torch.bmm(self.P.transpose(1, 2).expand(X.size(0), self.C, self.C),
                         X.view(X.size(0), X.size(1), -1)).view_as(X)


class Net(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=128, norm_layer=nn.InstanceNorm2d, n_blocks=6):
        super(Net, self).__init__()

        expansion = 4

        self.model1 = nn.Sequential(
            ConvLayer(input_nc, 64, kernel_size=7, stride=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            Bottleneck(64, 32, 2, 1, norm_layer),
            Bottleneck(32 * expansion, ngf, 2, 1, norm_layer)
        )
        self.ins = Inspiration(ngf * expansion)

        self.model = nn.Sequential(
            self.model1,
            self.ins
        )
        for i in range(n_blocks):
            self.model.add_module(f'{i + 2}', Bottleneck(ngf * expansion, ngf, 1, False, norm_layer))

        self.model.add_module('8', UpBottleneck(ngf * expansion, 32, 2, norm_layer))
        self.model.add_module('9', UpBottleneck(32 * expansion, 16, 2, norm_layer))
        self.model.add_module('10', norm_layer(16 * expansion))
        self.model.add_module('11', nn.ReLU(inplace=True))
        self.model.add_module('12', ConvLayer(16 * expansion, output_nc, kernel_size=7, stride=1))

    def set_style(self, Xs):
        F = self.model1(Xs)
        G = gram(F)
        self.ins.set_style(G)

    def forward(self, input):
        return self.model(input)