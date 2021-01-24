import torchvision.models as models

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

imsize = 256

transformer = transforms.Compose(
    [
     transforms.Resize(imsize),
     transforms.CenterCrop(imsize),
     transforms.ToTensor()
    ]
)


def gram_matrix(input):
    batch_size, h, w, f_map_num = input.size()
    features = input.view(batch_size * h, w * f_map_num)
    G = torch.mm(features, features.t())

    return G.div(batch_size * h * w * f_map_num)


class StyleLoss(nn.Module):

  def __init__(self):
    super(StyleLoss, self).__init__()

    # ADD TARGET TO ARGUMENTS

    self.target = None
    self.loss = None

    #self.target = gram_matrix(target).detach()
    #self.loss = F.mse_loss(self.target, self.target)


  def forward(self, style):
    self.loss = F.mse_loss(gram_matrix(style), self.target)
    return style


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

        self.target = None
        self.loss = None

        # self.target = target.detach()
        # self.loss = F.mse_loss(self.target, self.target)

    def forward(self, content):
        self.loss = F.mse_loss(content, self.target)
        return content


class Normalization(nn.Module):

  def __init__(self):
    super(Normalization, self).__init__()
    self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

  def forward(self, img):

    return (img - self.mean) / self.std




model = nn.Sequential(Normalization())
model.add_module('conv_1', nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
model.add_module('style_loss_1', StyleLoss())
model.add_module('relu_1', nn.ReLU(inplace=False))
model.add_module('conv_2', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
model.add_module('style_loss_2', StyleLoss())
model.add_module('relu_2', nn.ReLU(inplace=False))
model.add_module('pool_2', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
model.add_module('conv_3', nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
model.add_module('style_loss_3', StyleLoss())
model.add_module('relu_3', nn.ReLU(inplace=False))
model.add_module('conv_4', nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
model.add_module('content_loss_4', ContentLoss())
model.add_module('relu_4', nn.ReLU(inplace=False))
model.add_module('pool_4', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
model.add_module('conv_5', nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
model.add_module('style_loss_5', StyleLoss())

model.load_state_dict(torch.load('model_nst.pt'))
model.eval()
