from collections import namedtuple
import torch.nn as nn
from torchvision import models


class VGG16_arch(nn.Module):
    def __init__(self):
        super(VGG16_arch, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters(): # fix the parameters
            param.requires_grad = False

    def forward(self, x):
        f = self.slice1(x)
        f_relu1_2 = f
        f = self.slice2(f)
        f_relu2_2 = f
        f = self.slice3(f)
        f_relu3_3 = f
        f = self.slice4(f)
        f_relu4_3 = f
        vgg_outputs = namedtuple('VGG16_Outputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        y = vgg_outputs(f_relu1_2, f_relu2_2, f_relu3_3, f_relu4_3)
        return y
