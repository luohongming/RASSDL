from model.Base_model import Base_model
from model.archs.VGG16_arch import VGG16_arch
import torch.nn as nn
import torch


class VGG16_model(Base_model):
    def __init__(self, opt):
        Base_model.initialize(self, opt)
        self.set_model()
        self.set_criterion()

    def set_model(self):
        self.model = VGG16_arch()
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model, self.gpu_ids)

    def set_criterion(self):
        self.criterion = nn.MSELoss().to(self.device)

    def set_input(self, input):
        channel = input.size(1)
        if channel == 1:
            input = torch.cat((input, input, input), 1)
            input = input.to(self.device)
            return input
        elif channel == 3:
            input = input.to(self.device)
            return input
        else:
            exit('VGG16Model inputs have some problems.')

    def cal_loss(self, output, target):
        output = self.model(output)
        target = self.model(target)
        weights = [0.244140625, 0.06103515625, 0.0152587890625]
        loss = 0.0
        for i in range(len(weights)):
            loss += weights[i] * self.criterion(output[i], target[i])
        return loss