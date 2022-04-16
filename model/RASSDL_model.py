from model.Base_model import Base_model
from model.archs.RASSDL_arch import RASSDL_arch
import torch
import torch.nn as nn
import torch.optim as optim
from model.loss import VGG16_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import cal_k, cal_tao, cal_histogram


class RASSDL_model(Base_model):
    def __init__(self, opt):
        Base_model.initialize(self, opt)
        self.opt = opt
        self.n_dense = opt.n_dense
        self.nf = opt.nf
        self.lr = opt.lr
        self.nEpochs = opt.nEpochs
        self.eta_min = opt.eta_min
        self.bins_num = opt.bins_num
        self.pretrain_model_path = opt.pretrain_model_path
        self.network_label = opt.network_label
        self.set_model()

    def set_mode(self, train=True, save_path=None):
        if save_path == '':
            save_path = None
        if train:
            if not save_path == None:
                self.load(save_path)

                # change to use cpu
                gpu_model = self.model.module
                self.model = RASSDL_arch(self.n_dense, self.nf)
                self.model.load_state_dict(gpu_model.state_dict())
            self.model.to(self.device)

            self.model.train()
            self.set_criterion()
            self.set_optimizer()
            self.set_scheduler()
        else:
            self.load(save_path)

            # change to use cpu
            gpu_model = self.model.module
            self.model = RASSDL_arch(self.n_dense, self.nf)
            self.model.load_state_dict(gpu_model.state_dict())
            self.model.to(self.device)

            self.model.eval()

    def set_model(self):
        self.model = RASSDL_arch(self.n_dense, self.nf)
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model, self.gpu_ids)

    def set_criterion(self):
        self.criterion1 = VGG16_loss(self.opt).to(self.device)
        # self.criterion1 = nn.MSELoss().to(self.device)

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizers.append(self.optimizer)

    def set_scheduler(self):
        self.scheduler = CosineAnnealingLR(self.optimizer, self.nEpochs, self.eta_min)
        self.schedulers.append(self.scheduler)

    def set_input(self, data):
        self.input = data['input'].to(self.device)
        self.target = data['target'].to(self.device)

    def set_eval_input(self, hdr_y):
        max = 1
        min = 0
        k = cal_k(hdr_y)
        tao = cal_tao(hdr_y, k)

        hdr_y_tao = hdr_y + tao + 1e-8
        log_y = hdr_y_tao.log().sub(hdr_y_tao.min().log()).div(hdr_y_tao.max().log().sub(hdr_y_tao.min().log())).mul(
            max - min).add(min)
        self.input = log_y.to(self.device)

    def set_target(self):
        self.target_vgg = self.input
        self.target_vgg_equlized = self.target
        his_linear, his_equlized = cal_histogram(self.input, self.bins_num)
        his_linear = torch.cat(his_linear, 0)
        his_equlized = torch.cat(his_equlized, 0)
        self.his_linear = his_linear.to(self.device)
        self.his_equlized = his_equlized.to(self.device)

    def cal_loss(self):
        self.output = self.model(self.input)
        self.loss_vgg = 0.0
        self.loss_vgg_equlized = 0.0
        self.loss = 0.0
        for n in range(self.output.size(0)):
            self.loss_vgg_list = self.criterion1(self.output[n].unsqueeze(0), self.target_vgg[n].unsqueeze(0))
            self.loss_vgg_equlized_list = self.criterion1(self.output[n].unsqueeze(0), self.target_vgg_equlized[n].unsqueeze(0))

            weight_vgg_equlized = 0.7
            kl = ((self.his_equlized[n] + 1e-8) * ((self.his_equlized[n] + 1e-8).log() - (self.his_linear[n] + 1e-8).log())).sum()
            weight_vgg_equlized = weight_vgg_equlized * (1 / (1 + torch.exp((kl - 10.7) * 1.5)))

            if weight_vgg_equlized < 0:
                weight_vgg_equlized = 0
            weight_vgg = - weight_vgg_equlized + 1

            self.loss_vgg = self.loss_vgg + weight_vgg * self.loss_vgg_list
            self.loss_vgg_equlized = self.loss_vgg_equlized + weight_vgg_equlized * self.loss_vgg_equlized_list
            self.loss = self.loss_vgg + self.loss_vgg_equlized

    def train(self):
        self.cal_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def save(self, mat_name, epoch_label, size, scale):
        return self.save_network(self.model, mat_name, epoch_label, size, scale)

    def load(self, save_path):
        self.load_network(save_path, self.model)