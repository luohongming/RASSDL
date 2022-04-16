import torch
import torch.nn as nn
import os
from torch.nn import init
from collections import OrderedDict

class Base_model():
    def initialize(self, opt):
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.schedulers = []
        self.optimizers = []
        self.save_model_dir = opt.save_model_dir if opt.save_model_dir else './'
        if not os.path.exists(self.save_model_dir):
            os.mkdir(self.save_model_dir)

    def _set_lr(self, lr_group_l):
        """Set learning rate for warmup
        lr_group_l: list for lr_groups, each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_group_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler"""
        init_lr_grouops_l = []
        for optimizer in self.optimizers:
            init_lr_grouops_l.append(v['initial_lr'] for v in optimizer.param_groups)
        return init_lr_grouops_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        # set up warm-up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
                # set learning rate
                self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel):
            network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))

    def set_mode(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def save_network(self, network, network_label, epoch_label, size, scale):
        save_filename = '{}/{}_{}_{}.pth'.format(network_label, epoch_label, size, scale)
        if not os.path.exists(os.path.join(self.save_model_dir, '{}'.format(network_label))):
            os.mkdir(os.path.join(self.save_model_dir, '{}'.format(network_label)))
        save_path = os.path.join(self.save_model_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
        return save_path

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)


def init_weights(net, init_type):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)