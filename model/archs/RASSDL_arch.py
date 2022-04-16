import torch
import torch.nn as nn


class RASSDL_arch(nn.Module):
    def __init__(self, n_dense=3, nf=64):
        super(RASSDL_arch, self).__init__()
        self.n_dense = n_dense
        self.nf = nf
        self.conv1 = nn.Conv2d(1, self.nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.dense = nn.ModuleList()
        for n in range(1, self.n_dense + 2):
            self.dense.append(nn.Sequential(
                nn.Conv2d(self.nf * n, self.nf, 3, 1, 1, bias=True),
                nn.BatchNorm2d(self.nf),
                nn.ReLU(inplace=True)
            ))
        self.fusion = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.conv3 = nn.Conv2d(self.nf, 1, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f = self.relu(self.conv2(self.relu(self.conv1(x))))
        f_ = f
        for n in range(self.n_dense):
            f = torch.cat((f, self.dense[n](f)), 1)
        f = self.dense[self.n_dense](f)
        f1 = f_ + self.fusion(f)
        y = self.sigmoid(self.conv3(f1))
        return y