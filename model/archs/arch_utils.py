import torch.nn as nn


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    x---ReLU-Conv-ReLU-Conv-+-y
      |_____________________|
     """
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        f1 = self.conv1(self.relu(x))
        f2 = self.conv2(self.relu(f1))
        y = identity + f2
        return y

class ResidualBlock_Mod(nn.Module):
    """Residual block with modulation
    x1-ReLU-Conv-ReLU-Conv---|
    x2---ReLU-Conv-ReLU-Conv-*-+-y
       |_______________________|
    """
    def __init__(self, nf=64):
        super(ResidualBlock_Mod, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        identity = x2
        f1_1 = self.conv1(self.relu(x1))
        f1_2 = self.conv2(self.relu(f1_1))
        f2_1 = self.conv3(self.relu(x2))
        f2_2 = self.conv4(self.relu(f2_1))
        y = f1_2 * f2_2 + identity
        return y