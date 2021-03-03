# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# Modified by Sudeep Dasari
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_fc(inp_dim, out_dim, non_linear='relu'):
    """
    Mid-level API. It is useful to customize your own for large code repo.
    :param inp_dim: int, intput dimension
    :param out_dim: int, output dimension
    :param non_linear: str, 'relu', 'softmax'
    :return: list of layers [FC(inp_dim, out_dim), (non linear layer)]
    """
    layers = []
    layers.append(nn.Linear(inp_dim, out_dim))
    if non_linear == 'relu':
        layers.append(nn.ReLU())
    elif non_linear == 'softmax':
        layers.append(nn.Softmax(dim=1))
    elif non_linear == 'none':
        pass
    else:
        raise NotImplementedError
    return layers


class SimpleCNN(nn.Module):
    """
    Model definition
    """
    def __init__(self, num_classes=10, inp_size=28, c_dim=1):
        super().__init__()
        self.num_classes = num_classes
        # add your layer one by one -- one way to add layers
        self.conv1 = nn.Conv2d(c_dim, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # TODO: Modify the code here
        self.nonlinear = lambda x: torch.clamp(x, 0)
        # self.nonlinear = lambda x: torch.clamp(x,0)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)

        # TODO: q0.1 Modify the code here
        self.flat_dim = 7*7*64
        # self.flat_dim = 56*56*64
        # chain your layers by Sequential -- another way
        # TODO: Modify the code here
        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, 128, 'relu'))
        self.fc2 = nn.Sequential(*get_fc(128, num_classes, 'none'))

    def forward(self, x):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: out: classification logits in shape of (N, Nc)
        """
        N = x.size(0)
        # print("Initial Size of x: ",x.size())
        x = self.conv1(x)
        x = self.nonlinear(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool2(x)
        # print("Final size of x: ", x.size())

        # TODO: q0.1 hint (you might want to check the dimension of input here)
        flat_x = x.view(N, self.flat_dim) #dimension of the input is 64 x 64 x 7 x 7
        out = self.fc1(flat_x)
        # print("First FC layer size: ",out.size())
        out = self.fc2(out)
        # print("Second FC layer size: ",out.size()) #want the output size to be [64, 10]
        return out
