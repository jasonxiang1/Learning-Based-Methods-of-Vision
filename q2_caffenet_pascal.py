# %%
# ARCHITECTURE:
# -> image
# -> conv(11, 4, 96, ’VALID’)
# -> relu()
# -> max_pool(3, 2)
# -> conv(5, 1, 256, 'SAME')
# -> relu()
# -> max_pool(3, 2)
# -> conv(3, 1, 384, 'SAME')
# -> relu()
# -> conv(3, 1, 384, 'SAME')
# -> relu()
# -> conv(3, 1, 256, ’SAME’)
# -> relu()
# -> max_pool(3, 2)
# -> flatten()
# -> fully_connected(4096)
# -> relu()
# -> dropout(0.5)
# -> fully_connected(4096)
# -> relu()
# -> dropout(0.5)
# -> fully_connected(20)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline

import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset


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

class CaffeNet(nn.Module):
    def __init__(self):
        super().__init__()
        c_dim = 3
        self.conv1 = nn.Conv2d(c_dim,96,11,4,padding=0) # valid padding
        self.pool1 = nn.MaxPool2d(3,2)
        self.conv2 = nn.Conv2d(96, 256, 5,padding=2) # same padding
        self.pool2 = nn.MaxPool2d(3,2)
        self.conv3 = nn.Conv2d(256,384,3,padding=1) # same padding
        self.conv4 = nn.Conv2d(384,384,3,padding=1) # same padding
        self.conv5 = nn.Conv2d(384,256,3,padding=1) # same padding
        self.pool3 = nn.MaxPool2d(3,2)
        self.flat_dim = 5*5*256 # replace with the actual value
        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, 4096, 'relu'))
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Sequential(*get_fc(4096, 4096, 'relu'))
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Sequential(*get_fc(4096, 20, 'none'))

        self.nonlinear = lambda x: torch.clamp(x,0)
        # raise NotImplementedError
    
    def forward(self, x):
        N = x.size(0)
        x = self.conv1(x)
        x = self.nonlinear(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.nonlinear(x)
        x = self.pool3(x)
        x = x.view(N, self.flat_dim) # flatten the array

        out = self.fc1(x)
        out = self.nonlinear(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.nonlinear(out)
        out = self.dropout2(out)
        out = self.fc3(out)

        return out

        # raise NotImplementedError

# %%
def xavier_normal_init(m):
    if(type(m)==nn.Conv2d or type(m)==nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)

# %%
args = ARGS(batch_size = 32, epochs=50, lr = 0.0001)
args.gamma = 0.85
weightDecay = 5e-5
model = CaffeNet()

model.apply(xavier_normal_init)


optimizer = torch.optim.Adam(model.parameters(), lr = args.lr,weight_decay=weightDecay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
if __name__ == '__main__':
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)


# %%
