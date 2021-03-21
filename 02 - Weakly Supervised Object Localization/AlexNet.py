import numpy as np

import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision.models as models

import torch

model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

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

class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        #TODO: Define model
        # self.num_classes = num_classes
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=11,stride=4, padding=4)
        # self.pool1 = nn.MaxPool2d(3,2)
        # self.conv2 = nn.Conv2d(64, 192, kernel_size=5,stride=1, padding=2)
        # self.pool2 = nn.MaxPool2d(3,2)
        # self.conv3 = nn.Conv2d(192,384, kernel_size=3,stride=1,padding=1)
        # self.conv4 = nn.Conv2d(384,256, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
            )
        
        self.nonlinear = lambda x: torch.clamp(x,0)

    def forward(self, x):
        #TODO: Define forward pass

        x = self.features(x)

        out = self.classifier(x)


        return out


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        #TODO: Define model


    def forward(self, x):
        #TODO: Define fwd pass


        return x


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    #TODO: Initialize weights correctly based on whethet it is pretrained or not
    pretrained_dict = torch.load('saved_models/alexnet-owt-4df8aa71.pth')
    new = list(pretrained_dict.items())
    model_dict = model.state_dict()

    #modify the model's state_dict
    count = 0
    for key, value in model_dict.items():
        layer_name,weights=new[count]
        if count >= 10:
            if 'bias' in layer_name:
                torch.nn.init.zeros_(model_dict[key].data)
            else:
                torch.nn.init.xavier_normal(model_dict[key].data)
            count += 1
        else:
            model_dict[key] = weights
            count += 1
    
    # xavier initialize other layers


    return model


if __name__ == "__main__":
    modelCaffe = CaffeNet()
    torchLoadCaffe = torch.load('saved_models/CaffeNet-50.pth')
    modelCaffe.load_state_dict(torchLoadCaffe['model_state_dict'])
    modelCaffe = modelCaffe.to(args.device)
    modelCaffe.eval()
    testfindex, testOutput, testTarget = trainer.train_output_CaffeNet(modelCaffe, args)

def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    #TODO: Ignore for now until instructed
    

    return model