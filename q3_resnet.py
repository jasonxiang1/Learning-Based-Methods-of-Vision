# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
# %matplotlib inline

import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset

torch.cuda.empty_cache()
'''
model is already initialized with random weights
'''
# %%

args = ARGS(batch_size = 8, epochs=50, lr = 0.0001)
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 20)
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
if __name__ == '__main__':
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
# %%
