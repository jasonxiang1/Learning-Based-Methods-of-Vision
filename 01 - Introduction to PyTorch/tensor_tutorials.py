# installation directions can be found on pytorch's webpage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
#%matplotlib inline


# PyTorch tensor tutorial
# create empty tensor that is 2x3
x = torch.empty(2, 3)
print(x)


# create random value tensor
y = torch.rand(2,2)
print(y)

# create zero value tensor
z = torch.ones(3,1)
print(z)

#create ones value tensor
onesTens = torch.zeros(3,3)
print(onesTens)

print(sum(z==z).numpy()/4)