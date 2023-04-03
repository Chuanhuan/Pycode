#%%
def hook(grad):
   return grad + 2

import torch

#%%
#initializing two tensors(requires_grad = True is necessary to calculate gradients)
a = torch.tensor(7.0, requires_grad=True)
b = torch.tensor(13.0, requires_grad=True)
c = a * b
c.retain_grad()#to store the gradient of C
c.backward()

print(a.grad)
print(b.grad)
print(c.grad)

#%%
c = a * b
#registering the tensor c with the hook
c.register_hook(lambda grad: hook(grad))
c.retain_grad()
c.backward()
print(a.grad)
print(b.grad)
print(c.grad)

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
   def __init__(self):
      super().__init__() 
      self.conv = nn.Conv2d(3,8,2)
      self.pool = nn.AdaptiveAvgPool2d((4,4))
      self.fc = nn.Linear(8*4*4 , 1)
def forward(self, x):
      x = F.relu(self.conv(x))
      x = self.pool(x)
      x = x.view(x.shape[0] , -1)
      x = self.fc(x)
      return x
net = Net()
feats = {} #an empty dictionary
def hook_func(m , inp ,op):
   feats['feat'] = op.detach()

#%% 
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import shap
import os

# Load the iris dataset
