import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from . import *
from torch.autograd import Variable

class SimpleMLP(nn.Module):
    def __init__(self,config):
        super(SimpleMLP,self).__init__()
        self.config = config
        if self.config.n_steps == 1:
            self.fc1 = nn.Linear(4, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 2)
        else:
            self.fc1 = nn.Linear(4, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        if self.config.n_steps == 1:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()