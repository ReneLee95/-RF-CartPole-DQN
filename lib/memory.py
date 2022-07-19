from . import *
from typing import Tuple
from collections import deque
import numpy as np
from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self,capacity,config):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.config = config
        
        self.n_step_buffer = deque(maxlen = self.config.n_steps)
        self.n_step_capacity = 0
        
    
    def n_step_write(self, transition):
        if self.size() > self.capacity:
            print("enter1")
            self.buffer.popleft()
        self.buffer.append(transition)
    
    def write(self, transition):
        if self.config.n_steps == 1:
            if self.size() > self.capacity:
                self.popleft()
            self.buffer.append(transition)
        else:
            if self.size() >= self.config.n_steps:
                self.n_step_buffer.popleft()
                self.n_step_buffer.append(transition)
            else:    
                self.n_step_buffer.append(transition)
            
    def n_step_sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, done = transition
            states.append(state)
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)
            dones.append([done])
        return torch.tensor(states, dtype=torch.float), torch.tensor(actions),torch.tensor(rewards), torch.tensor(next_states, dtype=torch.float), torch.tensor(dones)
    
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, done = transition
            states.append(state)
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)
            dones.append([done])

        return torch.tensor(states, dtype=torch.float), torch.tensor(actions),torch.tensor(rewards), torch.tensor(next_states, dtype=torch.float), torch.tensor(dones)
    
    def size(self):
        if self.config.n_steps == 1:
            return len(self.buffer)
        else:
            return len(self.n_step_buffer)
