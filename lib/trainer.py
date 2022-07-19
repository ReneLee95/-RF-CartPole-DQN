import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

from . import *
from .memory import ReplayMemory
from .model import SimpleMLP
import math
import random

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class DQNTrainer:
    def __init__(
        self,
        config
    ):
        self.config = config
        self.env = gym.make(config.env_id)
        self.epsilon = self.config.eps_start  # My little gift for you
        
        self.model = SimpleMLP(self.config).to(self.config.device)      
        self.target = SimpleMLP(self.config).to(self.config.device)    
        self.target.load_state_dict(self.model.state_dict())
        
        self.steps_done = 0
        
        self.memory = ReplayMemory(30000,self.config)
        self.optimizer = self.config.optim_cls(self.model.parameters(), **self.config.optim_kwargs)
        self.gamma = self.config.discount_rate
        
    def train(self, num_train_steps: int):
        
        episode_rewards = []
                   
        state = self.env.reset()
        done = False
        steps = 0
        i = 0
        
        
        n_step_count = 0
        
        while i < num_train_steps + 1:
            i += 1
            if self.config.n_steps == 1:
                
                action = self.model.sample_action(torch.from_numpy(state).float(), self.epsilon)

                next_state,reward,done,info = self.env.step(action)

                done_mask = 0.0 if done else 1.0
                self.memory.write((state,action,reward,next_state,done_mask))
                state = next_state

                steps +=1
                if self.memory.size() > 500:
                    self.update_network(self.model,self.target,self.memory,self.optimizer)

                if i % 2000 == 0:
                    self.epsilon = self.update_epsilon()
                    
                if i % 5000 == 0:
                    self.update_target()
                    
                if done == True and self.config.verbose == True:
                    status_string = f"{self.config.run_name:10}, Whatever you want to print out to the console"
                    print(status_string + "\r", end="", flush=True)

                    episode_rewards.append(steps)
                    reward = -1
                    state = self.env.reset()
                    done = False
                    steps = 0
            else:
                
                action = self.model.sample_action(torch.from_numpy(state).float(), self.epsilon)

                next_state,reward,done,info = self.env.step(action)
                done_mask = 0.0 if done else 1.0
                self.memory.write((state,action,reward,next_state,done_mask))
                self.memory.n_step_write((state,action,reward,next_state,done_mask))
                state = next_state

                steps +=1
                  
                if len(self.memory.n_step_buffer) == self.config.n_steps and len(self.memory.buffer) > 8000:                   
                    self.update_network(self.model,self.target,self.memory,self.optimizer)
                
                if i % 2000 == 0:
                    self.epsilon = self.update_epsilon()
                    
                if i == 8000:
                    self.update_target()
                
                if i % 11500 == 0:
                    self.update_target()

                if done == True and self.config.verbose == True:
                    status_string = f"{self.config.run_name:10}, Whatever you want to print out to the console"
                    print(status_string + "\r", end="", flush=True)
                    
                    episode_rewards.append(steps)
                    reward = -1
                    state = self.env.reset()
                    done = False
                    steps = 0
                       
        return episode_rewards
    
    # Update online network with samples in the replay memory. 
    def update_network(self,model,target,memory,optimizer):
        if self.config.n_steps == 1:
            state,action,reward,next_state,done = memory.sample(self.config.batch_size)

            model_out = self.predict(model,state)
            model_action = model_out.gather(1,action)
            max_model_prime = target(next_state).max(1)[0].unsqueeze(1)

            target_value = reward + self.gamma * max_model_prime * done

            loss = self.config.loss_fn(model_action,target_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            new_gamma = self.gamma
            state,action,reward,next_state,done = memory.n_step_sample(self.config.batch_size)

            model_out = self.predict(model,state)
            model_action = model_out.gather(1,action)
            max_model_prime = target(next_state).max(1)[0].unsqueeze(1)

            new_reward = self.memory.n_step_buffer[0][2] * self.gamma #* self.memory.n_step_buffer[0][4]
            for k in reversed(range(1,self.config.n_steps)):
                new_gamma = new_gamma * self.gamma
                new_reward += self.memory.n_step_buffer[k][2] * new_gamma * self.memory.n_step_buffer[k][4]
            
            target_value = new_reward + self.gamma * max_model_prime * done
            
            loss = self.config.loss_fn(model_action,target_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    
    # Update the target network's weights with the online network's one. 
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
    
    # Return desired action(s) that maximizes the Q-value for given observation(s) by the online network.
    def predict(self, model,ob):
        return model(ob)
    
    # Update epsilon over training process.
    def update_epsilon(self):
        epsilon_threshold = self.config.eps_end + (self.config.eps_start - self.config.eps_end) * math.exp(-1. * self.steps_done / self.config.eps_decay)
        self.steps_done += 1
        
        return epsilon_threshold
