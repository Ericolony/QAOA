# -*- coding: utf-8 -*-

import torch
import numpy as np
from copy import deepcopy
from src.ising_gt import Ising_model

class Environment(torch.nn.Module):
    def __init__(self, cf, size, info_mtx):
        super(Environment, self).__init__()
        self.size = size
        self.ising = Ising_model(cf, info_mtx)
        
        
    def reset(self):
        state = torch.rand(1, self.size).round()
        done = False
        return state, done


    def forward(self, state):
        results = []
        for i in range(state.shape[0]):
            state_spin = (state[i]-0.5)*2
            results.append(self.ising(state_spin))
        results = torch.tensor(np.stack(results, axis=0))
        return results


    def step(self, state, action):
        new_state = deepcopy(state)
        for ind,act in enumerate(action):
            new_state[ind, act] = 1.0 - new_state[ind, act]
        done = False
        curr_val = self.forward(state)
        new_val = self.forward(new_state)
        reward = new_val - curr_val
        return new_state, reward, done