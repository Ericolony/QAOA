# -*- coding: utf-8 -*-

import torch
import numpy as np
from copy import deepcopy
from src.ising_gt import Ising_model

class Environment(torch.nn.Module):
    def __init__(self, cf, size, info_mtx):
        super(Environment, self).__init__()
        self.size = size
        self.info_mtx = info_mtx
        self.ising = Ising_model(cf, info_mtx)

        self.init_val = 0
        self.curr_val = 0
        self.next_val = 0

        self.best_num = 10
        self.best_states = torch.rand(1, self.size).round()
        self.best_vals = self.forward(self.best_states)
        
        
    def reset(self):
        state = torch.rand(1, self.size).round()
        # state = self.sample_best_buffer(1)
        self.init_val = self.forward(state)
        self.curr_val = self.init_val
        self.next_val = self.init_val
        return state


    def forward(self, state):
        results = []
        for i in range(state.shape[0]):
            state_spin = (state[i]-0.5)*2
            # results.append(self.ising(state_spin))
            results.append(self.ising.fast_forward(state_spin))
        results = torch.tensor(np.stack(results, axis=0))
        return results
    

    def update_best_buffer(self, state, val):
        min_best = torch.min(self.best_vals).item()
        if (val > min_best) and ((self.best_states-state).abs().sum(-1).min().item()!=0):
            self.best_states = torch.cat((self.best_states, state), dim=0)
            self.best_vals = torch.cat((self.best_vals, val), dim=0)
        if len(self.best_vals) > self.best_num:
            index = torch.argmin(self.best_vals).item()
            ind = torch.arange(len(self.best_vals))
            ind = ind[ind!=index]
            self.best_states = self.best_states[ind]
            self.best_vals = self.best_vals[ind]
            return True
        return False


    def sample_best_buffer(self, K):
        sample = np.random.choice(np.arange(len(self.best_vals)), K)
        return self.best_states[sample]


    def step(self, state, action):
        new_state = deepcopy(state)
        for ind,act in enumerate(action):
            new_state[ind, act] = 1.0 - new_state[ind, act]
        self.curr_val = self.next_val
        self.next_val = self.forward(new_state)
        reward = self.next_val - self.curr_val

        # goal = self.sample_best_buffer(1)
        # reward = reward-((new_state - goal).abs().sum()).unsqueeze(0)
        return new_state, reward, self.next_val


        # self.update_best_buffer(self.next_val, new_state)
        # best_state = self.sample_best_buffer(1)