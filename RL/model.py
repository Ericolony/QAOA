import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(torch.nn.Module):
    # def __init__(self, input_dim):
    #     super(Policy, self).__init__()
    #     self.input_dim = input_dim
    #     self.fid_emb = nn.Embedding(self.input_dim, self.input_dim)
    #     self.fc1 = torch.nn.Linear(2 * self.input_dim, 128)
    #     self.fc2 = torch.nn.Linear(128, 1)


    # def forward(self, state, action):
    #     inp = torch.cat((state, self.fid_emb(action)), -1)
    #     inp = F.relu(self.fc1(inp))
    #     inp = self.fc2(inp)
    #     return inp.squeeze(-1)
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(self.input_dim, self.input_dim*2)
        self.fc2 = torch.nn.Linear(self.input_dim*2, self.output_dim)


    def forward(self, state):
        inp = F.relu(self.fc1(state))
        inp = self.fc2(inp)
        return inp
