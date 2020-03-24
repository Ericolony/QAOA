
import torch 
import numpy as np
import copy
import random
import torch.nn.functional as F
from collections import deque
from copy import deepcopy

from RL.model import Policy



class DQN(torch.nn.Module):
    def __init__(self, cf, environment, gamma, buffer_size):
        super(DQN, self).__init__()
        self.cf = cf
        self.environment = environment
        self.size = environment.size
        self.model = Policy(cf, self.size*2, self.size).to(self.cf.device)
        self.target_model = copy.deepcopy(self.model).to(self.cf.device)
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=cf.learning_rate)
        self.batch_size = cf.batch_size
        self.epsilon = 0.1
        self.buffer_size = buffer_size
        self.step_counter = 0
        self.epsi_high = 0.9
        self.epsi_low = 0.05
        self.steps = 0
        self.count = 0
        self.decay = 200
        self.eps = self.epsi_high
        self.update_target_step = 1000
        self.replay_buffer = deque(maxlen=buffer_size)
        

    def run_episode(self):
        state = self.environment.reset()
        goal = self.environment.sample_best_buffer(1)
        sum_r = 0
        min_dist = self.size
        mean_loss = []

        for t in range(self.size):
            self.steps += 1
            self.eps = self.epsi_low + (self.epsi_high-self.epsi_low) * (np.exp(-1.0 * self.steps/self.decay))
            Q = self.model(state.to(self.cf.device), goal.to(self.cf.device))
            num = np.random.rand()
            if (num < self.eps):
                action = torch.randint(0, self.size, (1,)).type(torch.LongTensor)
            else:
                action = Q.argmax(dim=1)

            new_state, reward, new_val = self.environment.step(state, action)
            done = self.environment.update_best_buffer(new_state, new_val)
            sum_r = sum_r + reward[0].item()

            self.replay_buffer.append([deepcopy(state[0].numpy()),deepcopy(action[0]),deepcopy(reward[0]),deepcopy(new_state[0].numpy()),deepcopy(done)])
            loss = self.update_model()
            mean_loss.append(loss)
            state = deepcopy(new_state)
            
            self.step_counter = self.step_counter + 1
            if (self.step_counter > self.update_target_step):
                self.target_model.load_state_dict(self.model.state_dict())
                self.step_counter = 0
                print('updated target model')
            if (t + 1) == self.size:
                break
        
        print("init {}, final {}".format(self.environment.init_val, self.environment.next_val))
        print("Best buffer{}".format(self.environment.best_vals))
        print("Replay_buffer size{}".format(len(self.replay_buffer)))
        return sum_r, np.mean(np.array(mean_loss)), self.environment.next_val


    def update_model(self):
        self.optimizer.zero_grad()
        num = len(self.replay_buffer)
        K = np.min([num, self.batch_size])
        samples = random.sample(self.replay_buffer, K)
        
        S0, A0, R1, S1, D1 = zip(*samples)

        S0 = torch.tensor( S0, dtype=torch.float)
        A0 = torch.tensor( A0, dtype=torch.long).view(K, -1)
        R1 = torch.tensor( R1, dtype=torch.float).view(K, -1)
        S1 = torch.tensor( S1, dtype=torch.float)
        D1 = torch.tensor( D1, dtype=torch.float)

        goal = self.environment.sample_best_buffer(K)

        target_q = R1.squeeze().to(self.cf.device) + self.gamma*self.target_model(S1.to(self.cf.device), goal.to(self.cf.device)).max(dim=1)[0].detach()*(1 - D1.to(self.cf.device))
        policy_q = self.model(S0.to(self.cf.device), goal.to(self.cf.device)).gather(1,A0.to(self.cf.device))

        L = F.smooth_l1_loss(policy_q.squeeze(), target_q.squeeze())
        L.backward()
        self.optimizer.step()
        return L.detach().item()


    def forward(self):
        total_reward, average_loss, final_val = self.run_episode()
        return total_reward, average_loss, final_val
