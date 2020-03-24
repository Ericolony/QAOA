# -*- coding: utf-8 -*-
# https://github.com/orrivlin/Hindsight-Experience-Replay---Bit-Flipping

import os
import numpy as np
import time
import matplotlib.pyplot as plt

from RL.env import Environment
from RL.dqn import DQN

def smooth(x, window_len=11, window='hanning'):
    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def train(cf, info_mtx):
    size = cf.input_size[0]
    env = Environment(cf, size, info_mtx)
    gamma = 0.5
    buffer_size = int(1e3)
    alg = DQN(cf, env, gamma, buffer_size)
    epochs = cf.epochs

    results = []
    losses = []

    start_time = time.time()
    for i in range(epochs):
        total_reward, average_loss, final_result = alg()
        print('Done: {} of {}. reward: {}. loss: {}'.format(i, epochs, total_reward, average_loss))
        if i == 2000:
            for param_group in alg.optimizer.param_groups:
                param_group['lr'] = 0.0001
        if i == 4500:
            for param_group in alg.optimizer.param_groups:
                param_group['lr'] = 0.00005
        results.append(final_result)
        losses.append(average_loss)

    end_time = time.time()

    np.save(os.path.join(cf.dir, "result.npy"), results)

    Y = np.array(results)
    Y2 = smooth(Y)
    x = np.linspace(0, len(Y), len(Y))
    fig1 = plt.figure()
    ax1 = plt.axes()
    ax1.plot(x, Y, Y2)
    plt.xlabel('episodes')
    plt.ylabel('result')
    plt.title('MaxCut-{}'.format(cf.input_size[0]))
    plt.savefig(os.path.join(cf.dir, "result.png"))

    Y = np.array(losses)
    Y2 = smooth(Y)
    x = np.linspace(0, len(Y), len(Y))
    fig1 = plt.figure()
    ax1 = plt.axes()
    ax1.plot(x, Y, Y2)
    plt.xlabel('episodes')
    plt.ylabel('loss')
    plt.title('MaxCut-{}'.format(cf.input_size[0]))
    plt.savefig(os.path.join(cf.dir, "losses.png"))

    quant = np.max(results)
    time_ellapsed = end_time - start_time
    exp_name, sep, tail = (cf.dir).partition('-date')
    return exp_name, quant, time_ellapsed