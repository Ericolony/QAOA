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
    gamma = 0.99
    buffer_size = int(1e6)
    alg = DQN(cf, env, gamma, buffer_size)
    epochs = 2500
    results = []

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

    quant = np.max(results)
    time_ellapsed = end_time - start_time
    exp_name, sep, tail = (cf.dir).partition('-date')
    return exp_name, quant, time_ellapsed

    # Y = np.asarray(log.get_log('tot_return'))
    # Y2 = smooth(Y)
    # x = np.linspace(0, len(Y), len(Y))
    # fig1 = plt.figure()
    # ax1 = plt.axes()
    # ax1.plot(x, Y, Y2)
    # plt.xlabel('episodes')
    # plt.ylabel('episode return')

    # Y = np.asarray(log.get_log('avg_loss'))
    # Y2 = smooth(Y)
    # x = np.linspace(0, len(Y), len(Y))
    # fig2 = plt.figure()
    # ax2 = plt.axes()
    # ax2.plot(x, Y, Y2)
    # plt.xlabel('episodes')
    # plt.ylabel('average loss')

    # Y = np.asarray(log.get_log('final_dist'))
    # Y2 = smooth(Y)
    # x = np.linspace(0, len(Y), len(Y))
    # fig3 = plt.figure()
    # ax3 = plt.axes()
    # ax3.plot(x, Y, Y2)
    # plt.xlabel('episodes')
    # plt.ylabel('minimum distance')

    # Y = np.asarray(log.get_log('final_dist'))
    # Y[Y > 1] = 1.0
    # K = 100
    # Z = Y.reshape(int(epochs/K),K)
    # T = 1 - np.mean(Z,axis=1)
    # x = np.linspace(0, len(T), len(T))*K
    # fig4 = plt.figure()
    # ax4 = plt.axes()
    # ax4.plot(x, T)
    # plt.xlabel('episodes')
    # plt.ylabel('sucess rate')

    # plt.savefig("result.png")