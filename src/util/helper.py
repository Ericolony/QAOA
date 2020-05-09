import numpy as np
import os


def record_result(cf, exp_name, score, time_elapsed, bound=0, state=None):
    f=open("results.txt", "a+")
    f.write("[{}] - Score: {:.2f}, Time: {:.2f} seconds\n".format(exp_name, score, time_elapsed))
    if bound is not None:
        f.write("Bound: {}\n".format(bound))
    if state is not None:
        f.write("Optimal State: {}\n".format(state))
    f.write("----------------------------------------------------------------------------------------\n")
    f.close()
    print(exp_name + ": Score={}, Time_elapsed={:.2f}".format(score, time_elapsed))

    numpy_file_name = "result.npy"
    if not os.path.exists(numpy_file_name):
        dic = {}
        np.save(numpy_file_name, dic)
    dic = np.load(numpy_file_name, allow_pickle=True).item()
    if exp_name in dic:
        dic[exp_name].append([score, time_elapsed, bound])
    else:
        dic[exp_name] = [[score, time_elapsed, bound]]
    np.save(numpy_file_name, dic)


def compute_edge_weight_cut(operator, sample):
    _,energy,_ = operator.find_conn(sample)
    # _,_,energy = sampler.energy_observable.estimate(sampler.wave_function, sample)
    energy = -np.real(energy)
    var = np.var(energy)
    mean = np.mean(energy)
    best = np.max(energy)
    return best, mean, var, energy


def evaluate(sampler, operator):
    sample = next(sampler)
    best, mean, var, energy_sample = compute_edge_weight_cut(operator, sample)
    print("Total {} sampled configurations, best: {}, mean：{}， var: {}".format(sample.shape[0], best, mean, var))
    sample = operator.random_states(sample.shape[0])
    best, mean, var, energy_random = compute_edge_weight_cut(operator, sample)
    print("Total {} random configurations, best: {}, mean：{}， var: {}".format(sample.shape[0], best, mean, var))
    return energy_sample, energy_random


def make_locally_connect(cf, J_mtx):
    n_states = J_mtx.shape[0]
    length = np.sqrt(n_states)
    assert length == int(length)
    length = int(length)
    adj_mtx = np.zeros([n_states,n_states])
    for row in range(length):
        for col in range(length):
            right = [row, (col+1)%length]
            left = [row, (col-1)%length]
            up = [(row-1)%length, col]
            down = [(row+1)%length, col]

            orig = row*length + col
            up_ind = up[0]*length + up[1]
            down_ind = down[0]*length + down[1]
            left_ind = left[0]*length + left[1]
            right_ind = right[0]*length + right[1]

            adj_mtx[orig, up_ind], adj_mtx[orig, down_ind], adj_mtx[orig, left_ind], adj_mtx[orig, right_ind] = 1.0,1.0,1.0,1.0
    return J_mtx*adj_mtx