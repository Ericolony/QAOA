import numpy as np
import os

def load_data(cf):
    graph_data_path = "./data/graph{}.npy".format(cf.input_size)
    if not os.path.exists(graph_data_path):
        laplacian = np.random.randint(2, size=[cf.input_size,cf.input_size])
        laplacian = (laplacian + laplacian.transpose())//2
        np.fill_diagonal(laplacian, 0)
        np.save(graph_data_path, laplacian)
        plot_graph(laplacian, graph_data_path[:-4]+".png")
    else:
        laplacian = np.load(graph_data_path)
    return laplacian


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