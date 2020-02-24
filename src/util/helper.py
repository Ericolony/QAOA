import numpy as np
import os
from .plottings import plot_graph

def load_data(cf):
    size = np.prod(cf.input_size)
    if cf.pb_type == "maxcut":
        graph_data_path = "./data/maxcut/graph{}.npy".format(cf.input_size)
        if not os.path.exists(graph_data_path):
            laplacian = np.random.randint(2, size=[size,size])
            laplacian = (laplacian + laplacian.transpose())//2
            np.fill_diagonal(laplacian, 0)
            np.save(graph_data_path, laplacian)
            plot_graph(cf, laplacian, graph_data_path[:-4]+".png")

        else:
            laplacian = np.load(graph_data_path)
        return laplacian
    elif cf.pb_type == "spinglass":
        graph_data_path = "./data/spinglass/graph{}.npy".format(cf.input_size)
        J_data_path = "./data/spinglass/J{}.npy".format(cf.input_size)
        if not os.path.exists(graph_data_path):
            laplacian = np.random.randint(2, size=[size,size])
            laplacian = (laplacian + laplacian.transpose())//2
            np.fill_diagonal(laplacian, 0)
            np.save(graph_data_path, laplacian)
            plot_graph(cf, laplacian, graph_data_path[:-4]+".png")

            J = np.random.normal(0,0.5,size**2)
            J = np.reshape(J, [size,size])
            np.save(J_data_path, J)

        else:
            laplacian = np.load(graph_data_path)
            J = np.load(J_data_path)
        return laplacian, J


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