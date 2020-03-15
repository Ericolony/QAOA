import numpy as np
import os
from .plottings import plot_graph

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