import time
import numpy as np
import ising
import torch

from cvxgraphalgs.structures.cut import Cut

from src.util.plottings import laplacian_to_graph
import matplotlib.pyplot as plt

# code from https://github.com/dexter2206/ising

import maxCutPy.maxcutpy.graphcut as gc
import maxCutPy.maxcutpy.graphtest as gt
import maxCutPy.maxcutpy.maxcut as mc
import maxCutPy.maxcutpy.graphdraw as gd

def decode_state(state_repr, no_spins, labels):
    state = {}
    for i in range(no_spins):
        state[labels[no_spins-(i+1)]] = 1 if state_repr % 2 else -1
        state_repr //= 2
    return state


class Ising_model(torch.nn.Module):
    def __init__(self, cf, info_mtx):
        super(Ising_model, self).__init__()
        self.graph = {}
        self.info_mtx = info_mtx
        shape = info_mtx.shape
        self.cf = cf
        if self.cf.pb_type == "maxcut":
            G = -0.5*info_mtx
        elif self.cf.pb_type == "spinglass":
            G = info_mtx
        for i in range(shape[0]):
            for j in range(i, shape[1]):
                self.graph[(i,j)] = G[i,j]
        
        self.real_graph = laplacian_to_graph(self.info_mtx)
        self.nodes = list(self.real_graph.nodes)
        self.nodes.sort()


    def forward(self, state):
        total = 0
        for edge, energy in self.graph.items():
            if edge[0] == edge[1]:
                total += energy*state[edge[0]]
            else:
                total += energy*state[edge[0]]*state[edge[1]]
        total = -total
        if self.cf.pb_type == "maxcut":
            num_edges = self.info_mtx.sum()/2
            total = num_edges/2 - total
        return total


    def fast_forward(self, state):
        if self.cf.pb_type == "maxcut":
            sides = (state+1)/2
            left = {vertex for side, vertex in zip(sides, self.nodes) if side == 0}
            right = {vertex for side, vertex in zip(sides, self.nodes) if side == 1}
            cut = Cut(left, right)
            cut_size = cut.evaluate_cut_size(self.real_graph)
            return cut_size
        else:
            raise("not available for other problem than maxcut")



def ising_ground_truth(cf, info_mtx, fig_save_path=""):
    print("Running the Ground Truth...")
    ising_model = Ising_model(cf, info_mtx)
    dim = info_mtx.shape[0]
    if cf.pb_type == "maxcut":
        # 1/2\sum_edges 1-node1*node2 = 1/2*num_edges - 1/2*\sum_edges node1*node2
        graph = ising_model.graph
        start_time = time.time()
        result = ising.search(graph, num_states=3, show_progress=True, chunk_size=0)
        end_time = time.time()
        energy = result.energies[0]
        state = decode_state(result.states[0], dim, list(range(dim)))
        num_edges = info_mtx.sum()/2
        score = num_edges/2 - energy
        score1 = ising_model(state)
        if abs((score-score1)/score) > 1e-2:
            print("Mismatched energy - result1={}, result2={}".format(score, score1))
            raise

        # plot the graph
        # Laplacian matrices are real and symmetric, so we can use eigh, 
        # the variation on eig specialized for Hermetian matrices.
        G = laplacian_to_graph(info_mtx)

        # Laplacian matrices are real and symmetric, so we can use eigh, 
        # the variation on eig specialized for Hermetian matrices.
        # method from maxCutPy
        gt.execution_time(mc.local_consistent_max_cut, 1, G)
        score1 = gc.cut_edges(G)
        if abs((score-score1)) != 0:
            print("Mismatched maxcut - result1={}, result2={}".format(score, score1))
            raise

        nbunch = G.nodes()
        for i in nbunch:
            G.node[i]['partition'] = state[i]
        gd.draw_cut_graph(G)
        plt.savefig(fig_save_path)
        plt.close()
    if cf.pb_type == "spinglass":
        # \sum_edges J_ij*node1*node2
        graph = ising_model.graph
        start_time = time.time()
        result = ising.search(graph, num_states=3, show_progress=True, chunk_size=0)
        end_time = time.time()
        energy = result.energies[0]
        state = decode_state(result.states[0], dim, list(range(dim)))
        energy1 = ising_model(state)
        if abs((energy-energy1)/energy) > 1e-2:
            print("Mismatched energy - result1={}, result2={}".format(energy, energy1))
            raise
        score = energy
    time_elapsed = end_time - start_time
    return score, state, time_elapsed
