import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy as np

def laplacian_to_graph(L):
    N = L.shape[0]

    edges = []
    for i in range(N):
        for j in range(i, N):
            if L[i,j] != 0.: edges.append([i, j])
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def plot_graph(cf, L, save_path):
    if cf.pb_type == "maxcut":
        import maxCutPy.maxcutpy.graphdraw as gd
        import maxCutPy.maxcutpy.maxcut as mc
        import maxCutPy.maxcutpy.graphcut as gc
        import maxCutPy.maxcutpy.graphtest as gt
        seed = cf.random_seed
        # Laplacian matrices are real and symmetric, so we can use eigh, 
        # the variation on eig specialized for Hermetian matrices.
        # method from maxCutPy
        time = gt.execution_time(mc.local_consistent_max_cut, 1, G)
        result = gc.cut_edges(G)
        return time, result
    if cf.pb_type == "spinglass":
        import ising
        def decode_state(state_repr, no_spins, labels):
            state = {}
            for i in range(no_spins):
                state[labels[no_spins-(i+1)]] = 1 if state_repr % 2 else -1
                state_repr //= 2
            return state

        def check(dic, graph):
            total = 0
            for edge, energy in graph.items():
                total += energy*dic[edge[0]]*dic[edge[1]]
            return total

        J = L
        graph = {}
        shape = J.shape
        size = int(shape[0]*shape[1])
        for i in range(shape[0]):
            for j in range(shape[1]):
                graph[(i,j)] = J[i,j]
        result = ising.search(graph, num_states=4)
        print(result.energies, result.states)
        print(decode_state(result.states[0], size, list(range(size))))
        print(check(decode_state(result.states[0], size, list(range(size))), graph))



def plot_train_curve(cf, result_sample, result_random, save_path_name):
    if cf.pb_type == "maxcut":
        index = np.arange(1, result_sample.shape[0]+1)*cf.log_interval
        mean_sample = np.mean(result_sample, 1)
        std_sample = np.std(result_sample, 1)
        mean_random = np.mean(result_random, 1)
        std_random = np.std(result_random, 1)

        plt.plot(index, mean_sample, color='g', linewidth=2, label="sample") #mean curve.
        plt.fill_between(index, mean_sample-std_sample, mean_sample+std_sample, color='b', alpha=.1)

        plt.plot(index, mean_random, color='y', linewidth=2, label="random") #mean curve.
        plt.fill_between(index, mean_random-std_random, mean_random+std_random, color='b', alpha=.1)

        plt.legend(loc="upper left")
        plt.suptitle('Cut Weight over Training Iteration', fontsize=16)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Cut Weight', fontsize=12)
        plt.savefig(save_path_name)
    elif cf.pb_type == "spinglass":
        index = np.arange(1, result_sample.shape[0]+1)*cf.log_interval
        result_sample = -result_sample
        mean_sample = np.mean(result_sample, 1)
        std_sample = np.std(result_sample, 1)
        mean_random = np.mean(result_random, 1)
        std_random = np.std(result_random, 1)

        plt.plot(index, mean_sample, color='g', linewidth=2, label="sample") #mean curve.
        plt.fill_between(index, mean_sample-std_sample, mean_sample+std_sample, color='b', alpha=.1)

        plt.plot(index, mean_random, color='y', linewidth=2, label="random") #mean curve.
        plt.fill_between(index, mean_random-std_random, mean_random+std_random, color='b', alpha=.1)

        plt.legend(loc="upper left")
        plt.suptitle('Energy over Training Iteration', fontsize=16)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Energy', fontsize=12)
        plt.savefig(save_path_name)
