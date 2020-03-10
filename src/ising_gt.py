import time
import numpy as np
import ising
from src.util.plottings import laplacian_to_graph
import matplotlib.pyplot as plt

def decode_state(state_repr, no_spins, labels):
    state = {}
    for i in range(no_spins):
        state[labels[no_spins-(i+1)]] = 1 if state_repr % 2 else -1
        state_repr //= 2
    return state


def check(dic, graph):
    total = 0
    for edge, energy in graph.items():
        if edge[0] == edge[1]:
            total += energy*dic[edge[0]]
        else:
            total += energy*dic[edge[0]]*dic[edge[1]]
    return -total


def ising_ground_truth(cf, info_mtx, fig_save_path=""):
    print("Running the ground truth...")
    if cf.pb_type == "maxcut":
        # 1/2\sum_edges 1-node1*node2 = 1/2*num_edges - 1/2*\sum_edges node1*node2
        num_edges = info_mtx.sum()/2
        G = -0.5*info_mtx
        graph = {}
        shape = G.shape
        dim = shape[0]
        for i in range(shape[0]):
            for j in range(i, shape[1]):
                graph[(i,j)] = G[i,j]
        start_time = time.time()
        result = ising.search(graph, num_states=3, show_progress=True, chunk_size=0)
        end_time = time.time()
        energy = result.energies[0]
        state = decode_state(result.states[0], dim, list(range(dim)))
        energy1 = check(state, graph)
        if abs((energy-energy1)/energy) > 1e-2:
            print("Mismatched energy - result1={}, result2={}".format(energy, energy1))
            raise
        quant = num_edges/2 - energy


        # plot the graph
        import maxCutPy.maxcutpy.graphdraw as gd
        # Laplacian matrices are real and symmetric, so we can use eigh, 
        # the variation on eig specialized for Hermetian matrices.
        G = laplacian_to_graph(info_mtx)

        import maxCutPy.maxcutpy.graphcut as gc
        import maxCutPy.maxcutpy.graphtest as gt
        import maxCutPy.maxcutpy.maxcut as mc
        # Laplacian matrices are real and symmetric, so we can use eigh, 
        # the variation on eig specialized for Hermetian matrices.
        # method from maxCutPy
        gt.execution_time(mc.local_consistent_max_cut, 1, G)
        quant1 = gc.cut_edges(G)
        if abs((quant-quant1)) != 0:
            print("Mismatched maxcut - result1={}, result2={}".format(quant, quant1))
            raise

        nbunch = G.nodes()
        for i in nbunch:
            G.node[i]['partition'] = state[i]
        gd.draw_cut_graph(G)
        plt.savefig(fig_save_path)
        plt.close()
    if cf.pb_type == "spinglass":
        # \sum_edges J_ij*node1*node2
        G = info_mtx
        graph = {}
        shape = G.shape
        dim = shape[0]
        for i in range(shape[0]):
            for j in range(i, shape[1]):
                graph[(i,j)] = G[i,j]
        start_time = time.time()
        result = ising.search(graph, num_states=3, show_progress=True, chunk_size=0)
        end_time = time.time()
        energy = result.energies[0]
        state = decode_state(result.states[0], dim, list(range(dim)))
        energy1 = check(state, graph)
        if abs((energy-energy1)/energy) > 1e-2:
            print("Mismatched energy - result1={}, result2={}".format(energy, energy1))
            raise
        quant = energy
    time_ellapsed = end_time - start_time
    return quant, state, time_ellapsed

if __name__ == '__main__':
    from config import get_config
    cf, unparsed = get_config()
    num = 13
    laplacian = np.load("./data/maxcut/graph({}, 1).npy".format(num))
    J_mtx = laplacian
    print(ising_ground_truth(cf, laplacian, J_mtx))