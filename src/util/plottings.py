import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy as np

def plot_graph(L, save_path):
    # Laplacian matrices are real and symmetric, so we can use eigh, 
    # the variation on eig specialized for Hermetian matrices.
    N = L.shape[0]
    w, v = eigh(L) # w = eigenvalues, v = eigenvectors

    edges = []
    for i in range(N):
        for j in range(i, N):
            if L[i,j] != 0.: edges.append([i, j])

    x = v[:,1] 
    y = v[:,2]
    spectral_coordinates = {i : (x[i], y[i]) for i in range(N)}
    G = nx.Graph()
    G.add_edges_from(edges)

    nx.draw(G, pos=spectral_coordinates, with_labels=True, font_color='white')
    plt.savefig(save_path)


def plot_train_curve(result_sample, result_random, save_path_name):
    index = np.arange(1, result_sample.shape[0]+1)
    mean_sample = np.mean(result_sample, 1)
    std_sample = np.std(result_sample, 1)
    mean_random = np.mean(result_random, 1)
    std_random = np.std(result_random, 1)

    plt.plot(index, mean_sample, color='g', linewidth=2, label="sample") #mean curve.
    plt.fill_between(index, mean_sample-std_sample, mean_sample+std_sample, color='b', alpha=.1)

    plt.plot(index, mean_random, color='y', linewidth=2, label="random") #mean curve.
    plt.fill_between(index, mean_random-std_random, mean_random+std_random, color='b', alpha=.1)

    plt.legend(loc="upper left")
    fig.suptitle('test title', fontsize=20)
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Weight', fontsize=16)
    plt.savefig(save_path_name)
