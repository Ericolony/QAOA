import numpy as np
import netket as nk


class MaxCutEnergy:
    def __init__(self, cf):
        self.cf = cf


    def _graph_to_ising(self, laplacian):
        J = 0.25*(laplacian - np.diag(laplacian.sum(-1)))
        return J


    def _construct_ising_hamiltonian(self, J, offset=0):
        N = J.shape[0]
        # Pauli z Matrix
        sz = [[1., 0.], [0., -1.]]
        # create graph
        edges = []
        for i in range(N):
            for j in range(i, N):
                if J[i,j] != 0.: edges.append([i, j])
        g = nk.graph.CustomGraph(edges)
        # system with spin-1/2 particles
        hi = nk.hilbert.Spin(s=0.5, graph=g)
        ha = nk.operator.LocalOperator(hi, offset)
        for i in range(N):
            for j in range(N):
                if J[i, j] != 0.:
                    ha += J[i, j] * nk.operator.LocalOperator(hi, [np.kron(sz, sz)], [[i, j]])
        # ha.to_dense() for matrix representation of ha
        return ha, g, hi


    def laplacian_to_hamiltonian(self, laplacian):
        J = self._graph_to_ising(laplacian)
        hamiltonian, graph, hilbert = self._construct_ising_hamiltonian(J)
        return hamiltonian, graph, hilbert

