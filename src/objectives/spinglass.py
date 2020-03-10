import numpy as np
import netket as nk

class SpinGlassEnergy:
    def __init__(self, cf):
        self.cf = cf

    def _graph_to_ising(self, laplacian):
        # where the diagonal corresponds to the identity matrix in the objective
        # Edwards-Anderson model
        # H = -sum_over_edges J_{ij} S_iS_j
        # H = 1/2*sum_over_edges (I-'sigma_i^z sigma_j^z)
        laplacian
        return lap


    def _construct_ising_hamiltonian(self, J, offset=0):
        N = J.shape[0]
        # Pauli z Matrix
        sz = [[1., 0.], [0., -1.]]
        # create graph
        edges = []
        for i in range(N):
            for j in range(i, N):
                edges.append([i, j])
        g = nk.graph.CustomGraph(edges)
        # system with spin-1/2 particles
        hi = nk.hilbert.Spin(s=0.5, graph=g)
        ha = nk.operator.LocalOperator(hi, offset)
        # cost hamiltonian (eq2.2, Gomes et al.)
        for i in range(N):
            for j in range(i,N):
                ha += -J[i, j] * nk.operator.LocalOperator(hi, [np.kron(sz, sz)], [[i, j]])
        # ha.to_dense() for matrix representation of ha
        return ha, g, hi


    def laplacian_to_hamiltonian(self, J):
        hamiltonian, graph, hilbert = self._construct_ising_hamiltonian(J)
        return hamiltonian, graph, hilbert