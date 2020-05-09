import numpy as np
import netket as nk


class SpinGlassEnergy:
    def __init__(self, cf):
        self.cf = cf


    def _construct_ising_hamiltonian(self, J, offset=0):
        # Edwards-Anderson model
        N = J.shape[0]
        # Pauli z Matrix
        sz = [[1., 0.], [0., -1.]]
        # # create graph
        g = nk.graph.Hypercube(length=int(np.sqrt(N)), n_dim=2, pbc=True)
        # system with spin-1/2 particles
        hi = nk.hilbert.Spin(s=0.5, graph=g)
        ha = nk.operator.LocalOperator(hi, offset)
        for i in range(N):
            for j in range(i,N):
                ha += -J[i, j] * nk.operator.LocalOperator(hi, [np.kron(sz, sz)], [[i, j]])
        # ha.to_dense() for matrix representation of ha
        return ha, g, hi


    def laplacian_to_hamiltonian(self, J):
        hamiltonian, graph, hilbert = self._construct_ising_hamiltonian(J)
        return hamiltonian, graph, hilbert