import numpy as np
import netket as nk


class IsingEnergy:
    def __init__(self, cf):
        self.cf = cf

    def _graph_to_ising(self, laplacian):
        J = 0.25*(laplacian - np.diag(laplacian.sum(-1)))
        return J

    def _construct_ising_hamiltonian(self, J, offset=0):
        h = 0
        N = J.shape[0]
        # create graph
        edges = []
        for i in range(N):
            for j in range(i, N):
                if J[i,j] != 0.: edges.append([i, j])
        g = nk.graph.CustomGraph(edges)
        # Spin based Hilbert Space
        hi = nk.hilbert.Spin(s=0.5, graph=g)
        # Pauli Matrices
        sigmaz = np.array([[1, 0], [0, -1]])
        sigmax = np.array([[0, 1], [1, 0]])
        operators = []
        sites = []
        # Local Field term
        for i in range(N):
            operators.append((h*sigmax).tolist())
            sites.append([i])
        # Ising iteraction
        for i in range(N):
            for j in range(N):
                if J[i, j] != 0.:
                    operators.append((J[i, j]*np.kron(sigmaz,sigmaz)).tolist())
                    sites.append([i, j])
        ha = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)
        # res = nk.exact.lanczos_ed(ha, first_n=1, compute_eigenvectors=False)
        # print("Exact transverse Ising ground state energy = {0:.3f}".format(res.eigenvalues[0]))
        # ha.to_dense() for matrix representation of ha
        return ha, g, hi

    def laplacian_to_hamiltonian(self, laplacian):
        J = self._graph_to_ising(laplacian)
        hamiltonian, graph, hilbert = self._construct_ising_hamiltonian(J)
        return hamiltonian, graph, hilbert


if __name__ == '__main__':
    aa = IsingEnergy()
    ha, hi = aa._construct_ising_hamiltonian()
    print(ha.to_dense())
