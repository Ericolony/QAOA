import numpy as np
import netket as nk


<<<<<<< HEAD
class MaxCutEnergy:
    def __init__(self, cf):
        self.cf = cf

    def _graph_to_ising(self, laplacian):
        # where the diagonal corresponds to the identity matrix in the objective
        # H = 1/2*sum_over_edges (I-'sigma_i^z sigma_j^z)
        # 0.25 as multiplier because each edges is counted twice below
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
        # cost hamiltonian (eq2.2, Gomes et al.)
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
=======
def graph_to_ising(laplacian):
    J = 0.25*(laplacian - np.diag(laplacian.sum(-1)))
    return J


def construct_ising_hamiltonian(J, offset=0):
    N = J.shape[0]
    # Pauli z Matrix
    sz = [[1., 0.], [0., -1.]]
    # create graph
    edges = []
    for i in range(N):
        for j in range(i, N):
            if J[i,j] != 0.: edges.append([i, j])
    g = nk.graph.CustomGraph(edges)
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    ha = nk.operator.LocalOperator(hi, offset)
    # cost hamiltonian (eq2.2, Gomes et al.)
    for i in range(N):
        for j in range(N):
            if J[i, j] != 0.:
                ha += J[i, j] * nk.operator.LocalOperator(hi, [np.kron(sz, sz)], [[i, j]])
    return ha


def laplacian_to_hamiltonian(laplacian):
    J = graph_to_ising(laplacian)
    hamiltonian = construct_ising_hamiltonian(J)
    return hamiltonian
>>>>>>> b1250baaa20a9bb578d8d052b6ec67bd5aa80232


# class Energy(torch.nn.Module):
#     def __init__(self, cf):
#         super(Energy, self).__init__()
#         self.cf = cf
#         self.hamiltonian = self._hamiltonian()


#     def _hamiltonian():
#         pass


#     def _local_observable(self, sigma, model):
#         configurations = nonzero_configuration(self.hamiltonian, sigma)
#         probability_ratio = model(configurations) / model(sigma)
#         matrix_element = XXX
#         local_value = (matrix_element * probability_ratio).sum(-1)
#         return local_value
    

#     def forward(self, sigma, model):
#         energy = _local_observable(sigma, model)
#         return energy

