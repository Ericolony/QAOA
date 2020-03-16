# import netket
# g=netket.graph.Hypercube(length=10,n_dim=2,pbc=True)
# print(g.n_sites)
# import pdb;pdb.set_trace()


import netket as nk

from netket.layer import SumOutput
from netket.layer import FullyConnected
from netket.layer import Lncosh
from netket.hilbert import Spin
from netket.graph import Hypercube
from netket.machine import FFNN

from netket.layer import ConvolutionalHypercube


# 2D Lattice
from config import get_config
from src.objectives.max_cut import MaxCutEnergy
from src.util.directory import prepare_dirs_and_logger
from src.util.data_loader import load_data
cf, unparsed = get_config()
prepare_dirs_and_logger(cf)
data = load_data(cf)
energy = MaxCutEnergy(cf)
laplacian = data
ha,g,hi = energy.laplacian_to_hamiltonian(laplacian)
import pdb;pdb.set_trace()
colors = [(1,2,0.5)]
g = nk.graph.Hypercube(length=4, n_dim=2, pbc=True)
import pdb;pdb.set_trace()
# # Hilbert space of spins on the graph
# hi = nk.hilbert.Spin(s=0.5, graph=g)

# # Ising spin hamiltonian at the critical point
# ha = nk.operator.Ising(h=3.0, hilbert=hi)

# RBM Spin Machine
# ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)

input_size = hi.size
# layers = (FullyConnected(input_size=input_size,output_size=input_size,use_bias=True),Lncosh(input_size=input_size),SumOutput(input_size=input_size))
layers = (ConvolutionalHypercube(length=4, n_dim=2, input_channels=1, output_channels=1, stride=1, kernel_length=2, use_bias=True),Lncosh(input_size=input_size),SumOutput(input_size=input_size))
ma = FFNN(hi, layers)
# ma = FFNN(hi, layers)

ma.init_random_parameters(seed=1234, sigma=0.01)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(machine=ma)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.1)

# Stochastic reconfiguration
gs = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=0.1,
    method="Sr",
)

gs.run(output_prefix="test", n_iter=50)

result = gs.get_observable_stats()
quant = result['Energy']['Mean']

print(quant)