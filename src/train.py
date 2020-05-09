import netket as nk
import os
import time

from src.util.models import build_model_netket
from src.objectives.max_cut import MaxCutEnergy
from src.objectives.spinglass import SpinGlassEnergy

###############################################################################
################################ NetKet #######################################
###############################################################################
def run_netket(cf, data, seed):
    # build objective
    if cf.pb_type == "maxcut":
        energy = MaxCutEnergy(cf)
        laplacian = data
        hamiltonian,graph,hilbert = energy.laplacian_to_hamiltonian(laplacian)
    if cf.pb_type == "spinglass":
        energy = SpinGlassEnergy(cf)
        J = data
        hamiltonian,graph,hilbert = energy.laplacian_to_hamiltonian(J)

    # build model
    model = build_model_netket(cf, hilbert)
    model.init_random_parameters(seed=seed, sigma=cf.param_init)
    sampler = nk.sampler.MetropolisLocal(machine=model)
    sampler.seed(seed)

    # build optimizer
    if cf.optimizer == "adadelta":
        op = nk.optimizer.AdaDelta()
    elif cf.optimizer == "adagrad":
        op = nk.optimizer.AdaGrad(learning_rate=cf.learning_rate)
    elif cf.optimizer == "adamax":
        op = nk.optimizer.AdaMax(alpha=cf.learning_rate)
    elif cf.optimizer == "momentum":
        op = nk.optimizer.Momentum(learning_rate=cf.learning_rate)
    elif cf.optimizer == "rmsprop":
        op = nk.optimizer.RmsProp(learning_rate=cf.learning_rate)
    elif cf.optimizer == "sgd":
        op = nk.optimizer.Sgd(learning_rate=cf.learning_rate, decay_factor=cf.decay_factor)

    if cf.use_sr:
        method = "Sr"
    else:
        method = "Gd"

    # build algorithm
    gs = nk.variational.Vmc(
        hamiltonian=hamiltonian,
        sampler=sampler,
        method=method,
        optimizer=op,
        n_samples=cf.batch_size,
        use_iterative=cf.use_iterative,
        use_cholesky=cf.use_cholesky,
        diag_shift=0.1)
    # run
    start_time = time.time()
    gs.run(output_prefix=os.path.join(cf.dir,"result"), n_iter=cf.num_of_iterations, save_params_every=cf.num_of_iterations)
    end_time = time.time()
    # type(gs).__name__
    result = gs.get_observable_stats()

    score = result['Energy']['Mean']
    time_elapsed = end_time - start_time
    # exp_name,_,_ = (cf.dir).partition('-date')
    exp_name = cf.framework + str(cf.input_size)
    return exp_name, score, time_elapsed
###############################################################################
###############################################################################
###############################################################################
