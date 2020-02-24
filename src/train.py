import netket as nk
import numpy as np
import os
import argparse
import time
import numpy
import functools

import netket as nk

from flowket.operators import NetketOperatorWrapper
from flowket.utils.jacobian import predictions_jacobian as get_predictions_jacobian
from flowket.optimizers import ComplexValuesStochasticReconfiguration
from flowket.optimization import VariationalMonteCarlo, loss_for_energy_minimization
from flowket.samplers import MetropolisHastingsLocal
from flowket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, \
    default_wave_function_stats_callbacks_factory
from flowket.samplers import AutoregressiveSampler, FastAutoregressiveSampler, MetropolisHastingsHamiltonian

import tensorflow as tf
from tensorflow.python.ops.parallel_for import gradients
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.backend as K

from src.util.helper import evaluate
from src.util.models import build_model
from src.util.plottings import plot_train_curve


###############################################################################
################################ FlowKet ######################################
###############################################################################

K.set_floatx('float64')

def run_pyket(cf, data):
    hilbert_state_shape = cf.input_size
    # build model
    model, conditional_log_probs_model = build_model(cf, hilbert_state_shape)

    # build optimizer
    if cf.optimizer == "sr":
        if cf.fast_jacobian:
            predictions_jacobian = lambda x: get_predictions_jacobian(keras_model=model)
        else:
            predictions_jacobian = lambda x: gradients.jacobian(tf.real(model.output), x, use_pfor=not cf.no_pfor)
        optimizer = ComplexValuesStochasticReconfiguration(model, predictions_jacobian, name="sr",
                                                           lr=cf.learning_rate, diag_shift=0.1, 
                                                           iterative_solver=cf.use_iterative,
                                                           use_cholesky=cf.use_cholesky,
                                                           iterative_solver_max_iterations=None)
        model.compile(optimizer=optimizer, loss=loss_for_energy_minimization, metrics=optimizer.metrics)
    elif cf.optimizer == "sgd":
        optimizer = SGD(lr=cf.learning_rate, momentum=0.9)
        model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
    elif cf.optimizer == "adam":
        optimizer = Adam(lr=cf.learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)

    
    # build objective
    if cf.pb_type == "maxcut":
        from src.objectives.max_cut import MaxCutEnergy
        energy = MaxCutEnergy(cf)
        laplacian = data
        hamiltonian,_,_ = energy.laplacian_to_hamiltonian(laplacian)
    if cf.pb_type == "spinglass":
        from src.objectives.spinglass import SpinGlassEnergy
        laplacian, J = data
        energy = SpinGlassEnergy(cf)
        hamiltonian,_,_ = energy.laplacian_to_hamiltonian(laplacian, J)

    operator = NetketOperatorWrapper(hamiltonian, hilbert_state_shape)

    # sampler = MetropolisHastingsHamiltonian(model, cf.batch_size, operator, num_of_chains=16, unused_sampels=numpy.prod(hilbert_state_shape))
    sampler = FastAutoregressiveSampler(conditional_log_probs_model, cf.batch_size)
    # sampler = AutoregressiveSampler(conditional_log_probs_model, cf.batch_size)
    variational_monte_carlo = VariationalMonteCarlo(model, operator, sampler)

    # # set up tensorboard logger
    # validation_sampler = AutoregressiveSampler(conditional_log_probs_model, cf.batch_size * 16)
    # validation_generator = VariationalMonteCarlo(model, operator, validation_sampler)
    # tensorboard = TensorBoardWithGeneratorValidationData(log_dir=os.path.join(cf.dir,'tensorboard_logs/try_%s_run_1' % cf.batch_size),
    #                                                     generator=variational_monte_carlo, update_freq=1,
    #                                                     histogram_freq=1, batch_size=cf.batch_size, write_output=False)
    # callbacks = default_wave_function_stats_callbacks_factory(variational_monte_carlo,
    #                                                         validation_generator=validation_generator,
    #                                                         true_ground_state_energy=-37) + [tensorboard]

    # # warmup
    # model.fit_generator(variational_monte_carlo.to_generator(), steps_per_epoch=5, epochs=1, max_queue_size=0, workers=0)

    # train
    result_sample = np.zeros([int(cf.num_of_iterations//cf.log_interval), cf.batch_size])
    result_random = np.zeros([int(cf.num_of_iterations//cf.log_interval), cf.batch_size])

    start_time = time.time()
    for i in range(int(cf.num_of_iterations//cf.log_interval)):
        model.fit_generator(variational_monte_carlo.to_generator(), steps_per_epoch=cf.log_interval, epochs=1, max_queue_size=0, workers=0)
        energy_sample, energy_random = evaluate(sampler, operator)
        result_sample[i] = energy_sample
        result_random[i] = energy_random
    end_time = time.time()
    plot_train_curve(cf, result_sample, result_random, os.path.join(cf.dir, "train_curve.png"))
    return end_time - start_time

###############################################################################
###############################################################################
###############################################################################



###############################################################################
################################ NetKet #######################################
###############################################################################
def run_netket(cf, data):
    # build objective
    if cf.pb_type == "maxcut":
        from src.objectives.max_cut import MaxCutEnergy
        energy = MaxCutEnergy(cf)
        laplacian = data
        hamiltonian,graph,hilbert = energy.laplacian_to_hamiltonian(laplacian)
    if cf.pb_type == "spinglass":
        from src.objectives.spinglass import SpinGlassEnergy
        file1 = open("data.txt","r")
        laplacian, J = data
        energy = SpinGlassEnergy(cf)
        hamiltonian,graph,hilbert = energy.laplacian_to_hamiltonian(laplacian, J)

    model = nk.machine.RbmSpin(alpha=1, hilbert=hilbert)
    model.init_random_parameters(seed=1234, sigma=0.01)
    sampler = nk.sampler.MetropolisLocal(machine=model)

    op = nk.optimizer.Sgd(learning_rate=cf.learning_rate)
    if cf.optimizer == "sr":
        method = "Sr"
    elif cf.optimizer == "sgd":
        method = "Gd"
    gs = nk.variational.Vmc(
        hamiltonian=hamiltonian,
        sampler=sampler,
        method=method,
        optimizer=op,
        n_samples=cf.batch_size,
        use_iterative=cf.use_iterative,
        use_cholesky=cf.use_cholesky,
        diag_shift=0.1)

    gs.run(output_prefix=os.path.join(cf.dir,"result"), n_iter=5, save_params_every=5)
    start_time = time.time()
    gs.run(output_prefix=os.path.join(cf.dir,"result"), n_iter=cf.num_of_iterations, save_params_every=cf.num_of_iterations)
    end_time = time.time()
    return end_time - start_time
###############################################################################
###############################################################################
###############################################################################
