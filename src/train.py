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
from flowket.samplers import AutoregressiveSampler

import tensorflow as tf
from tensorflow.python.ops.parallel_for import gradients
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.backend as K

from src.objectives.max_cut import laplacian_to_hamiltonian
from src.util.helper import evaluate
from src.util.models import build_model
from src.util.plottings import plot_train_curve


###############################################################################
################################ FlowKet ######################################
###############################################################################

K.set_floatx('float64')

def run_pyket(cf, laplacian):
    hilbert_state_shape = (cf.input_size, 1)

    # build model
    model, conditional_log_probs_model = build_model(cf)

    # # build optimizer
    # if cf.fast_jacobian:
    #     predictions_jacobian = lambda x: get_predictions_jacobian(keras_model=model)
    # else:
    #     predictions_jacobian = lambda x: gradients.jacobian(tf.real(model.output), x, use_pfor=not cf.no_pfor)
    # if cf.use_stochastic_reconfiguration:
    #     optimizer = ComplexValuesStochasticReconfiguration(model, predictions_jacobian,
    #                                                        lr=cf.learning_rate, diag_shift=10.0, 
    #                                                        iterative_solver=cf.use_iterative,
    #                                                        use_cholesky=cf.use_cholesky,
    #                                                        iterative_solver_max_iterations=None)
    #     model.compile(optimizer=optimizer, loss=loss_for_energy_minimization, metrics=optimizer.metrics)
    # else:
    #     optimizer = SGD(lr=cf.learning_rate)
    #     model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
    optimizer = Adam(lr=cf.learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
    
    # build objective
    hamiltonian = laplacian_to_hamiltonian(laplacian)
    operator = NetketOperatorWrapper(hamiltonian, hilbert_state_shape)

    # # build sampler
    # sampler = MetropolisHastingsLocal(model, cf.batch_size,
    #                                           num_of_chains=cf.pyket_num_of_chains,
    #                                           unused_sampels=numpy.prod(hilbert_state_shape))
    # variational_monte_carlo = VariationalMonteCarlo(model, operator, sampler)
    sampler = AutoregressiveSampler(conditional_log_probs_model, cf.batch_size)
    variational_monte_carlo = VariationalMonteCarlo(model, operator, sampler)
    validation_sampler = AutoregressiveSampler(conditional_log_probs_model, cf.batch_size * 16)
    validation_generator = VariationalMonteCarlo(model, operator, validation_sampler)

    # set up tensorboard logger
    tensorboard = TensorBoardWithGeneratorValidationData(log_dir=os.path.join(cf.dir,'tensorboard_logs/try_%s_run_1' % cf.batch_size),
                                                        generator=variational_monte_carlo, update_freq=1,
                                                        histogram_freq=1, batch_size=cf.batch_size, write_output=False)
    callbacks = default_wave_function_stats_callbacks_factory(variational_monte_carlo,
                                                            validation_generator=validation_generator,
                                                            true_ground_state_energy=-37) + [tensorboard]

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
    # model.fit_generator(variational_monte_carlo.to_generator(), steps_per_epoch=cf.num_of_iterations, epochs=1, callbacks=callbacks,
    #                     max_queue_size=0, workers=0)
    end_time = time.time()
    plot_train_curve(result_sample, result_random, os.path.join(cf.dir, "train_curve.png"))
    return end_time - start_time

###############################################################################
###############################################################################
###############################################################################



###############################################################################
################################ NetKet #######################################
###############################################################################
def run_netket(cf, laplacian):
    g = nk.graph.Hypercube(length=cf.input_size, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)
    ha = laplacian_to_hamiltonian(laplacian)

    ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
    ma.init_random_parameters(seed=1234, sigma=0.01)
    sa = nk.sampler.MetropolisLocal(machine=ma)

    op = nk.optimizer.Sgd(learning_rate=cf.learning_rate)
    method = 'Sr' if cf.use_stochastic_reconfiguration else 'Gd'

##############################################################################
    # gs = nk.variational.Vmc(
    #     hamiltonian=ha,
    #     sampler=sa,
    #     method=method,
    #     optimizer=op,
    #     n_samples=cf.batch_size,
    #     use_iterative=cf.use_iterative,
    #     use_cholesky=cf.use_cholesky,
    #     diag_shift=10.0)
    gs = nk.variational.Vmc(
        hamiltonian=ha,
        sampler=sa,
        optimizer=op,
        n_samples=ma.n_par,
        diag_shift=0.1,
        use_iterative=True,
        method='Sr')
##############################################################################

    gs.run(output_prefix="ffnn_test", n_iter=5, save_params_every=5)
    start_time = time.time()
    gs.run(output_prefix="ffnn_test", n_iter=cf.num_of_iterations, save_params_every=cf.num_of_iterations)
    end_time = time.time()
    return end_time - start_time
###############################################################################
###############################################################################
###############################################################################
