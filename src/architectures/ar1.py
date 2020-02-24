
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from flowket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, \
    default_wave_function_stats_callbacks_factory
from flowket.layers import LogSpaceComplexNumberHistograms
from flowket.machines import ConvNetAutoregressive2D
from flowket.operators import Ising
from flowket.optimization import VariationalMonteCarlo, loss_for_energy_minimization
from flowket.samplers import AutoregressiveSampler
from flowket.machines import SimpleConvNetAutoregressive1D, ComplexValuesSimpleConvNetAutoregressive1D


def ar(cf, input_shape):
    inputs = Input(shape=input_shape, dtype='int8')
    convnet = ComplexValuesSimpleConvNetAutoregressive1D(inputs, depth=2, num_of_channels=32)
    predictions, conditional_log_probs = convnet.predictions, convnet.conditional_log_probs
    predictions = LogSpaceComplexNumberHistograms(name='psi')(predictions)
    model = Model(inputs=inputs, outputs=predictions)
    conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_probs)
    return model, conditional_log_probs_model