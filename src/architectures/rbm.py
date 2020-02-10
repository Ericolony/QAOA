from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from flowket.machines import RBMSym, RBM


def rbm(cf):
    hilbert_state_shape = (cf.input_size, 1)
    inputs = Input(shape=hilbert_state_shape, dtype='int8')
    rbm = RBM(inputs, stddev=0.1, use_float64_ops=True)
    predictions = rbm.predictions
    model = Model(inputs=inputs, outputs=predictions)
    return model