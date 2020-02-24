from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from flowket.machines import RBMSym, RBM


def rbm(cf, input_shape):
    inputs = Input(shape=input_shape, dtype='int8')
    rbm = RBM(inputs, stddev=0.01, use_float64_ops=True)
    predictions = rbm.predictions
    model = Model(inputs=inputs, outputs=predictions)
    return model