from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from flowket.machines import RBMSym, RBM


<<<<<<< HEAD
def rbm(cf, input_shape):
    inputs = Input(shape=input_shape, dtype='int8')
    rbm = RBM(inputs, stddev=0.01, use_float64_ops=True)
=======
def rbm(cf):
    hilbert_state_shape = (cf.input_size, 1)
    inputs = Input(shape=hilbert_state_shape, dtype='int8')
    rbm = RBM(inputs, stddev=0.1, use_float64_ops=True)
>>>>>>> b1250baaa20a9bb578d8d052b6ec67bd5aa80232
    predictions = rbm.predictions
    model = Model(inputs=inputs, outputs=predictions)
    return model