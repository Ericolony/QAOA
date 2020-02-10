import tensorflow as tf

from flowket.layers import ToComplex64, ToComplex128, ComplexConv1D
from flowket.deepar.layers import PeriodicPadding
from flowket.layers.complex.tensorflow_ops import lncosh

from tensorflow.keras.layers import Input, Flatten, Activation, Lambda
from tensorflow.keras.models import Model

def drbm(cf):
    hilbert_state_shape = (cf.input_size, 1)
    padding = ((0, cf.kernel_size - 1),)
    inputs = Input(shape=hilbert_state_shape, dtype='int8')
    x = ToComplex128()(inputs)
    for i in range(cf.depth):
        x = PeriodicPadding(padding)(x)
        x = ComplexConv1D(cf.width, cf.kernel_size, use_bias=False, dtype=tf.complex128)(x)
        x = Activation(lncosh)(x)
    x = Flatten()(x)
    predictions = Lambda(lambda y: tf.reduce_sum(y, axis=1, keepdims=True))(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model


class RBM:
    ''' Implementation of the Restricted Boltzmann Machine for collaborative filtering. The model is based on the paper of 
        Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton: https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf
    '''
    def __init__(self, FLAGS):
        '''Initialization of the model  '''
        self.FLAGS=FLAGS
        self.weight_initializer=model_helper._get_weight_init()
        self.bias_initializer=model_helper._get_bias_init()
        self.init_parameter()
        

    def init_parameter(self):
        ''' Initializes the weights and the bias parameters of the neural network.'''

        with tf.variable_scope('Network_parameter'):
            self.W=tf.get_variable('Weights', shape=(self.FLAGS.num_v, self.FLAGS.num_h),initializer=self.weight_initializer)
            self.bh=tf.get_variable('hidden_bias', shape=(self.FLAGS.num_h), initializer=self.bias_initializer)
            self.bv=tf.get_variable('visible_bias', shape=(self.FLAGS.num_v), initializer=self.bias_initializer)