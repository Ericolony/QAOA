
import tensorflow as tf
from tensorflow.keras.layers import Add, Flatten, Lambda

from tensorflow.keras.layers import Input, Flatten, Activation, Lambda
from flowket.layers import ToComplex128, ToComplex64, ComplexDense
from tensorflow.keras.models import Model

def my_rbm(cf):
    hilbert_state_shape = (cf.input_size, 1)

    weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.5)
    bias_initializer = tf.zeros_initializer()

    inputs = Input(shape=hilbert_state_shape, dtype='int8')
    x = ToComplex64()(inputs)
    x = tf.squeeze(x, -1)
    c = ComplexDense(cf.input_size, use_bias=True)(x)
    lncoshc = 2*tf.math.log(tf.math.cosh(c))
    lncoshc_sum = tf.math.reduce_sum(lncoshc, 1)
    sigma_a = ComplexDense(1, use_bias=False)(x)
    sigma_a = tf.math.reduce_sum(sigma_a, 1)
    predictions = tf.expand_dims(sigma_a + lncoshc_sum, 1)
    model = Model(inputs=inputs, outputs=predictions)

    # # real
    # inputs = Input(shape=hilbert_state_shape, dtype='int8')
    # x = tf.cast(inputs, tf.float32)
    # c = tf.keras.layers.Dense(cf.input_size, use_bias=True)(x)
    # lncoshc = 2*tf.math.log(tf.math.cosh(c))
    # lncoshc_sum = tf.math.reduce_sum(lncoshc, [1,2])
    # sigma_a = tf.keras.layers.Dense(1)(x)
    # sigma_a = tf.math.reduce_sum(sigma_a, [1,2])
    # predictions = tf.expand_dims(sigma_a + lncoshc_sum, 1)
    # model = Model(inputs=inputs, outputs=predictions)
    return model
