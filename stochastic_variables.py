import tensorflow as tf
import math
from tensorflow.contrib.rnn import GRUBlockCell

from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope


def gaussian_mixture_nll(samples, mixing_weights, mean1, mean2, std1, std2):
    """
    Computes the NLL from a mixture of two gaussian distributions with the given
    means and standard deviations, mixing weights and samples.
    """
    gaussian1 = (1.0/tf.sqrt(2.0 * std1 * math.pi)) * tf.exp(- tf.square(samples - mean1) / (2.0 * std1))
    gaussian2 = (1.0/tf.sqrt(2.0 * std2 * math.pi)) * tf.exp(- tf.square(samples - mean2) / (2.0 * std2))

    mixture = (mixing_weights[0] * gaussian1) + (mixing_weights[1] * gaussian2)

    return - tf.log(mixture)


def get_random_normal_variable(name, mean, standard_dev, shape, dtype):

    """
    A wrapper around tf.get_variable which lets you get a "variable" which is
     explicitly a sample from a normal distribution.
    """

    # Inverse of a softplus function, so that the value of the standard deviation
    # will be equal to what the user specifies, but we can still enforce positivity
    # by wrapping the standard deviation in the softplus function.
    standard_dev = tf.log(tf.exp(standard_dev) - 1.0) * tf.ones(shape)

    mean = tf.get_variable(name + "_mean", shape,
                           initializer=tf.constant_initializer(mean),
                           dtype=dtype)
    standard_deviation = tf.get_variable(name + "_standard_deviation",
                                         initializer=standard_dev,
                                         dtype=dtype)

    standard_deviation = tf.nn.softplus(standard_deviation)
    weights = mean + (standard_deviation * tf.random_normal(shape, 0.0, 1.0, dtype))
    return weights, mean, standard_deviation


class ExternallyParameterisedGRU(GRUBlockCell):
    """
    A simple extension of an GRU in which the weights are passed in to the class,
    rather than being automatically generated inside the cell when it is called.
    This allows us to parameterise them in other, funky ways.
    """

    def __init__(self, weight, bias, **kwargs):
        self.weight = weight
        self.bias = bias
        super(ExternallyParameterisedGRU, self).__init__(**kwargs)

    def __call__(self, inputs, h, scope=None):
        """Block GRU cell."""
        with _checked_scope(self, scope or "block_gru_cell", reuse=self._reuse):
            # Parameters of gates are concatenated into one multiply for efficiency.

            W_z, W_r, W = tf.split(value=self.weight, num_or_size_splits=3, axis=1)
            b_z, b_r, b = tf.split(value=self.bias, num_or_size_splits=3, axis=1)
            all_inputs = tf.concat([inputs, h], 1)

            # z = update_gate, r = reset_gate
            z = tf.sigmoid(tf.nn.bias_add(tf.mutmul(all_inputs, W_z), b_z))
            r = tf.sigmoid(tf.nn.bias_add(tf.mutmul(all_inputs, W_r), b_r))
            h_hat = self._activation(tf.nn.bias_add(tf.mutmul(tf.concat([inputs, r * h], 1), W), b))

            new_h = (1 - z) * h + z * h_hat

            return new_h, new_h