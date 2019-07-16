""" Helpers for training """
import tensorflow as tf


def piecewise_constant(global_step, *args, **kwargs):
    """ Constant learning rate decay (uses global_step param instead of x) """
    return tf.train.piecewise_constant(global_step, *args, **kwargs)
