""" Contains base tf losses """
import tensorflow as tf


def softmax_cross_entropy(labels, logits, *args, **kwargs):
    """ Multi-class CE which takes plain or one-hot labels

    Parameters
    ----------
    labels : tf.Tensor

    logits : tf.Tensor

    args
        other positional parameters from `tf.losses.softmax_cross_entropy`
    kwargs
        other named parameters from `tf.losses.softmax_cross_entropy`

    Returns
    -------
    tf.Tensor
    """
    labels_shape = tf.shape(labels)
    logits_shape = tf.shape(logits)
    c = tf.cast(tf.equal(labels_shape, logits_shape), tf.int32)
    e = tf.equal(tf.reduce_sum(c, axis=-1), logits_shape.shape[-1])
    labels = tf.cond(e, lambda: tf.cast(labels, dtype=logits.dtype),
                     lambda: tf.one_hot(tf.cast(labels, tf.int32), logits_shape[-1], dtype=logits.dtype))
    return tf.losses.softmax_cross_entropy(labels, logits, *args, **kwargs)
