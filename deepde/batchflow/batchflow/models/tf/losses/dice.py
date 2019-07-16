""" Contains various flavours of dice """
import tensorflow as tf


def _dice(targets, predictions, weights=1.0, label_smoothing=0, scope=None,
          loss_collection=tf.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
          _square=False, _per_batch=False, _weights_power=0):

    with tf.name_scope(scope or 'dice'):
        e = 1e-15

        axis = tuple(range(0 if _per_batch else 1, targets.shape.ndims))

        class_axis = -1
        multiclass = targets.shape[class_axis] > 1

        if multiclass:
            predictions = tf.nn.softmax(predictions)

            if label_smoothing > 0:
                num_classes = targets.shape[class_axis]
                targets = targets * (1 - label_smoothing) + label_smoothing / num_classes

            class_weights = tf.reduce_sum(targets, axis=tuple(range(targets.shape.ndims - 1)))
            class_weights = 1. - class_weights / tf.reduce_sum(targets, axis=None)
            class_weights = class_weights / tf.reduce_sum(class_weights)
            axis = axis[:-1]
        else:
            predictions = tf.sigmoid(predictions)
            class_weights = 1.

        true_positive = tf.reduce_sum(targets * predictions, axis=axis)
        false_positive = tf.reduce_sum((1 - targets) * predictions, axis=axis)
        false_negative = tf.reduce_sum(targets * (1 - predictions), axis=axis)

        if _square:
            targets = tf.reduce_sum(tf.square(targets), axis=axis)
            predictions = tf.reduce_sum(tf.square(predictions), axis=axis)
        else:
            targets = tf.reduce_sum(targets, axis=axis)
            predictions = tf.reduce_sum(predictions, axis=axis)

        loss = -(2. * true_positive + e) / (targets + predictions + e)

        if _per_batch:
            if multiclass:
                loss = tf.reduce_sum(loss * class_weights)
            tf.losses.add_loss(loss, loss_collection)
        else:
            if _weights_power > 0:
                item_weights = 2. * true_positive + false_positive + false_negative + e
                item_weights = item_weights / tf.reduce_sum(item_weights)
                item_weights = tf.pow(item_weights, _weights_power)
                item_weights = item_weights / tf.reduce_sum(item_weights)
                loss = loss * item_weights * tf.to_float(tf.shape(targets)[0])
            elif multiclass:
                weights = tf.reshape(class_weights, (1, -1)) * tf.reshape(weights, (-1, 1))
            loss = tf.losses.compute_weighted_loss(loss, weights, None, loss_collection, reduction)

    return loss


def dice(targets, predictions, weights=1.0, label_smoothing=0, scope=None,
         loss_collection=tf.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """ Dice coefficient over batch items

    Parameters
    ----------
    targets : tf.Tensor
        tensor with target values

    predictions : tf.Tensor
        tensor with predicted logits

    Returns
    -------
    Tensor of the same type as targets.
    If reduction is NONE, this has the same shape as targets; otherwise, it is scalar.

    Notes
    -----
    By default, dice will be added to ``loss_collection`` which will affect loss calculation and model training.
    To prevent this, specify ``loss_collection=None``.
    """
    return _dice(targets, predictions, weights, label_smoothing, scope, loss_collection, reduction)


def dice2(targets, predictions, weights=1.0, label_smoothing=0, scope=None,
          loss_collection=tf.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """ Dice coefficient over batch items with squares in denominator

    Parameters
    ----------
    targets : tf.Tensor
        tensor with target values

    predictions : tf.Tensor
        tensor with predicted logits

    Returns
    -------
    Tensor of the same type as targets.
    If reduction is NONE, this has the same shape as targets; otherwise, it is scalar.

    Notes
    -----
    By default, dice will be added to ``loss_collection`` which will affect loss calculation and model training.
    To prevent this, specify ``loss_collection=None``.
    """
    return _dice(targets, predictions, weights, label_smoothing, scope, loss_collection, reduction, _square=True)


def dice_weighted(targets, predictions, weights=1.0, label_smoothing=0, scope=None,
                  loss_collection=tf.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                  weights_power=1):
    """ Dice coefficient weighted by item impact

    Parameters
    ----------
    targets : tf.Tensor
        tensor with target values

    predictions : tf.Tensor
        tensor with predicted logits

    Returns
    -------
    Tensor of the same type as targets.
    If reduction is NONE, this has the same shape as targets; otherwise, it is scalar.

    Notes
    -----
    By default, dice will be added to ``loss_collection`` which will affect loss calculation and model training.
    To prevent this, specify ``loss_collection=None``.
    """
    return _dice(targets, predictions, weights, label_smoothing, scope, loss_collection, reduction,
                 _weights_power=weights_power)


def dice_weighted2(targets, predictions, weights=1.0, label_smoothing=0, scope=None,
                   loss_collection=tf.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                   weights_power=1):
    """ Dice coefficient weighted by item impact with squares in denominator

    Parameters
    ----------
    targets : tf.Tensor
        tensor with target values

    predictions : tf.Tensor
        tensor with predicted logits

    Returns
    -------
    Tensor of the same type as targets.
    If reduction is NONE, this has the same shape as targets; otherwise, it is scalar.

    Notes
    -----
    By default, dice will be added to ``loss_collection`` which will affect loss calculation and model training.
    To prevent this, specify ``loss_collection=None``.
    """
    return _dice(targets, predictions, weights, label_smoothing, scope, loss_collection, reduction,
                 _square=True, _weights_power=weights_power)


def dice_batch(targets, predictions, label_smoothing=0, scope=None, loss_collection=tf.GraphKeys.LOSSES):
    """ Dice coefficient over the whole batch

    Parameters
    ----------
    targets : tf.Tensor
        tensor with target values

    predictions : tf.Tensor
        tensor with predicted logits

    Returns
    -------
    Tensor of the same type as targets.
    If reduction is NONE, this has the same shape as targets; otherwise, it is scalar.

    Notes
    -----
    By default, dice will be added to ``loss_collection`` which will affect loss calculation and model training.
    To prevent this, specify ``loss_collection=None``.
    """
    return _dice(targets, predictions, 1., label_smoothing, scope, loss_collection, _per_batch=True)


def dice_batch2(targets, predictions, label_smoothing=0, scope=None, loss_collection=tf.GraphKeys.LOSSES):
    """ Dice coefficient over the whole batch with squares in denominator

    Parameters
    ----------
    targets : tf.Tensor
        tensor with target values

    predictions : tf.Tensor
        tensor with predicted logits

    Returns
    -------
    Tensor of the same type as targets.
    If reduction is NONE, this has the same shape as targets; otherwise, it is scalar.

    Notes
    -----
    By default, dice will be added to ``loss_collection`` which will affect loss calculation and model training.
    To prevent this, specify ``loss_collection=None``.
    """
    return _dice(targets, predictions, 1., label_smoothing, scope, loss_collection,
                 _per_batch=True, _square=True)
