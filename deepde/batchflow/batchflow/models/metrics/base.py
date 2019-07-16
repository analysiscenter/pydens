""" Contains a base metrics class """

import numpy as np


class Metrics:
    """ Base metrics evaluation class

    This class is not supposed to be instantiated.
    Use specific children classes instead (e.g. :class:`.ClassificationMetrics`).

    Examples
    --------

    ::

        m = ClassificationMetrics(targets, predictions, num_classes=10, fmt='labels')
        m.evaluate(['sensitivity', 'specificity'], multiclass='micro')
    """
    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        self._agg_fn_dict = {}

    def _aggregate(self, metric, agg=None):
        """ Aggregate metrics calculated for different batches or instances """
        if not np.isscalar(metric) and agg is not None:
            agg_fn = self._agg_fn_dict.get(agg)
            if agg_fn is None:
                raise ValueError("Unknown aggregation type", agg)
            metric = agg_fn(metric)

        metric = np.squeeze(metric)
        if metric.ndim == 0:
            metric = metric.item()
        return metric

    def evaluate(self, metrics, agg='mean', *args, **kwargs):
        """ Calculates metrics

        Parameters
        ----------
        metrics : str or list of str
            metric names
        agg : str
            inter-batch aggregation type
        args
            metric-specific parameters
        kwargs
            metric-specific parameters

        Returns
        -------
        metric value or dict

            if metrics is a list, then a dict is returned::
            - key - metric name
            - value - metric value
        """
        _metrics = [metrics] if isinstance(metrics, str) else metrics

        res = {}
        for name in _metrics:
            metric_fn = getattr(self, name)
            metric_val = metric_fn(*args, **kwargs)
            res[name] = self._aggregate(metric_val, agg)
        res = res[metrics] if isinstance(metrics, str) else res

        return res
