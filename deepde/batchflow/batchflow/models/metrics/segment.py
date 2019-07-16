""" Contains metrics for segmentation """
import numpy as np

from . import ClassificationMetrics, get_components


class SegmentationMetricsByPixels(ClassificationMetrics):
    """ Metrics to assess segmentation models pixel-wise

    Notes
    -----
    Rate metrics are evaluated for each item independently. So there are two levels of metrics aggregation:

    - multi-class averaging
    - dataset aggregation.

    For instance, if you have a dataset of 100 pictures (each having size of 256x256) of 10 classes and
    you need to calculate an accuracy of semantic segmentation, then:

    - `evaluate(['accuracy'], agg=None, multiclass=None)` will return an array of shape (100, 10) containing accuracy
      of each class for each image separately.

    - `evaluate(['accuracy'], agg='mean', multiclass=None)` will return a vector of shape (10,) containing
      an accuracy of each class averaged across all images.

    - `evaluate(['accuracy'], agg=None, multiclass='macro')` will return a vector of shape (100,) containing
      an accuracy of each image separately averaged across all classes.

    - `evaluate(['accuracy'], agg='mean', multiclass='macro')` will return a single value of
      an average accuracy of all classes and images combined.

    The default values are `agg='mean', multiclass='macro'`.

    For multi-class averaging see :class:`~.ClassificationMetrics`.

    Examples
    --------
    ::

        metrics = SegmentationMetricsByPixels(targets, predictions, num_classes=10, fmt='labels')
        metrics.evaluate('specificity')
        metrics.evaluate(['sensitivity', 'jaccard'], agg='mean', multiclass=None)

    """
    pass

class SegmentationMetricsByInstances(ClassificationMetrics):
    """ Metrics to assess segmentation models by instances (i.e. connected components of one class,
    e.g. cancer nodules, faces, )

    Parameters
    ----------
    iot : float
        if the ratio of a predicted instance size to the corresponding target size >= `iot`,
        then instance is considered correctly predicted (true postitive).

    Notes
    -----
    For other parameters see :class:`~.ClassificationMetrics`.

    """
    def __init__(self, targets, predictions, fmt='proba', num_classes=None, axis=None,
                 skip_bg=True, threshold=.5, iot=.5, calc=True):
        super().__init__(targets, predictions, fmt, num_classes, axis, threshold, skip_bg, calc=False)

        self.iot = iot
        self.target_instances = self._get_instances(self.one_hot(self.targets), axis)
        self.predicted_instances = self._get_instances(self.one_hot(self.predictions), axis)
        if calc:
            self._calc()

    def free(self):
        """ Free memory allocated for intermediate data """
        super().free()
        self.target_instances = None
        self.predicted_instances = None

    def _get_instances(self, inputs, axis):
        """ Find instances of each class within inputs

        Parameters
        ----------
        inputs : np.ndarray
            one-hot array
        axis : int
            a class axis

        Returns
        -------
        nested list with ndarray of coords
            num_classes - 1, batch_items, num_instances, inputs.shape, number of pixels
        """
        if axis is None:
            instances = [get_components(inputs, batch=True)]
        else:
            instances = []
            shape = [slice(None)] * inputs.ndim
            for i in range(1, inputs.shape[axis]):
                shape[axis] = i
                one_class = get_components(inputs[shape], batch=True)
                instances.append(one_class)
        return instances

    def _calc(self):
        self._confusion_matrix = np.zeros((self.targets.shape[0], self.num_classes - 1, 2, 2), dtype=np.intp)

        for k in range(1, self.num_classes):
            for i, item_instances in enumerate(self.target_instances[k-1]):
                for coords in item_instances:
                    targ = len(coords[0])
                    pred = np.sum(self.predictions[i][coords] == k)
                    if np.sum(pred) / targ >= self.iot:
                        self._confusion_matrix[i, k-1, 1, 1] += 1
                    else:
                        self._confusion_matrix[i, k-1, 0, 1] += 1

        for k in range(1, self.num_classes):
            for i, item_instances in enumerate(self.predicted_instances[k-1]):
                for coords in item_instances:
                    pred = len(coords[0])
                    targ = np.sum(self.targets[i][coords] == k)
                    if targ == 0 or pred / targ < self.iot:
                        self._confusion_matrix[i, k-1, 1, 0] += 1

    def true_positive(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self._confusion_matrix[:, l-1, 1, 1], label)

    def true_negative(self, label=None, *args, **kwargs):
        raise ValueError("True negative is inapplicable for instance-based metrics")

    def condition_positive(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self._confusion_matrix[:, l-1, :, 1].sum(axis=1), label)

    def prediction_positive(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self._confusion_matrix[:, l-1, 1].sum(axis=1), label)

    def total_population(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._return(self._confusion_matrix[:, label - 1].sum(axis=(1, 2)))
