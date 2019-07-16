""" Contains models for sci-kit learn estimators """

try:
    from sklearn.external import joblib as pickle
except ImportError:
    pass
try:
    import dill as pickle
except ImportError:
    pass
from .base import BaseModel


class SklearnModel(BaseModel):
    """ Base class for scikit-learn models

    Attributes
    ----------
    estimator
        an instance of scikit-learn estimator

    Notes
    -----
    **Configuration**

    estimator - an instance of scikit-learn estimator

    load / path - a path to a pickled estimator

    Examples
    --------
    .. code-block:: python

        pipeline
            .init_model('static', SklearnModel, 'my_model',
                        config={'estimator': sklearn.linear_model.SGDClassifier(loss='huber')})

        pipeline
            .init_model('static', SklearnModel, 'my_model',
                        config={'load': {'path': '/path/to/estimator.pickle'}})
    """
    def __init__(self, *args, **kwargs):
        self.estimator = None
        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        """ Define the model """
        _ = args, kwargs
        self.estimator = self.get('estimator', self.config)

    def load(self, path):
        """ Load the model.

        Parameters
        ----------
        path : str
            a full path to a file from which a model will be loaded
        """
        self.estimator = pickle.load(path)

    def save(self, path):
        """ Save the model.

        Parameters
        ----------
        path : str
            a full path to a file where a model will be saved to
        """
        if self.estimator is not None:
            pickle.dump(self.estimator, path)
        else:
            raise ValueError("Scikit-learn estimator does not exist. Check your config for 'estimator'.")

    def train(self, X, y, *args, **kwargs):
        """ Train the model with the data provided

        Parameters
        ----------
        X : array-like
            Subset of the training data, shape (n_samples, n_features)

        y : numpy array
            Subset of the target values, shape (n_samples,)

        Notes
        -----
        For more details and other parameters look at the documentation for the estimator used.
        """
        if hasattr(self.estimator, 'partial_fit'):
            self.estimator.partial_fit(X, y, *args, **kwargs)
        else:
            self.estimator.fit(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        """ Predict with the data provided

        Parameters
        ----------
        X : array-like
            Subset of the training data, shape (n_samples, n_features)

        Notes
        -----
        For more details and other parameters look at the documentation for the estimator used.

        Returns
        -------
        array
            Predicted value per sample, shape (n_samples,)
        """
        return self.estimator.predict(X, *args, **kwargs)
