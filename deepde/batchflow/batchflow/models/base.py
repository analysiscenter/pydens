""" Contains a base model class"""

from ..config import Config

class BaseModel:
    """ Base class for all models

    Attributes
    ----------
    name : str
        a model name
    config : dict
        configuration parameters

    Notes
    -----

    **Configuration**:

    * build : bool or 1 or 'first'
        whether to build a model by calling `self.build()`. Default is True.
        If build == 1 or 'first', then build is called before load, otherwise afterwards.
    * load : dict
        parameters for model loading. If present, a model will be loaded
        by calling `self.load(**config['load'])`.

    """
    def __init__(self, config=None, *args, **kwargs):
        self.config = Config(config)
        load = self.config.get('load')
        build = self.config.get('build', default=load is None)
        if not isinstance(build, bool) and build in [1, 'first']:
            self.build(*args, **kwargs)
            build = False
        if load:
            self.load(**load)
        if build:
            self.build(*args, **kwargs)

    @property
    def default_name(self):
        """: str - the class name (serve as a default for a model name) """
        return self.__class__.__name__

    @classmethod
    def pop(cls, variables, config, **kwargs):
        """ Return variables and remove them from config"""
        return Config().pop(variables, config, **kwargs)

    @classmethod
    def get(cls, variables, default=None, config=None):
        """ Return variables from config """
        return Config().get(variables, default=default, config=config)

    @classmethod
    def put(cls, variable, value, config):
        """ Put a new variable into config """
        return Config().put(variable, value, config)

    def _make_inputs(self, names=None, config=None):
        """ Make model input data using config

        Parameters
        ----------
        names : a sequence of str - names for input variables

        Returns
        -------
        None or dict - where key is a variable name and a value is a corresponding variable after configuration
        """
        _ = names, config
        return None

    def build(self, *args, **kwargs):
        """ Define the model """
        _ = self, args, kwargs

    def load(self, *args, **kwargs):
        """ Load the model """
        _ = self, args, kwargs

    def save(self, *args, **kwargs):
        """ Save the model """
        _ = self, args, kwargs

    def train(self, *args, **kwargs):
        """ Train the model """
        _ = self, args, kwargs

    def predict(self, *args, **kwargs):
        """ Make a prediction using the model  """
        _ = self, args, kwargs
