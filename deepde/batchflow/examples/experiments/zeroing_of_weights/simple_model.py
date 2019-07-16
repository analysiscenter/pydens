""" File with simple model example of convolution network """
from batchflow.models.tf import TFModel


class ConvModel(TFModel):
    """ Class to build conv model """

    @classmethod
    def default_config(cls):
        """ Default parameters for conv model.

         Returns
        -------
        config : dict
            default parameters to network
        """
        config = TFModel.default_config()
        config['body'].update(dict(layout='cpacpa', filters=[16, 32], kernel_size=[7, 5],
                                   strides=[2, 1], pool_size=[4, 3], pool_strides=2))
        config['head'].update(dict(layout='faf'))
        return config
