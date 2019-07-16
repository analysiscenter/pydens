"""
Dongyoon Han et al. "`Deep Pyramidal Residual Networks
<https://arxiv.org/abs/1610.02915>`_"

"""
from . import ResNet


class PyramidNet(ResNet):
    """ The base PyramidNet model

    Notes
    -----
    This class is intended to define custom PyramidNets.
    For more convenience use predefined :class:`.tf.PyramidNet18`, :class:`.tf.PyramidNet34`,
    and others described down below.

    **Configuration**

    inputs : dict
        dict with 'images' and 'labels' (see :meth:`~.TFModel._make_inputs`)

    initial_block : dict
        parameters for the initial block (see :func:`.conv_block`).

    body : dict
        num_blocks : list of int
            number of blocks in each group with the same number of filters.

        block : dict
            widening : int
                an increment of filters number in each block (default=8)

            and other :class:`~.tf.ResNet` block params

    head : dict
        'Vdf' with dropout_rate=.4

    Notes
    -----
    Also see :class:`~.TFModel` and :class:`~.tf.ResNet` configuration.
    """
    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        config['body/block/widening'] = 8
        config['body/block/zero_pad'] = True
        return config

    @classmethod
    def default_layout(cls, bottleneck, **kwargs):
        return 'nc nac nac n' if bottleneck else 'nc nac n'

    def build_config(self, names=None):
        config = super(ResNet, self).build_config(names)

        if config.get('body/filters') is None:
            w = config['body/block/widening']
            filters = config['initial_block/filters']
            config['body/filters'] = []
            for g in config['body/num_blocks']:
                bfilters = [filters +  w * b for b in range(1, g + 1)]
                filters = bfilters[-1]
                config['body/filters'].append(bfilters)

        if config.get('head/units') is None:
            config['head/units'] = self.num_classes('targets')
        if config.get('head/filters') is None:
            config['head/filters'] = self.num_classes('targets')

        return config

class PyramidNet18(PyramidNet):
    """ 18-layer PyramidNet architecture """
    @classmethod
    def default_config(cls):
        config = PyramidNet.default_config()
        config['body/num_blocks'] = [2, 2, 2, 2]
        config['body/block/bottleneck'] = False
        return config


class PyramidNet34(PyramidNet):
    """ 34-layer PyramidNet architecture """
    @classmethod
    def default_config(cls):
        config = PyramidNet.default_config()
        config['body/num_blocks'] = [3, 4, 6, 3]
        config['body/block/bottleneck'] = False
        return config


class PyramidNet50(PyramidNet):
    """ 50-layer PyramidNet architecture with bottleneck blocks """
    @classmethod
    def default_config(cls):
        config = PyramidNet.default_config()
        config['body/block/bottleneck'] = True
        return config


class PyramidNet101(PyramidNet):
    """ 101-layer PyramidNet architecture with bottleneck blocks """
    @classmethod
    def default_config(cls):
        config = PyramidNet.default_config()
        config['body/num_blocks'] = [3, 4, 23, 3]
        config['body/block/bottleneck'] = True
        return config


class PyramidNet152(PyramidNet):
    """ 152-layer PyramidNet architecture with bottleneck blocks """
    @classmethod
    def default_config(cls):
        config = PyramidNet.default_config()
        config['body/num_blocks'] = [3, 8, 36, 3]
        config['body/block/bottleneck'] = True
        return config
