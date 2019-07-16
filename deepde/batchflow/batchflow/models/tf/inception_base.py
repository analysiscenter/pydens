""" Parent class to inception models """
import tensorflow as tf

from . import TFModel
from .layers import conv_block


class Inception(TFModel):
    """ The base class for all inception models

    **Configuration**

    body : dict
        layout : str
            a sequence of blocks in the network:

            - b - inception block for v1 and v3 models
            - r - reduction block
            - f - factorization_block for Inception_v3 model (see :meth:`.factorization_block`)
            - m - mixed_block for Inception_v3 model (see :meth:`.mixed_block`)
            - e - expanded_block for Inception_v3 model (see :meth:`.expanded_block`)
            - A - inception block A for Inception_v4 model (see :meth:`.inception_a_block`)
            - B - inception block B for Inception_v4 model (see :meth:`.inception_b_block`)
            - C - Inception block C for Inception_v4 model (see :meth:`.inception_c_block`)
            - G - grid-reduction block for Inception_v4 model (see :meth:`.reduction_grid_block`)

        arch : dict
            parameters for each block:

            key : str
                block's short name
            value : dict
                specific parameters (e.g. filters)
    """

    def build_config(self, names=None):
        config = super().build_config(names)
        if config.get('head/units') is None:
            config['head/units'] = self.num_classes('targets')
        if config.get('head/filters') is None:
            config['head/filters'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        layout : str
            a sequence of blocks
        arch : dict
            parameters for each block
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        arch, layout = cls.pop(['arch', 'layout'], kwargs)

        with tf.variable_scope(name):
            x, inputs = inputs, None

            layout_dict = {}
            for block in layout:
                if block not in layout_dict:
                    layout_dict[block] = [-1, 0]
                layout_dict[block][1] += 1

            for i, block in enumerate(layout):
                layout_dict[block][0] += 1
                block_no = layout_dict[block][0]

                block_args = {**kwargs, **arch[block]}
                filters = block_args.pop('filters', None)
                if isinstance(filters, list):
                    filters = filters[block_no]

                if block == 'b':
                    x = cls.block(x, filters=filters, name='block-%d'%i, **block_args)
                elif block == 'r':
                    x = cls.reduction_block(x, filters=filters, name='reduction_block-%d'%i, **block_args)
                elif block == 'f':
                    x = cls.factorization_block(x, filters=filters, name='factorization_block-%d'%i, **block_args)
                elif block == 'm':
                    x = cls.mixed_block(x, filters=filters, name='mixed_block-%d'%i, **block_args)
                elif block == 'e':
                    x = cls.expanded_block(x, filters=filters, name='expanded_block-%d'%i, **block_args)
                elif block == 'A':
                    x = cls.inception_a_block(x, filters=filters, name='inception_a_block-%d'%i, **block_args)
                elif block == 'B':
                    x = cls.inception_b_block(x, filters=filters, name='inception_b_block-%d'%i, **block_args)
                elif block == 'C':
                    x = cls.inception_c_block(x, filters=filters, name='inception_c_block-%d'%i, **block_args)
                elif block == 'G':
                    x = cls.reduction_grid_block(x, filters=filters, name='reduction_grid_block-%d'%i, **block_args)

        return x

    @classmethod
    def reduction_block(cls, inputs, filters, layout='cna', name='reduction_block', **kwargs):
        """ Reduction block.

        For details see figure 10 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 3 ints
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            branch_3 = conv_block(inputs, layout, filters[3], 3, name='conv_3', strides=2, padding='valid', **kwargs)
            branch_1_3 = conv_block(inputs, layout*2, [filters[0]]+[filters[1]], [1, 3], name='conv_1_3', **kwargs)
            branch_1_3_3 = conv_block(branch_1_3, layout, filters[2], 3, name='conv_1_3_3', strides=2,
                                      padding='valid', **kwargs)

            branch_pool = conv_block(inputs, layout='p', pool_size=3, pool_strides=2, name='max_pooling',
                                     padding='valid', **kwargs)

            axis = cls.channels_axis(kwargs['data_format'])
            output = tf.concat([branch_3, branch_1_3_3, branch_pool], axis, name='output')
        return output
