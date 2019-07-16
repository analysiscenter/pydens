""" File with class batch with resnet network """
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier

from batchflow import action, model, Batch # pylint: disable=no-name-in-module

def conv_block(input_tensor, kernel, filters, name, strides=(2, 2)):
    """ Function to create block of ResNet network which include
    three convolution layers and one skip-connection layer.

    Args:
        input_tensor: input tensorflow layer
        kernel: tuple of kernel size in convolution layer
        filters: list of nums filters in convolution layers
        name: name of block
        strides: typle of strides in convolution layer

    Output:
        x: Block output layer """
    filters1, filters2, filters3 = filters
    x = tf.layers.conv2d(input_tensor, filters1, (1, 1), strides, name='convfir' + name, activation=tf.nn.relu,\
                         kernel_initializer=xavier())

    x = tf.layers.conv2d(x, filters2, kernel, name='convsec' + name, activation=tf.nn.relu, padding='SAME',\
                         kernel_initializer=xavier())

    x = tf.layers.conv2d(x, filters3, (1, 1), name='convthr' + name,\
                         kernel_initializer=xavier())

    shortcut = tf.layers.conv2d(input_tensor, filters3, (1, 1), strides, name='short' + name, \
                         kernel_initializer=xavier())
    x = tf.concat([x, shortcut], axis=1)
    x = tf.nn.relu(x)
    return x

def identity_block(input_tensor, kernel, filters, name):
    """ Function to create block of ResNet network which include
    three convolution layers.

    Args:
        input_tensor: input tensorflow layer.
        kernel: tuple of kernel size in convolution layer.
        filters: list of nums filters in convolution layers.
        name: name of block.

    Output:
        x: Block output layer """
    filters1, filters2, filters3 = filters
    x = tf.layers.conv2d(input_tensor, filters1, (1, 1), name='convfir' + name, activation=tf.nn.relu,\
                         kernel_initializer=xavier())

    x = tf.layers.conv2d(x, filters2, kernel, name='convsec' + name, activation=tf.nn.relu, padding='SAME',\
                         kernel_initializer=xavier())

    x = tf.layers.conv2d(x, filters3, (1, 1), name='convthr' + name,\
                         kernel_initializer=xavier())


    x = tf.concat([x, input_tensor], axis=1)
    x = tf.nn.relu(x)
    return x

def create_train(opt, src, global_step, loss, it, global_it, learn, scaled):
    """ Function for create optimizer to each layer.
        Args:
            src: name of layer which be optimize.
            glogal_step: tenforflow Variable. Need to count train steps.
            loss: loss function.
            it: number of last iteraion for current layer.
            global_it: number of last interation for all layers.
            learn: Basic learning rate for current layer.
            scaled: method of disable layers.
        Output:
            New optimizer. """
    def learning_rate(last, src, global_it, learn, scaled):
        """ Function for create step of changing learning rate.
        Args:
            last: number of last iteration.
            src: mane of layer which be optimize.
            global_it: number of last interation for all layers.
            learn: Basic learning rate for current layer.
            scaled: method of disable layers.
        Output:
            bound: list of bounders - number of iteration, after which learning rate will change.
            values: list of new learnings rates.
            var: name of optimize layers"""

        last = int(last)
        if scaled is True:
            values = [0.5 * learn/last * (1 + np.cos(np.pi * i / last)) for i in range(2, last+1)] + [1e-2]
        else:
            values = [0.5 * learn * (1 + np.cos(np.pi * i / last)) for i in range(2, last+1)] + [1e-2]

        bound = list(np.linspace(0, last, len(range(2, last+1)), dtype=np.int32)) + [global_it]
        var = [i for i in tf.trainable_variables() if src in i.name or 'dense' in i.name]
        return list(np.int32(bound)), list(np.float32(values)), var

    b, val, var = learning_rate(it, src, global_it, learn, scaled)
    learning_rate = tf.train.piecewise_constant(global_step, b, val)

    return opt(learning_rate, 0.9, use_nesterov=True).minimize(loss, global_step, var)

class ResBatch(Batch):
    """ Batch to train models with and without FreezeOut """

    def __init__(self, index, *args, **kwargs):
        """ Init function """
        super().__init__(index, *args, **kwargs)

    @property
    def components(self):
        """ Define componentis. """
        return 'images', 'lables'

    @model(mode='dynamic')
    def freeznet(self, config=None):
        """ Simple implementation of ResNet with FreezeOut method.
        Args:
            config: dict with params:
                -iteartions: Total number iteration for train model.
                -degree: 1 or 3.
                -learning_rate: initial learning rate.
                -scaled: True or False.
        Outputs:
            Method return list with len = 2 and some params:
            [0][0]: indices - Plcaeholder which takes batch indices.
            [0][1]: all_data - Placeholder which takes all images.
            [0][2]; all_lables - Placeholder for lables.
            [0][3]: loss - Value of loss function.
            [0][4]: train - List of train optimizers.
            [0][5]: prob - softmax output, need to prediction.
            [1][0]: accuracy - Current accuracy
            [1][1]: session - tf session """
        iteration = config['iteration']
        learning_rate = config['learning_rate']
        scaled = config['scaled']

        with tf.Graph().as_default():

            indices = tf.placeholder(tf.int32, shape=[None, 1], name='indices')
            all_data = tf.placeholder(tf.float32, shape=[50000, 28, 28], name='all_data')
            input_batch = tf.gather_nd(all_data, indices, name='input_batch')
            input_batch = tf.reshape(input_batch, shape=[-1, 28, 28, 1], name='x_to_tens')

            net = tf.layers.conv2d(input_batch, 32, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                                   kernel_initializer=xavier(), name='1')
            net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), name='max_pool')

            net = conv_block(net, 3, [32, 32, 128], name='2', strides=(1, 1))
            net = identity_block(net, 3, [32, 32, 128], name='3')

            net = conv_block(net, 3, [64, 64, 256], name='4', strides=(1, 1))
            net = identity_block(net, 3, [64, 64, 256], name='5')

            net = tf.layers.average_pooling2d(net, (7, 7), strides=(1, 1))
            net = tf.contrib.layers.flatten(net)

            with tf.variable_scope('dense'):
                net = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense')

            prob = tf.nn.softmax(net, name='soft')
            all_labels = tf.placeholder(tf.float32, [None, 10], name='all_labels')
            y = tf.gather_nd(all_labels, indices, name='y')

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y), name='loss')
            global_steps = []
            train = []

            for i in range(1, 6):
                global_steps.append(tf.Variable(0, trainable=False, name='var_{}'.format(i)))
                train.append(create_train(tf.train.MomentumOptimizer, str(i), \
                                          global_steps[-1], loss, iteration * (i / 10 + 0.5) ** config['degree'], \
                                           iteration, learning_rate, scaled))

            lables_hat = tf.cast(tf.argmax(net, axis=1), tf.float32, name='lables_hat')
            lables = tf.cast(tf.argmax(y, axis=1), tf.float32, name='lables')
            accuracy = tf.reduce_mean(tf.cast(tf.equal(lables_hat, lables), tf.float32, name='accuracy'))

            session = tf.Session()
            session.run(tf.global_variables_initializer())

        return [[indices, all_data, all_labels, loss, train, prob], [accuracy, session]]

    @action(model='freeznet')
    def train_freez(self, models, train_loss, data, lables):
        """ Function for traning ResNet with freezeout method.
        Args:
            sess: tensorflow session.
            train_loss: list with info of train loss.
            train_acc: list with info of train accuracy.

        Output:
            self """
        indices, all_data, all_lables, loss, train, _ = models[0]
        session = models[1][1]

        loss, _ = session.run([loss, train], feed_dict={indices:self.indices.reshape(-1, 1), all_lables:lables, \
            all_data:data})

        train_loss.append(loss)

        return self

    @model(mode='dynamic')
    def resnet(self):
        """ Simple implementation of Resnet.
        Args:
            self

        Outputs:
            Method return list with len = 2 and some params:
            [0][0]: indices - Placeholder which takes batch indices.
            [0][1]: all_data - Placeholder which takes all images.
            [0][2]; all_lables - Placeholder for lables.
            [0][3]: loss - Value of loss function.
            [0][4]: train - List of train optimizers.
            [0][5]: prob - softmax output, need to prediction.

            [1][0]: accuracy - Current accuracy
            [1][1]: session - tf session """
        with tf.Graph().as_default():
            indices = tf.placeholder(tf.int32, shape=[None, 1])
            all_data = tf.placeholder(tf.float32, shape=[50000, 28, 28])
            input_batch = tf.gather_nd(all_data, indices)
            x1_to_tens = tf.reshape(input_batch, shape=[-1, 28, 28, 1])

            net1 = tf.layers.conv2d(x1_to_tens, 32, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                kernel_initializer=xavier(), name='11')
            net1 = tf.layers.max_pooling2d(net1, (2, 2), (2, 2))

            net1 = conv_block(net1, 3, [32, 32, 128], name='22', strides=(1, 1))

            net1 = identity_block(net1, 3, [32, 32, 128], name='33')

            net1 = conv_block(net1, 3, [64, 64, 256], name='53', strides=(1, 1))
            net1 = identity_block(net1, 3, [64, 64, 256], name='63')

            net1 = tf.layers.average_pooling2d(net1, (7, 7), strides=(1, 1))
            net1 = tf.contrib.layers.flatten(net1)

            with tf.variable_scope('dense3'):
                net1 = tf.layers.dense(net1, 10, kernel_initializer=tf.contrib.layers.xavier_initializer())


            prob1 = tf.nn.softmax(net1)
            all_lables = tf.placeholder(tf.float32, [None, 10])

            y = tf.gather_nd(all_lables, indices)

            loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net1, labels=y), name='loss3')
            train1 = tf.train.MomentumOptimizer(0.03, 0.8, use_nesterov=True).minimize(loss1)
            lables_hat1 = tf.cast(tf.argmax(net1, axis=1), tf.float32, name='lables_3at')
            lables1 = tf.cast(tf.argmax(y, axis=1), tf.float32, name='labl3es')

            accuracy1 = tf.reduce_mean(tf.cast(tf.equal(lables_hat1, lables1), tf.float32, name='a3ccuracy'))
            session = tf.Session()
            session.run(tf.global_variables_initializer())
        return [[indices, all_data, all_lables, loss1, train1, prob1], [accuracy1, session]]

    @action(model='resnet')
    def train_res(self, models, train_loss, data, lables):
        """ Function for traning ResNet.
        Args:
            sess: tensorflow session.
            train_loss: list with info of train loss.
            train_acc: list with info of train accuracy.

        Output:
            self """

        session = models[1][1]
        indices, all_data, all_lables, loss, train, _ = models[0]
        loss, _ = session.run([loss, train], feed_dict={indices:self.indices.reshape(-1, 1),\
         all_lables:lables, all_data:data})

        train_loss.append(loss)

        return self
