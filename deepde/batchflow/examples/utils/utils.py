""" File with some useful functions"""
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import ewma


plt.style.use('seaborn-poster')
plt.style.use('ggplot')

def draw(first, first_label, second=None, second_label=None, type_data='loss', window=5, bound=None, axis=None):
    """ Draw on graph first and second data.

    The graph shows a comparison of the average values calculated with a 'window'. You can draw one graph
    or create your oun subplots and one of it in 'axis'.

    Parameters
    ----------
    first : list or numpy array
        Have a values to show

    first_label : str
        Name of first data

    second : list or numpy array, optional
        Have a values to show

    second_label : str, optional
        Name of second data

    type_data : str, optional
        Type of data. Example 'loss', 'accuracy'

    window : int, optional
        window width for calculate average value

    bound : list or None
        Bounds to limit graph: [min x, max x, min y, max y]

    axis : None or element of subplot
        If you want to draw more subplots give the element of subplot """

    firt_ewma = ewma(np.array(first), span=window, adjust=False)
    second_ewma = ewma(np.array(second), span=window, adjust=False) if second else None

    plot = axis or matplotlib.pyplot
    plot.plot(firt_ewma, label='{} {}'.format(first_label, type_data))
    if second_label:
        plot.plot(second_ewma, label='{} {}'.format(second_label, type_data))

    if axis is None:
        plot.xlabel('Iteration', fontsize=16)
        plot.ylabel(type_data, fontsize=16)
    else:
        plot.set_xlabel('Iteration', fontsize=16)
        plot.set_ylabel(type_data, fontsize=16)

    plot.legend(fontsize=14)
    if bound:
        plot.axis(bound)

def get_weights(session):
    """ Get weigths from model

    Parameters
    ----------
    graph : tf.Graph
        your model graph

    session : tf.Session

    Returns
    -------
    names : list with str
        names of all layers

    weights : np.array
        weights of all layers

    biases : np.array
        biases of all layers
    """
    weights, biases = [], []
    graph = session.graph
    variables = graph.get_collection('trainable_variables')
    variables = [var for var in variables if 'batch_normalization' not in var.name][2:]
    names = np.array([layer.name.split('/')[5] if 'shortcut' not in layer.name else 'shortcut'\
                      for layer in variables[:-2:2]])

    weights_variables = [var for var in variables[:-2] if 'bias:0' not in var.name]
    biases_variables = [var for var in variables[:-2] if 'kernel:0' not in var.name]
    weights.append(session.run(weights_variables))
    biases.append(session.run(biases_variables))

    return names, np.array(weights[0]), np.array(biases[0])


def separate(layers_names, weights, num_params, bottle, num_blocks): # pylint: disable=too-many-locals
    """Support fuction that allows yield the data about layer.

    Parameters
    ----------
    layers_names : list of str
        names of layers

    weights : list of str
        weights of layers

    num_params : list or tuple
        number of parameters in each layer

    bottle : bool
        use bottleneck

    num_blocks : list
        numbers of blocks to draw

    Yields
    ------
    layer_names : str
         name of layer

    layer_weights : list
        weights of layer

    layer_params : list
        number of parameters in layer
    """
    blocks = np.where(layers_names == 'layer-1')[0]
    main_name = ['layer-1', 'layer-4']
    len_block = 4 if bottle else 3
    for num in num_blocks:
        data = None
        names = main_name.copy()
        if bottle:
            names.append('layer-7')

        div = blocks[num+1] - blocks[num] if len(blocks) < num+1 else blocks[num] - blocks[num-1]
        names.append('shortcut' if div == len_block else 'zeros')
        for name in names:
            indices = np.where(layers_names == name)[0]
            if name == 'shortcut':
                indices = blocks[num+1] - 1
                zipp = np.array(['shortcut', weights[indices], num_params[indices], None])[:-1]
            elif name == 'zeros':
                zipp = np.array([0, 0, 0])
            else:
                zipp = np.array([layers_names[indices][num], weights[indices][num],
                                 num_params[indices][num], None])[:-1]

            if data is not None:
                data = np.hstack((data, zipp))
            else:
                data = zipp
        layer_names, layer_weights, layer_params = data[::3], data[1::3], data[2::3]
        yield layer_names, layer_weights, layer_params

def plot_weights(model_names, model_weights, model_params, colors, num_axis, num_blocks, bottleneck=True): # pylint: disable=too-many-locals
    """Plot distribution of weights

    Parameters
    ----------
    model_names : list or str
        name layers of model

    model_weights : list
        all weights of model

    model_params : list
        number of parameters in layers

    colors : list
        names of colors

    num_axis : list with two elements
        [nrows, ncols] in plt.subplots

    bottleneck : bool
        use bottleneck

    num_blocks : list
        numbers of blocks to draw
        """
    nrows, ncols = num_axis
    _, subplot = plt.subplots(nrows, ncols, sharex='all', figsize=(23, 24))
    subplot = subplot.reshape(-1)
    num_plot = 0
    dict_names = {'bottleneck': {'layer-1': 'first conv 1x1',
                                 'layer-4': 'conv 3x3',
                                 'layer-7': 'second conv 1x1'},
                  'no_bottle': {'layer-1': 'first conv 3x3',
                                'layer-4': 'second conv 3x3'}}

    bottle = 'bottleneck' if bottleneck else 'no_bottle'

    for names, weights, num_params in separate(model_names, model_weights, model_params, bottleneck, num_blocks):
        for name, weight, num in zip(names, weights, num_params):

            if name != 'shortcut' and name != 0:
                name = dict_names[bottle][name]

            subplot[num_plot].set_title('Number of parameners={}\n{}'.format(num, name), fontsize=18)

            if not isinstance(weight, int):
                sns.distplot(weight.reshape(-1), ax=subplot[num_plot], color=colors[int(num_plot % ncols)])

                if num_plot % 1 == 0:
                    dis = (6. / ((weight.shape[2] + weight.shape[3]) * weight.shape[0] * weight.shape[1])) ** 0.5
                    subplot[num_plot].axvline(x=dis, ymax=10, color='k')
                    subplot[num_plot].axvline(x=-dis, ymax=10, color='k')

            subplot[num_plot].set_xlabel('value', fontsize=20)
            subplot[num_plot].set_ylabel('quantity', fontsize=20)
            num_plot += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

def draw_avgpooling(maps, answers, axis=None, span=350, model=True):
    """ Draw maps from GAP

    Parameters
    ----------
    maps : np.array
        all maps from GAP layers

    answers : np.array
        answers to all maps

    span : float, optional
        Specify decay in terms of span

    axis : list, optional
        sets the min and max of the x and y axes, with ``[xmin, xmax, ymin, ymax]``

    model : bool, optional
        se resnet or simple resnet
    """
    axis = [0, 2060, 0, 1] if axis is None else axis
    col = sns.color_palette("Set2", 8) + sns.color_palette(["#9b59b6", "#3498db"])

    indices = np.array([np.where(answers == i)[0] for i in range(10)])

    filters = np.array([np.mean(maps[indices[i]], axis=0).reshape(-1) for i in range(10)])
    for i in range(10):
        plt.plot(ewma(filters[i], span=span, adjust=False), color=col[i], label=str(i))

    plt.title("Distribution of average pooling in "+("SE ResNet" if model else 'simple ResNet'))
    plt.legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Activation value', fontsize=18)
    plt.xlabel('Future map index', fontsize=18)
    plt.axis(axis)
    plt.show()

def axis_draw(freeze_loss, res_loss, src, axis):
    """ Draw graphs to compare models. Theaxis graph shows a comparison of the average
        values calculated with a window in 10 values.

    Parameters
    ----------
    freeze_loss : list
        loss value in resnet and freezeout model

    res_loss : list
        loss value in clear resnet

    src : list
        parameters of model with FreezeOut

    axis : plt sublot
    """
    fr_loss = []
    n_loss = []

    for i in range(10, len(res_loss) - 10):
        fr_loss.append(np.mean(freeze_loss[i-10:i+10]))
        n_loss.append(np.mean(res_loss[i-10:i+10]))

    axis.set_title('Freeze model with: LR={} Degree={} It={} Scaled={}'.format(*src))
    axis.plot(fr_loss, label='freeze loss')
    axis.plot(n_loss, label='no freeze loss')
    axis.set_xlabel('Iteration', fontsize=16)
    axis.set_ylabel('Loss', fontsize=16)
    axis.legend(fontsize=14, loc=3)

def four_losses_draw(losses, names, title):
    """ Draw two graphs. First - last 100 iterations. Second - all iterations.

    Parameters
    ----------
    losses : list
        loss values

    names : list
        names of loss

    title : str
        title to graph
    """
    _, axis = plt.subplots(1, 2)
    for loss, name in zip(losses, names):
        axis[0].plot(loss[-100:], label='%s'%name)
        axis[0].plot(ewma(np.array(loss[-100:]), span=10, adjust=False), label='%s'%name)
        axis[1].plot(loss, label='%s'%name)
        axis[1].plot(ewma(np.array(loss), span=10, adjust=False), label='%s'%name)

    axis[0].set_title(title)
    axis[0].legend()
    axis[1].legend()
    plt.show()

def calculate_accuracy(batch, pipeline, predict_name):
    """ Calculate top1 and top3 accuracy

    Parameters
    ----------
    batch : batch class
        model batch

    pipeline : pipeline class
        pipeline with prob variable to calculate accuracy

    predict_name : str
        name of pipeline variable

    Returns
    -------
        top one and top three accuracy"""
    predict = pipeline.get_variable(predict_name)
    predict_top3 = predict[-1].argsort()[:, -3::]

    top1 = np.mean(np.argmax(predict[-1], axis=1) == batch.labels)
    top3 = np.mean([1 if batch.labels[i] in pred else 0 for i, pred in enumerate(predict_top3)])
    return top1, top3
