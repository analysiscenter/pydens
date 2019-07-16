#pylint:disable=cell-var-from-loop
#pylint:disable=attribute-defined-outside-init

"""
Ren S. et al "`Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
<https://arxiv.org/abs/1506.01497>`_"
"""
import tensorflow as tf
import numpy as np

from . import TFModel, VGG7
from .layers import conv_block, non_max_suppression, roi_pooling_layer


class FasterRCNN(TFModel):
    """ Faster Region-based convolutional neural network.
    WARNING: FasterRCNN works with TensorFlow 1.3.0 and with fixed batch size. """

    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        config['output']['prefix'] = ['reg', 'cls']
        config['anchors_batch'] = 64
        config['rcn_batch'] = 64
        config['nms_threshold'] = 0.2
        config['rpn_thresholds'] = (0.3, 0.7)
        config['initial_block']['base_network'] = VGG7

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['image_shape'] = self.get_spatial_shape('images')
        config['head']['batch_size'] = config['batch_size']
        config['head']['nms_threshold'] = config['nms_threshold']
        config['head']['rpn_thresholds'] = config['rpn_thresholds']
        config['head']['rcn_batch'] = config['rcn_batch']

        self.anchors_batch = config['anchors_batch']
        self.rpn_thresholds = config['rpn_thresholds']
        return config

    def initial_block(self, inputs, base_network, name='initial_block', **kwargs):
        train_mode = tf.placeholder(tf.bool, shape=(), name='train_mode')
        self.store_to_attr('train_mode', train_mode)

        return base_network.body(inputs, name=name, **kwargs)

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        return conv_block(inputs, 'ca', filters=512, kernel_size=3, name=name, **kwargs)

    def create_anchors_tensors(self):
        """ Create placheholders for anchor-based inputs. """
        n_anchors = self.n_anchors
        with tf.variable_scope('anchors'):
            tensors = {'batch': tf.placeholder(tf.float32, shape=[None, n_anchors]),
                       'clsf': tf.placeholder(tf.float32, shape=[None, n_anchors]),
                       'labels': tf.placeholder(tf.int32, shape=[None, n_anchors]),
                       'reg': tf.placeholder(tf.float32, shape=[None, n_anchors, 4])
                      }
            self.store_to_attr('anchors', self.create_anchors())
            tensors['anchors'] = tf.constant(self.anchors, dtype=tf.float32)
        self.store_to_attr('anchors_placeholders', tensors)
        return tensors


    def head(self, inputs, image_shape, name='head', **kwargs):
        _ = name

        if kwargs['data_format'] == 'channels_last':
            self.map_shape = np.array(inputs.get_shape().as_list()[1:3])
        else:
            self.map_shape = np.array(inputs.get_shape().as_list()[2:4])
        self.n_anchors = self.map_shape[0] * self.map_shape[1] * 9

        self.image_shape = image_shape
        self.create_anchors_tensors()

        rpn_reg, rpn_clsf, loss1 = self._rpn_head(inputs, **kwargs)
        _, loss2 = self._rcn_head([inputs, rpn_reg, rpn_clsf], image_shape, **kwargs)

        loss = tf.cond(self.train_mode, lambda: loss1, lambda: loss2)

        tf.losses.add_loss(loss)

        return rpn_reg, rpn_clsf


    def _rpn_head(self, inputs, name='rpn_head', **kwargs):
        n_anchors = self.n_anchors
        anchors = self.anchors_placeholders['anchors']
        anchor_reg = self.anchors_placeholders['reg']
        anchor_clsf = self.anchors_placeholders['clsf']
        anchor_batch = self.anchors_placeholders['batch']

        with tf.variable_scope(name):

            rpn_reg = conv_block(inputs, 'c', filters=4*9, kernel_size=1, name='conv_reg', **kwargs)
            rpn_clsf = conv_block(inputs, 'c', filters=1*9, kernel_size=1, name='conv_clsf', **kwargs)

            if kwargs['data_format'] == 'channels_first':
                rpn_reg = tf.transpose(rpn_reg, [0, 2, 3, 1])
                rpn_clsf = tf.transpose(rpn_clsf, [0, 2, 3, 1])


            rpn_reg = tf.reshape(rpn_reg, [-1, n_anchors, 4])
            rpn_clsf = tf.reshape(rpn_clsf, [-1, n_anchors])

            anchor_reg_param = self.parametrize(anchor_reg, anchors)

            loss = self.rpn_loss(rpn_reg, rpn_clsf, anchor_reg_param, anchor_clsf, anchor_batch)
            loss = tf.identity(loss, 'loss')

            rpn_reg = tf.identity(self.unparametrize(rpn_reg, anchors), 'reg')
            rpn_clsf = tf.sigmoid(rpn_clsf, 'clsf')

        return rpn_reg, rpn_clsf, loss

    def _rcn_head(self, inputs, image_shape, nms_threshold, rpn_thresholds,
                  rcn_batch, batch_size, name='rcn_head', **kwargs):
        anchors_labels = self.anchors_placeholders['labels']
        feature_maps, rpn_reg, rpn_cls = inputs
        n_anchors = self.n_anchors

        with tf.variable_scope(name):
            rcn_input_indices = non_max_suppression(rpn_reg, rpn_cls, batch_size, n_anchors,
                                                    iou_threshold=nms_threshold,
                                                    score_threshold=rpn_thresholds[1],
                                                    nonempty=True)

            rcn_input_indices = tf.cond(self.is_training,
                                        lambda: self.create_bbox_batch(rcn_input_indices, rcn_batch),
                                        lambda: rcn_input_indices)

            rcn_input_rois, rcn_input_labels = self._get_rois_and_labels(rpn_reg, anchors_labels, rcn_input_indices)

            for tensor in rcn_input_rois:
                tf.add_to_collection('roi', tensor)
            for tensor in rcn_input_labels:
                tf.add_to_collection('targets', tensor)
            roi_factor = np.array(self.map_shape/image_shape)

            rcn_input_rois = self.stop_gradient_tuple(rcn_input_rois)
            rcn_input_labels = self.stop_gradient_tuple(rcn_input_labels)

            roi_cropped = roi_pooling_layer(feature_maps, rcn_input_rois,
                                            factor=roi_factor, shape=(7, 7), data_format=kwargs['data_format'])
            indices, roi_cropped, rcn_input_labels = self._stack_tuple(roi_cropped, rcn_input_labels) # pylint: disable=unbalanced-tuple-unpacking
            rcn_clsf = conv_block(roi_cropped, 'f', units=10, name='output_conv', **kwargs)

            loss = self.rcn_loss(rcn_clsf, rcn_input_labels)

            rcn_clsf = tf.argmax(rcn_clsf, axis=-1)
            rcn_clsf = self._unstack_tuple(rcn_clsf, indices)
            rcn_clsf = tf.tuple(rcn_clsf, name='clsf')
            for tensor in rcn_clsf:
                tf.add_to_collection('rcn_output', tensor)
            loss = tf.identity(loss, 'loss')

        return rcn_clsf, loss

    def _fill_feed_dict(self, feed_dict=None, is_training=True):

        anchors = dict()
        _feed_dict = dict()
        if 'bboxes' in feed_dict:
            bboxes = feed_dict.pop('bboxes')
            labels = feed_dict.pop('labels')
            thresholds = self.rpn_thresholds

            anchors['reg'], anchors['clsf'], anchors['labels'] = self.create_rpn_inputs(self.anchors,
                                                                                        bboxes,
                                                                                        labels,
                                                                                        thresholds)

            anchors['batch'] = self.create_anchors_batch(anchors['clsf'], self.anchors_batch)
            anchors['clsf'] = np.array(anchors['clsf'] == 1, dtype=np.int32)
            _feed_dict = {self.anchors_placeholders[k]: anchors[k] for k in anchors}

        feed_dict = super()._fill_feed_dict(feed_dict, is_training)
        feed_dict = {**feed_dict, **_feed_dict}

        return feed_dict

    def stop_gradient_tuple(self, inputs):
        """ Stop gradients through tf.tuple. """
        for i, _ in enumerate(inputs):
            inputs[i] = tf.stop_gradient(inputs[i])
        return inputs

    def create_anchors(self, scales=(4, 8, 16), ratio=2):
        """ Create anchors for image_shape depending on output_map_shape. """
        image_shape = self.image_shape
        map_shape = self.map_shape
        ratios = ((np.sqrt(ratio), 1/np.sqrt(ratio)),
                  (1, 1),
                  (1/np.sqrt(ratio), np.sqrt(ratio)))

        anchors = []
        for scale in scales:
            for current_ratio in ratios:
                image_height, image_width = image_shape
                map_height, map_width = map_shape
                n = map_height * map_width

                j = np.array(list(range(map_height)))
                j = np.expand_dims(j, 1)
                j = np.tile(j, (1, map_width))
                j = j.reshape((-1))

                i = np.array(list(range(map_width)))
                i = np.expand_dims(i, 0)
                i = np.tile(i, (map_height, 1))
                i = i.reshape((-1))

                s = np.ones((n)) * scale
                ratio0 = np.ones((n)) * current_ratio[0]
                ratio1 = np.ones((n)) * current_ratio[1]

                height = s * ratio0
                width = s * ratio1
                y = (j + 0.5) * image_height / map_height - height * 0.5
                x = (i + 0.5) * image_width / map_width - width * 0.5

                y, x = [np.maximum(vector, np.zeros((n))) for vector in [y, x]]
                height = np.minimum(height, image_height-y)
                width = np.minimum(width, image_width-x)

                cur_anchors = [np.expand_dims(vector, 1) for vector in [y, x, height, width]]
                cur_anchors = np.concatenate(cur_anchors, axis=1)
                anchors.append(np.array(cur_anchors, np.int32))
        anchors = np.array(anchors)
        anchors = anchors.transpose(1, 0, 2).reshape(-1, 4)
        return anchors

    @classmethod
    def create_rpn_inputs(cls, anchors, bboxes, labels, thresholds):
        """ Create reg and clsf targets of RPN. """
        anchor_reg = []
        anchor_clsf = []
        anchor_labels = []
        for ind, image_bboxes in enumerate(bboxes): # TODO: for -> np
            image_labels = labels[ind]

            n = anchors.shape[0]
            k = image_bboxes.shape[0]

            # Compute the IoUs of the anchors and ground truth boxes
            tiled_anchors = np.tile(np.expand_dims(anchors, 1), (1, k, 1))
            tiled_bboxes = np.tile(np.expand_dims(image_bboxes, 0), (n, 1, 1))

            tiled_anchors = tiled_anchors.reshape((-1, 4))
            tiled_bboxes = tiled_bboxes.reshape((-1, 4))

            ious = cls.iou_bbox(tiled_anchors, tiled_bboxes)[0]
            ious = ious.reshape(n, k)

            # Label each anchor based on its max IoU
            max_ious = np.max(ious, axis=1)
            best_bbox_for_anchor = np.argmax(ious, axis=1)

            _anchor_reg = image_bboxes[best_bbox_for_anchor]
            _anchor_labels = image_labels[best_bbox_for_anchor].reshape(-1)

            # anchor has at least one gt-bbox with IoU > thresholds[1]
            image_clsf = np.array(max_ious > thresholds[1], dtype=np.int32)

            # anchor intersects with at least one bbox
            best_anchor_for_bbox = np.argmax(ious, axis=0)

            image_clsf[best_anchor_for_bbox] = 1
            _anchor_reg[best_anchor_for_bbox] = image_bboxes
            _anchor_labels[best_anchor_for_bbox] = image_labels

            # max IoU for anchor < thresholds[0]
            image_clsf[np.logical_and(max_ious < thresholds[0], image_clsf == 0)] = -1
            anchor_reg.append(_anchor_reg)
            anchor_labels.append(_anchor_labels)
            anchor_clsf.append(image_clsf)
        return np.array(anchor_reg), np.array(anchor_clsf), np.array(anchor_labels)

    @classmethod
    def iou_bbox(cls, bboxes1, bboxes2):
        """ Compute the IoUs between bounding boxes. """
        bboxes1 = np.array(bboxes1, np.float32)
        bboxes2 = np.array(bboxes2, np.float32)

        intersection_min_y = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
        intersection_max_y = np.minimum(bboxes1[:, 0] + bboxes1[:, 2] - 1, bboxes2[:, 0] + bboxes2[:, 2] - 1)
        intersection_height = np.maximum(intersection_max_y - intersection_min_y + 1, np.zeros_like(bboxes1[:, 0]))

        intersection_min_x = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
        intersection_max_x = np.minimum(bboxes1[:, 1] + bboxes1[:, 3] - 1, bboxes2[:, 1] + bboxes2[:, 3] - 1)
        intersection_width = np.maximum(intersection_max_x - intersection_min_x + 1, np.zeros_like(bboxes1[:, 1]))

        area_intersection = intersection_height * intersection_width
        area_first = bboxes1[:, 2] * bboxes1[:, 3]
        area_second = bboxes2[:, 2] * bboxes2[:, 3]
        area_union = area_first + area_second - area_intersection

        iou = area_intersection * 1.0 / area_union
        iof = area_intersection * 1.0 / area_first
        ios = area_intersection * 1.0 / area_second

        return iou, iof, ios

    @classmethod
    def create_anchors_batch(cls, anchor_clsf, batch_size=64):
        """ Create batch indices for anchors. """
        anchor_batch = []
        for clsf in anchor_clsf:
            batch_size = min(batch_size, len(clsf))
            positive = clsf == 1
            negative = clsf == -1
            if sum(positive) + sum(negative) < batch_size:
                batch_size = sum(positive) + sum(negative)
            if sum(positive) < batch_size / 2:
                positive_batch_size = sum(positive)
                negative_batch_size = batch_size - sum(positive)
            elif sum(negative) < batch_size / 2:
                positive_batch_size = batch_size - sum(negative)
                negative_batch_size = sum(negative)
            else:
                positive_batch_size = batch_size // 2
                negative_batch_size = batch_size // 2

            p = positive / sum(positive)
            positive_batch = np.random.choice(len(clsf), size=positive_batch_size, replace=False, p=p)
            p = negative / sum(negative)
            negative_batch = np.random.choice(len(clsf), size=negative_batch_size, replace=False, p=p)
            image_anchor_batch = np.array([False]*len(clsf))
            image_anchor_batch[positive_batch] = True
            image_anchor_batch[negative_batch] = True
            anchor_batch.append(image_anchor_batch)
        return np.array(anchor_batch)

    @classmethod
    def create_bbox_batch(cls, inputs, batch_size=64):
        """ Create batch indices for bboxes. """
        batch = []
        for indices in inputs:
            indices = tf.random_shuffle(indices)
            start = [0] * 2
            size = [tf.minimum(batch_size, tf.shape(indices)[0]), -1]
            sample = tf.slice(indices, start, size)
            sample.set_shape([None, 1])
            batch.append(sample)
        batch = tf.tuple(batch)
        return batch

    @classmethod
    def rpn_loss(cls, reg, clsf, true_reg, true_cls, anchor_batch):
        """ Mixed MSE+CE Loss for RPN. """
        with tf.variable_scope('rpn_loss'):
            cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_cls, logits=clsf)

            anchor_batch_size_norm = tf.expand_dims(1.0 / tf.reduce_sum(anchor_batch, axis=-1), axis=0)

            cls_loss = tf.matmul(anchor_batch_size_norm, cls_loss * anchor_batch)
            cls_loss = cls_loss / tf.cast(tf.shape(clsf)[0], dtype=tf.float32)
            cls_loss = tf.reduce_sum(cls_loss, name='cls_loss')

            sums = tf.reduce_sum((true_reg - reg) ** 2, axis=-1)

            reg_loss = sums * true_cls * anchor_batch
            reg_loss = tf.reduce_mean(reg_loss, axis=-1)
            reg_loss = tf.reduce_mean(reg_loss, name='reg_loss')

            loss = reg_loss * 100 + cls_loss
        return loss

    @classmethod
    def rcn_loss(cls, clsf, true_cls):
        """ CE loss for RCN. """
        with tf.variable_scope('rcn_loss'):
            true_cls = tf.one_hot(true_cls, clsf.get_shape().as_list()[-1])
            cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=true_cls, logits=clsf)
            loss = tf.reduce_mean(cls_loss)
        return loss

    def parametrize(self, inputs, base):
        """ Parametrize inputs coordinates with respect of base. """
        with tf.variable_scope('parametrize'):
            y = (inputs[:, :, 0] - base[:, 0]) * (1.0 / base[:, 2])
            x = (inputs[:, :, 1] - base[:, 1]) * (1.0 / base[:, 3])
            height = tf.log(inputs[:, :, 2] * (1.0 / base[:, 2]))
            width = tf.log(inputs[:, :, 3] * (1.0 / base[:, 3]))
            output = tf.stack((y, x, height, width), axis=-1)
        return output

    def unparametrize(self, inputs, base):
        """ Unparametrize inputs coordinates with respect of base. """
        with tf.variable_scope('parametrize'):
            y = inputs[:, :, 0] * base[:, 2] + base[:, 0]
            x = inputs[:, :, 1] * base[:, 3] + base[:, 1]
            height = tf.exp(inputs[:, :, 2]) * base[:, 2]
            width = tf.exp(inputs[:, :, 3]) * base[:, 3]
            res = tf.stack((y, x, height, width), axis=-1)
        return res


    def _get_rois_and_labels(self, rois, labels, indices):
        with tf.variable_scope('get_rois_and_labels'):
            output_rois = []
            output_labels = []
            for i, index in enumerate(indices):
                output_rois.append(tf.gather_nd(rois[i], index))
                output_labels.append(tf.gather_nd(labels[i], index))
            output_rois = tf.tuple(output_rois)
            output_labels = tf.tuple(output_labels)
        return output_rois, output_labels


    def _stack_tuple(self, inputs, *args):
        tuple_size = len(inputs)
        tensor_sizes = [tf.shape(inputs[i])[0] for i in range(tuple_size)]
        outputs = [tf.concat(x, axis=0) for x in [inputs, *args]]
        return (tensor_sizes, *outputs)

    def _unstack_tuple(self, inputs, tensor_sizes):
        size = len(tensor_sizes)
        start_position = tf.constant(0)
        output = []
        dim = len(inputs.get_shape().as_list())-1
        for i in range(size):
            output.append(tf.slice(inputs, begin=[start_position, *([0]*dim)], size=[tensor_sizes[i], *([-1]*dim)]))
            start_position = start_position + tensor_sizes[i]
        return tf.tuple(output)
