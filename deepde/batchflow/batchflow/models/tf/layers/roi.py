#pylint:disable=cell-var-from-loop

""" Layers for proposal regions. """

import tensorflow as tf

def roi_pooling_layer(inputs, rois, factor=(1, 1), shape=(7, 7), data_format='channels_last', name='roi-pooling'):
    """ ROI pooling layer with resize instead max-pool.

    Parameters
    ----------
        inputs: tf.Tensor
            input tensor
        rois: tf.Tuple
            coordinates of bboxes for each image
        factor: tuple
            factor to transform coordinates of bboxes
        shape: tuple
            resize to
        data_format: str, 'channels_last' or 'channels_first'
        name: str
            scope name

    Return
    ------
        tf.Tuple
            cropped regions
    """
    with tf.variable_scope(name):
        image_index = tf.constant(0)
        output_tuple = tf.TensorArray(dtype=tf.float32, size=len(rois))

        for image_index, image_rois in enumerate(rois):
            image = inputs[image_index]
            if data_format == 'channels_first':
                image = tf.transpose(image, [1, 2, 0])
            cropped_regions = tf.TensorArray(dtype=tf.float32, size=tf.shape(image_rois)[0])
            roi_index = tf.constant(0)

            cond_rois = lambda roi_index, cropped_regions: tf.less(roi_index, tf.shape(image_rois)[0])

            def _roi_body(roi_index, cropped_regions):
                with tf.variable_scope('crop-from-image-{}'.format(image_index)):
                    roi = image_rois[roi_index]

                    spatial_start = roi[:2] * factor
                    spatial_size = roi[2:] * factor

                    spatial_start = tf.cast(tf.ceil(spatial_start), dtype=tf.int32)
                    spatial_size = tf.cast(tf.ceil(spatial_size), dtype=tf.int32)

                    spatial_start = tf.maximum(tf.constant((0, 0)), spatial_start)
                    spatial_start = tf.minimum(tf.shape(image)[:2]-2, spatial_start)

                    spatial_size = tf.maximum(tf.constant((1, 1)), spatial_size)
                    spatial_size = tf.minimum(tf.shape(image)[:2]-spatial_start, spatial_size)

                    start = tf.concat([spatial_start, tf.constant((0,))], axis=0)
                    size = tf.concat([spatial_size, (tf.shape(image)[-1], )], axis=0)
                    cropped = tf.slice(image, start, size)
                    cropped = tf.image.resize_images(cropped, shape)
                    cropped.set_shape([*shape, image.get_shape().as_list()[-1]])
                    if data_format == 'channels_first':
                        cropped = tf.transpose(cropped, [2, 0, 1])
                    cropped_regions = cropped_regions.write(roi_index, cropped)
                return [roi_index+1, cropped_regions]

            _, res = tf.while_loop(cond_rois, _roi_body, [roi_index, cropped_regions])
            res = res.stack()
            output_tuple = output_tuple.write(image_index, res)
        res = _array_to_tuple(output_tuple, len(rois))
    return res

def non_max_suppression(inputs, scores, batch_size, max_output_size,
                        score_threshold=0.7, iou_threshold=0.7, nonempty=False, name='nms'):
    """ Perform NMS on batch of images.

    Parameters
    ----------
        inputs: tf.Tuple
            each components is a set of bboxes for corresponding image
        scores: tf.Tuple
            scores of inputs
        batch_size:
            size of batch of inputs
        max_output_size:
            maximal size of bboxes per image
        score_threshold: float
            bboxes with score less the score_threshold will be dropped
        iou_threshold: float
            bboxes with iou which is greater then iou_threshold will be merged
        nonempty: bool
            if True at least one bbox per image will be returned
        name: str
            scope name

    Returns
    -------
        tf.Tuple
            indices of selected bboxes for each image

    """
    with tf.variable_scope(name):
        ix = tf.constant(0)
        filtered_rois = tf.TensorArray(dtype=tf.int32, size=batch_size, infer_shape=False)
        loop_cond = lambda ix, filtered_rois: tf.less(ix, batch_size)
        def _loop_body(ix, filtered_rois):
            indices, score, roi = _filter_tensor(scores[ix], score_threshold, inputs[ix]) # pylint: disable=unbalanced-tuple-unpacking
            roi_corners = tf.concat([roi[:, :2], roi[:, :2]+roi[:, 2:]], axis=-1)
            roi_after_nms = tf.image.non_max_suppression(roi_corners, score, max_output_size, iou_threshold)
            if nonempty:
                is_not_empty = lambda: filtered_rois.write(ix,
                                                           tf.cast(tf.gather(indices, roi_after_nms),
                                                                   dtype=tf.int32))
                is_empty = lambda: filtered_rois.write(ix, tf.constant([[0]]))
                filtered_rois = tf.cond(tf.not_equal(tf.shape(indices)[0], 0), is_not_empty, is_empty)
            else:
                filtered_rois = filtered_rois.write(ix, tf.cast(tf.gather(indices, roi_after_nms), dtype=tf.int32))
            return [ix+1, filtered_rois]
        _, res = tf.while_loop(loop_cond, _loop_body, [ix, filtered_rois])
        res = _array_to_tuple(res, batch_size, [-1, 1])
    return res



def _filter_tensor(inputs, cond, *args):
    """ Create indixes and elements of inputs which consists for which cond is True.

    Parameters
    ----------
        inputs: tf.Tensor
            input tensor
        cond: callable or float
            condition to choose elements. If float, elements which greater the cond will be choosen
        *args: tf.Tensors:
            tensors with the same shape as inputs. Will be returned corresponding elements of them.

    Returns
    -------
        indices: tf.Tensor
            indices of elements of inputs for which cond is True
        tf.Tensors:
            filtred inputs and tensors from args.
    """
    with tf.variable_scope('filter_tensor'):
        if not callable(cond):
            callable_cond = lambda x: x > cond
        else:
            callable_cond = cond
        indices = tf.where(callable_cond(inputs))
        output = (indices, *[tf.gather_nd(x, indices) for x in [inputs, *args]])
    return output

def _array_to_tuple(inputs, size, shape=None):
    """ Convert tf.TensorArray to tf.Tuple. """
    with tf.variable_scope('array_to_tuple'):
        if shape is None:
            output = tf.tuple([inputs.read(i) for i in range(size)])
        else:
            output = tf.tuple([tf.reshape(inputs.read(i), shape) for i in range(size)])
    return output
