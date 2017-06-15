#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   14.06.2017
#
# This essentially is the code from
# http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
# with minor modifications done by me. See test_upscale.py to see how it works.
#-------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np

#-------------------------------------------------------------------------------
def get_bilinear_filter(filter_shape, upscale_factor):
    """
    Creates a weight matrix that performs a bilinear interpolation
    :param filter_shape:   shape of the upscaling filter
    :param upscale_factor: scaling factor
    :return:               weight tensor
    """

    kernel_size = filter_shape[1]

    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            value = (1 - abs((x - centre_location)/upscale_factor)) * \
                    (1 - abs((y - centre_location)/upscale_factor))
            bilinear[x, y] = value

    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="bilinear_filter", initializer=init,
                                       shape=weights.shape)
    return bilinear_weights

#-------------------------------------------------------------------------------
def upsample(x, n_channels, upscale_factor, name):
    """
    Create an upsampling tensor
    :param x:              input tensor
    :param n_channels:     number of channels
    :param upscale_factor: scale factor
    :param name:           name of the tensor
    :return:               upsampling tensor
    """

    kernel_size = 2*upscale_factor - upscale_factor%2
    stride      = upscale_factor
    strides     = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_shape = tf.shape(x)

        h = in_shape[1] * stride
        w = in_shape[2] * stride
        new_shape = [in_shape[0], h, w, n_channels]

        output_shape = tf.stack(new_shape)

        filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

        weights = get_bilinear_filter(filter_shape, upscale_factor)
        deconv = tf.nn.conv2d_transpose(x, weights, output_shape,
                                        strides=strides, padding='SAME')
    return deconv
