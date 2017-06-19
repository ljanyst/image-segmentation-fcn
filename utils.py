#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   14.06.2017
#-------------------------------------------------------------------------------

import cv2

import tensorflow as tf
import numpy as np

#-------------------------------------------------------------------------------
def draw_labels(img, labels, label_colors, convert=True):
    """
    Draw the labels on top of the input image
    :param img:          the image being classified
    :param labels:       the output of the neural network
    :param label_colors: the label color map defined in the source
    :param convert:      should the output be converted to RGB
    """
    labels_colored = np.zeros_like(img)
    for label in label_colors:
        label_mask = labels == label
        labels_colored[label_mask] = label_colors[label]
    img = cv2.addWeighted(img, 1, labels_colored, 0.8, 0)
    if not convert:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#-------------------------------------------------------------------------------
def draw_labels_batch(imgs, labels, label_colors, convert=True):
    """
    Perform `draw_labels` on all the images in the batch
    """
    imgs_labelled = np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
        imgs_labelled[i, :, :, :] = draw_labels(imgs[i,:, :, :],
                                                labels[i, :, :],
                                                label_colors,
                                                convert)
    return imgs_labelled

#-------------------------------------------------------------------------------
def initialize_uninitialized_variables(sess):
    """
    Only initialize the weights that have not yet been initialized by other
    means, such as importing a metagraph and a checkpoint. It's useful when
    extending an existing model.
    """
    uninit_vars    = []
    uninit_tensors = []
    for var in tf.global_variables():
        uninit_vars.append(var)
        uninit_tensors.append(tf.is_variable_initialized(var))
    uninit_bools = sess.run(uninit_tensors)
    uninit = zip(uninit_bools, uninit_vars)
    uninit = [var for init, var in uninit if not init]
    sess.run(tf.variables_initializer(uninit))

#-------------------------------------------------------------------------------
def load_data_source(data_source):
    """
    Load a data source given it's name
    """
    source_module = __import__('source_'+data_source)
    get_source    = getattr(source_module, 'get_source')
    return get_source()
