#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   14.06.2017
#-------------------------------------------------------------------------------

import zipfile
import shutil
import os

import tensorflow as tf

from urllib.request import urlretrieve
from upscale import upsample
from tqdm import tqdm

#-------------------------------------------------------------------------------
class DLProgress(tqdm):
    last_block = 0

    #---------------------------------------------------------------------------
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

#-------------------------------------------------------------------------------
def reshape(x, num_classes, upscale_factor, name):
    """
    Reshape the tensor so that it matches the number of classes and output size
    :param x:              input tensor
    :param num_classes:    number of classes
    :param upscale_factor: scaling factor
    :param name:           name of the resulting tensor
    :return:               reshaped tensor
    """
    with tf.variable_scope(name):
        w_shape = [1, 1, int(x.get_shape()[3]), num_classes]
        w = tf.Variable(tf.truncated_normal(w_shape, 0, 0.1),
                        name=name+'_weights')
        b = tf.Variable(tf.zeros(num_classes), name=name+'_bias')
        resized = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID',
                               name=name+'_resized')
        resized = tf.nn.bias_add(resized, b, name=name+'_add_bias')

    upsampled = upsample(resized, num_classes, upscale_factor,
                         name+'_upsampled')
    return upsampled

#-------------------------------------------------------------------------------
class FCNVGG:
    #---------------------------------------------------------------------------
    def __init__(self, session):
        self.session     = session

    #---------------------------------------------------------------------------
    def build_from_vgg(self, vgg_dir, num_classes, progress_hook):
        """
        Build the model for training based on a pre-define vgg16 model.
        :param vgg_dir:       directory where the vgg model should be stored
        :param num_classes:   number of classes
        :param progress_hook: a hook to show download progress of vgg16;
                              the value may be a callable for urlretrieve
                              or string "tqdm"
        """
        self.num_classes = num_classes
        self.__download_vgg(vgg_dir, progress_hook)
        self.__load_vgg(vgg_dir)
        self.__make_result_tensors()

    #---------------------------------------------------------------------------
    def build_from_metagraph(self, metagraph_file, checkpoint_file):
        """
        Build the model for inference from a metagraph shapshot and weights
        checkpoint.
        """
        sess = self.session
        saver = tf.train.import_meta_graph(metagraph_file)
        saver.restore(sess, checkpoint_file)
        self.image_input = sess.graph.get_tensor_by_name('image_input:0')
        self.keep_prob   = sess.graph.get_tensor_by_name('keep_prob:0')
        self.logits      = sess.graph.get_tensor_by_name('sum/Add_1:0')
        self.softmax     = sess.graph.get_tensor_by_name('result/Softmax:0')
        self.classes     = sess.graph.get_tensor_by_name('result/ArgMax:0')

    #---------------------------------------------------------------------------
    def __download_vgg(self, vgg_dir, progress_hook):
        #-----------------------------------------------------------------------
        # Check if the model needs to be downloaded
        #-----------------------------------------------------------------------
        vgg_archive = 'vgg.zip'
        vgg_files   = [
            vgg_dir + '/variables/variables.data-00000-of-00001',
            vgg_dir + '/variables/variables.index',
            vgg_dir + '/saved_model.pb']

        missing_vgg_files = [vgg_file for vgg_file in vgg_files \
                             if not os.path.exists(vgg_file)]

        if missing_vgg_files:
            if os.path.exists(vgg_dir):
                shutil.rmtree(vgg_dir)
            os.makedirs(vgg_dir)

            #-------------------------------------------------------------------
            # Download vgg
            #-------------------------------------------------------------------
            url = 'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip'
            if not os.path.exists(vgg_archive):
                if callable(progress_hook):
                    urlretrieve(url, vgg_archive, progress_hook)
                else:
                    with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
                        urlretrieve(url, vgg_archive, pbar.hook)

            #-------------------------------------------------------------------
            # Extract vgg
            #-------------------------------------------------------------------
            zip_archive = zipfile.ZipFile(vgg_archive, 'r')
            zip_archive.extractall(vgg_dir)
            zip_archive.close()

    #---------------------------------------------------------------------------
    def __load_vgg(self, vgg_dir):
        sess = self.session
        graph = tf.saved_model.loader.load(sess, ['vgg16'], vgg_dir+'/vgg')
        self.image_input = sess.graph.get_tensor_by_name('image_input:0')
        self.keep_prob   = sess.graph.get_tensor_by_name('keep_prob:0')
        self.vgg_layer3  = sess.graph.get_tensor_by_name('layer3_out:0')
        self.vgg_layer4  = sess.graph.get_tensor_by_name('layer4_out:0')
        self.vgg_layer7  = sess.graph.get_tensor_by_name('layer7_out:0')

    #---------------------------------------------------------------------------
    def __make_result_tensors(self):
        vgg3_reshaped = reshape(self.vgg_layer3, self.num_classes,  8,
                                'layer3_resize')
        vgg4_reshaped = reshape(self.vgg_layer4, self.num_classes, 16,
                                'layer4_resize')
        vgg7_reshaped = reshape(self.vgg_layer7, self.num_classes, 32,
                                'layer7_resize')

        with tf.variable_scope('sum'):
            self.logits   = tf.add(vgg3_reshaped,
                                   tf.add(2*vgg4_reshaped, 4*vgg7_reshaped))
        with tf.name_scope('result'):
            self.softmax  = tf.nn.softmax(self.logits)
            self.classes  = tf.argmax(self.softmax, axis=3)

    #---------------------------------------------------------------------------
    def get_optimizer(self, labels, learning_rate=0.0001):
        with tf.variable_scope('reshape'):
            labels_reshaped  = tf.reshape(labels, [-1, self.num_classes])
            logits_reshaped  = tf.reshape(self.logits, [-1, self.num_classes])
            losses          = tf.nn.softmax_cross_entropy_with_logits(
                                  labels=labels_reshaped,
                                  logits=logits_reshaped)
            loss            = tf.reduce_mean(losses)
        with tf.variable_scope('optimizer'):
            optimizer       = tf.train.AdamOptimizer(learning_rate)
            optimizer       = optimizer.minimize(loss)

        return optimizer, loss
