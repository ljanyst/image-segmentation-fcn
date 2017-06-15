#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   14.06.2017
#-------------------------------------------------------------------------------

import cv2
import sys

import tensorflow as tf
import numpy as np

from upscale import upsample

if len(sys.argv) != 3:
    print('Make the input image 8 times larger.')
    print('Usage:', sys.argv[0], 'input_image output_image')
    sys.exit(1)

img = cv2.imread(sys.argv[1])

if img is None:
    print('Unable to load input image ' + sys.argv[1])
    sys.exit(1)

print('Original size:', img.shape)

imgs = np.zeros([1, *img.shape], dtype=np.float32)
imgs[0,:,:,:] = img

img_input = tf.placeholder(tf.float32, [None, *img.shape])
upscale = upsample(img_input, 3, 8, 'upscaled')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    upscaled = sess.run(upscale, feed_dict={img_input: imgs})

print('Upscaled:', upscaled.shape[1:])

cv2.imwrite(sys.argv[2], upscaled[0,:, :, :])
