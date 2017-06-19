#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   15.06.2017
#-------------------------------------------------------------------------------

import math
import sys
import os

import tensorflow as tf

from utils import *
from tqdm import tqdm

#-------------------------------------------------------------------------------
# Check validity of the lavel
#-------------------------------------------------------------------------------
def check_label_validity(y, ny):
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            for k in range(y.shape[2]):
                nnzeros = np.count_nonzero(y[i, j, k, :])
                has_one = 1 in y[i, j, k, :]
                if nnzeros != 1 or not has_one:
                    print('{} label for pixel ({}, {}) is invalid'
                          .format(ny[i], j, k))
                    print(y[i, j, k, :])

#-------------------------------------------------------------------------------
# Check the commandline
#-------------------------------------------------------------------------------
if len(sys.argv) != 4:
    print('Test a data source')
    print('Usage:', sys.argv[0], 'source_name data_dir output_dir')
    sys.exit(1)

source_name = sys.argv[1]
data_dir    = sys.argv[2]
output_dir  = sys.argv[3]

#-------------------------------------------------------------------------------
# Import the source
#-------------------------------------------------------------------------------
try:
    source = load_data_source(source_name)
    source.load_data(data_dir, 0.1)
    train_generator = source.train_generator
    valid_generator = source.valid_generator
    label_colors    = source.label_colors
except (ImportError, AttributeError, RuntimeError) as e:
    print('[!] Unable to load data source:', str(e))
    sys.exit(1)

#-------------------------------------------------------------------------------
# Print out the info
#-------------------------------------------------------------------------------
print('[i] # training samples:   ', source.num_training)
print('[i] # validation samples: ', source.num_validation)
print('[i] # classes:            ', source.num_classes)
print('[i] Image size:           ', source.image_size)

#-------------------------------------------------------------------------------
# Store 10 annotated training samples in the output_dir
#-------------------------------------------------------------------------------
gen = source.train_generator(10)
x, y = next(gen)

labels = tf.placeholder(tf.float32,
                        shape=[None, None, None, source.num_classes])
label_mapper = tf.argmax(labels, axis=3)

with tf.Session() as sess:
    y_mapped = sess.run(label_mapper, feed_dict={labels: y})

imgs_labelled = draw_labels_batch(x, y_mapped, source.label_colors, False)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(x.shape[0]):
    filename = 'outpout_{:04d}.png'.format(i)
    cv2.imwrite(output_dir + '/' + filename, imgs_labelled[i, :, :, :])

#-------------------------------------------------------------------------------
# Check if all the labels are correct
#-------------------------------------------------------------------------------
gen = source.train_generator(10, names=True)
n_batches = int(math.ceil(source.num_training/10))

for x, y, nx, ny in tqdm(gen, total=n_batches,
                         desc='[i] Checking training labels', unit='batches'):
    check_label_validity(y, ny)

gen = source.valid_generator(10, names=True)
n_batches = int(math.ceil(source.num_validation/10))

for x, y, nx, ny in tqdm(gen, total=n_batches,
                         desc='[i] Checking validation labels', unit='batches'):
    check_label_validity(y, ny)

print('[i] All done.')
