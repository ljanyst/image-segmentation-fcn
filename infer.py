#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   15.06.2017
#-------------------------------------------------------------------------------

import argparse
import math
import sys
import cv2
import os

import tensorflow as tf
import numpy as np

from fcnvgg import FCNVGG
from utils import *
from glob import glob
from tqdm import tqdm

#-------------------------------------------------------------------------------
def sample_generator(samples, image_size, batch_size):
    for offset in range(0, len(samples), batch_size):
        files = samples[offset:offset+batch_size]
        images = []
        names  = []
        for image_file in files:
            image = cv2.resize(cv2.imread(image_file), image_size)
            images.append(image.astype(np.float32))
            names.append(os.path.basename(image_file))
        yield np.array(images), names

#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate data based on a model')
parser.add_argument('--name', default='test',
                    help='project name')
parser.add_argument('--checkpoint', type=int, default=-1,
                    help='checkpoint to restore; -1 is the most recent')
parser.add_argument('--samples-dir', default='test',
                    help='directory containing samples to analyse')
parser.add_argument('--output-dir', default='test-output',
                    help='directory for the resulting images')
parser.add_argument('--batch-size', type=int, default=20,
                    help='batch size')
parser.add_argument('--data-source', default='kitti',
                    help='data source')
args = parser.parse_args()

#-------------------------------------------------------------------------------
# Check if we can get the checkpoint
#-------------------------------------------------------------------------------
state = tf.train.get_checkpoint_state(args.name)
if state is None:
    print('[!] No network state found in ' + args.name)
    sys.exit(1)

try:
    checkpoint_file = state.all_model_checkpoint_paths[args.checkpoint]
except IndexError:
    print('[!] Cannot find checkpoint ' + str(args.checkpoint_file))
    sys.exit(1)

metagraph_file = checkpoint_file + '.meta'

if not os.path.exists(metagraph_file):
    print('[!] Cannot find metagraph ' + metagraph_file)
    sys.exit(1)

#-------------------------------------------------------------------------------
# Load the data source
#-------------------------------------------------------------------------------
try:
    source       = load_data_source(args.data_source)
    label_colors = source.label_colors
except (ImportError, AttributeError, RuntimeError) as e:
    print('[!] Unable to load data source:', str(e))
    sys.exit(1)

#-------------------------------------------------------------------------------
# Create a list of files to analyse and make sure that the output directory
# exists
#-------------------------------------------------------------------------------
samples = glob(args.samples_dir + '/*.png')
if len(samples) == 0:
    print('[!] No input samples found in', args.samples_dir)
    sys.exit(1)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
print('[i] Project name:      ', args.name)
print('[i] Network checkpoint:', checkpoint_file)
print('[i] Metagraph file:    ', metagraph_file)
print('[i] Number of samples: ', len(samples))
print('[i] Output directory:  ', args.output_dir)
print('[i] Image size:        ', source.image_size)
print('[i] # classes:         ', source.num_classes)
print('[i] Batch size:        ', args.batch_size)

#-------------------------------------------------------------------------------
# Create the network
#-------------------------------------------------------------------------------
with tf.Session() as sess:
    print('[i] Creating the model...')
    net = FCNVGG(sess)
    net.build_from_metagraph(metagraph_file, checkpoint_file)

    #---------------------------------------------------------------------------
    # Process the images
    #---------------------------------------------------------------------------
    generator = sample_generator(samples, source.image_size, args.batch_size)
    n_sample_batches = int(math.ceil(len(samples)/args.batch_size))
    description = '[i] Processing samples'

    for x, names in tqdm(generator, total=n_sample_batches,
                        desc=description, unit='batches'):
        feed = {net.image_input:  x,
                net.keep_prob:    1}
        img_labels = sess.run(net.classes, feed_dict=feed)
        imgs = draw_labels_batch(x, img_labels, label_colors, False)

        for i in range(len(names)):
            cv2.imwrite(args.output_dir + '/' + names[i], imgs[i, :, :, :])
print('[i] All done.')
