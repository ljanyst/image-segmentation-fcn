#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   15.06.2017
#-------------------------------------------------------------------------------

import random
import cv2
import os

import numpy as np

from collections import namedtuple
from glob import glob

#-------------------------------------------------------------------------------
# Labels
#-------------------------------------------------------------------------------
Label = namedtuple('Label', ['name', 'color'])

def rgb2bgr(tpl):
    return (tpl[2], tpl[1], tpl[0])

label_defs = [
    Label('unlabeled',     rgb2bgr((0,     0,   0))),
    Label('dynamic',       rgb2bgr((111,  74,   0))),
    Label('ground',        rgb2bgr(( 81,   0,  81))),
    Label('road',          rgb2bgr((128,  64, 128))),
    Label('sidewalk',      rgb2bgr((244,  35, 232))),
    Label('parking',       rgb2bgr((250, 170, 160))),
    Label('rail track',    rgb2bgr((230, 150, 140))),
    Label('building',      rgb2bgr(( 70,  70,  70))),
    Label('wall',          rgb2bgr((102, 102, 156))),
    Label('fence',         rgb2bgr((190, 153, 153))),
    Label('guard rail',    rgb2bgr((180, 165, 180))),
    Label('bridge',        rgb2bgr((150, 100, 100))),
    Label('tunnel',        rgb2bgr((150, 120,  90))),
    Label('pole',          rgb2bgr((153, 153, 153))),
    Label('traffic light', rgb2bgr((250, 170,  30))),
    Label('traffic sign',  rgb2bgr((220, 220,   0))),
    Label('vegetation',    rgb2bgr((107, 142,  35))),
    Label('terrain',       rgb2bgr((152, 251, 152))),
    Label('sky',           rgb2bgr(( 70, 130, 180))),
    Label('person',        rgb2bgr((220,  20,  60))),
    Label('rider',         rgb2bgr((255,   0,   0))),
    Label('car',           rgb2bgr((  0,   0, 142))),
    Label('truck',         rgb2bgr((  0,   0,  70))),
    Label('bus',           rgb2bgr((  0,  60, 100))),
    Label('caravan',       rgb2bgr((  0,   0,  90))),
    Label('trailer',       rgb2bgr((  0,   0, 110))),
    Label('train',         rgb2bgr((  0,  80, 100))),
    Label('motorcycle',    rgb2bgr((  0,   0, 230))),
    Label('bicycle',       rgb2bgr((119,  11,  32)))]

#-------------------------------------------------------------------------------
def build_file_list(images_root, labels_root, sample_name):
    image_sample_root = images_root + '/' + sample_name
    image_root_len    = len(image_sample_root)
    label_sample_root = labels_root + '/' + sample_name
    image_files       = glob(image_sample_root + '/**/*png')
    file_list         = []
    for f in image_files:
        f_relative      = f[image_root_len:]
        f_dir           = os.path.dirname(f_relative)
        f_base          = os.path.basename(f_relative)
        f_base_gt = f_base.replace('leftImg8bit', 'gtFine_color')
        f_label   = label_sample_root + f_dir + '/' + f_base_gt
        if os.path.exists(f_label):
            file_list.append((f, f_label))
    return file_list

#-------------------------------------------------------------------------------
class CityscapesSource:
    #---------------------------------------------------------------------------
    def __init__(self):
        self.image_size      = (512, 256)
        self.num_classes     = len(label_defs)

        self.label_colors    = {i: np.array(l.color) for i, l \
                                                     in enumerate(label_defs)}

        self.num_training    = None
        self.num_validation  = None
        self.train_generator = None
        self.valid_generator = None

    #---------------------------------------------------------------------------
    def load_data(self, data_dir, valid_fraction):
        """
        Load the data and make the generators
        :param data_dir:       the directory where the dataset's file are stored
        :param valid_fraction: what franction of the dataset should be used
                               as a validation sample
        """
        images_root = data_dir + '/samples/leftImg8bit'
        labels_root = data_dir + '/labels/gtFine'

        train_images = build_file_list(images_root, labels_root, 'train')
        valid_images = build_file_list(images_root, labels_root, 'val')

        if len(train_images) == 0:
            raise RuntimeError('No training images found in ' + data_dir)
        if len(valid_images) == 0:
            raise RuntimeError('No validatoin images found in ' + data_dir)

        self.num_training    = len(train_images)
        self.num_validation  = len(valid_images)
        self.train_generator = self.batch_generator(train_images)
        self.valid_generator = self.batch_generator(valid_images)

    #---------------------------------------------------------------------------
    def batch_generator(self, image_paths):
        def gen_batch(batch_size, names=False):
            random.shuffle(image_paths)
            for offset in range(0, len(image_paths), batch_size):
                files = image_paths[offset:offset+batch_size]

                images = []
                labels = []
                names_images = []
                names_labels = []
                for f in files:
                    image_file = f[0]
                    label_file = f[1]

                    image = cv2.resize(cv2.imread(image_file), self.image_size)
                    label = cv2.resize(cv2.imread(label_file), self.image_size)

                    label_bg   = np.zeros([image.shape[0], image.shape[1]], dtype=bool)
                    label_list = []
                    for ldef in label_defs[1:]:
                        label_current  = np.all(label == ldef.color, axis=2)
                        label_bg      |= label_current
                        label_list.append(label_current)

                    label_bg   = ~label_bg
                    label_all  = np.dstack([label_bg, *label_list])
                    label_all  = label_all.astype(np.float32)

                    images.append(image.astype(np.float32))
                    labels.append(label_all)

                    if names:
                        names_images.append(image_file)
                        names_labels.append(label_file)

                if names:
                    yield np.array(images), np.array(labels), \
                          names_images, names_labels
                else:
                    yield np.array(images), np.array(labels)
        return gen_batch

#-------------------------------------------------------------------------------
def get_source():
    return CityscapesSource()
