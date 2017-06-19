
Semantic Image Segmentation using a Fully Convolutional Neural Network
======================================================================

Overview
--------

The programs in this repository train and use a fully convolutional neural
network to take an image and classify its pixels. The network is
transfer-trained basing on the VGG-16 model using the approach described in
[this paper][1] by Jonathan Long et al. The software is generic and should be
easily extendable to any dataset, although I only tried with [KITTI][2] Road
Dataset so far. All you need to do to introduce a new dataset is to create a new
`source_xxxxxx.py` file defining your dataset. The definition is a class that
contains seven attributes:

 * `image_size` - self-evident, both horizontal and vertical dimention need to
   be divisible by 32
 * `num_classes` - number of classes that the model is supposed to handle
 * `label_colors` - a dictionary mapping a class number to a color; used for
    blending of the classification results with input image
 * `num_training` - number of training samples
 * `num_validation` - number of validation samples
 * `train_generator` - a generator producing training batches
 * `valid_generator` - a generator producing validation batches

See `source_kitti.py` for a concrete example. The trainer picks the source
based on the value of the `--data-source` parameter.

The KITTI dataset
-----------------

Training the model on the [KITTI Road Dataset][2] essentially means that
`infer.py` will be able to take images from a car's dashcam and paint the road
pink. It generalizes fairly well even to pretty complicated cases:

![Example #1][img1]
![Example #2][img2]
![Example #3][img3]

The model that produces the above image was trained for 500 epochs on the images
contained in [this][3] zip file. The training program fills tensorboard with the
loss summary and a sneak peek of the current performance on validation examples.
The top row contains the ground truth and the bottom one the network's output.

![Loss][img4]
![Validation examples][img5]

[1]: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
[2]: http://www.cvlibs.net/datasets/kitti/eval_road.php
[3]: http://www.cvlibs.net/download.php?file=data_road.zip

[img1]: assets/uu_000022.png
[img2]: assets/uu_000095.png
[img3]: assets/uu_000099.png
[img4]: assets/validation_loss.png
[img5]: assets/validation_examples.png
