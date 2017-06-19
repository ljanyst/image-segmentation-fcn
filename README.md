
Semantic Image Segmentation using a Fully Convolutional Neural Network
======================================================================

Overview
--------

The programs in this repository train and use a fully convolutional neural
network to take an image and classify its pixels. The network is
transfer-trained basing on the VGG-16 model using the approach described in
[this paper][1] by Jonathan Long et al. The software is generic and easily
extendable to any dataset, although I only tried with [KITTI][2] Road
Dataset and [Cityscapes][4] dataset so far. All you need to do to introduce a
new dataset is to create a new `source_xxxxxx.py` file defining your dataset.
The definition is a class that contains seven attributes:

 * `image_size` - self-evident, both horizontal and vertical dimention need to
   be divisible by 32
 * `num_classes` - number of classes that the model is supposed to handle
 * `label_colors` - a dictionary mapping a class number to a color; used for
    blending of the classification results with input image
 * `num_training` - number of training samples
 * `num_validation` - number of validation samples
 * `train_generator` - a generator producing training batches
 * `valid_generator` - a generator producing validation batches

See `source_kitti.py` or `source_cityscapes.py` for a concrete example. The
trainer picks the source based on the value of the `--data-source` parameter.

The KITTI dataset
-----------------

Training the model on the [KITTI Road Dataset][2] essentially means that
`infer.py` will be able to take images from a car's dashcam and paint the road
pink. It generalizes fairly well even to pretty complicated cases:

![Example #1][img1]
![Example #2][img2]
![Example #3][img3]

The model that produced the above images was trained for 500 epochs on the images
contained in [this][3] zip file. The training program fills tensorboard with the
loss summary and a sneak peek of the current performance on validation examples.
The top row contains the ground truth and the bottom one the network's output.

![Loss][img4]
![Validation examples][img5]

The Cityscapes dataset
----------------------

This dataset is more complex than the previous one. It has fine image
annotations for 29 classes of objects. The images are video frames taken in
German cities and there is around 11GB of them.

![Example #1][img6]
![Example #2][img7]
![Example #3][img8]

The model that produced the images was trained for 150 epochs.

![Loss][img9]
![Validation examples][img10]

[1]: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
[2]: http://www.cvlibs.net/datasets/kitti/eval_road.php
[3]: http://www.cvlibs.net/download.php?file=data_road.zip
[4]: https://www.cityscapes-dataset.com/

[img1]: assets/uu_000022.png
[img2]: assets/uu_000095.png
[img3]: assets/uu_000099.png
[img4]: assets/kitti_validation_loss.png
[img5]: assets/kitti_validation_examples.png

[img6]: assets/berlin_000000_000019_leftImg8bit.png
[img7]: assets/berlin_000002_000019_leftImg8bit.png
[img8]: assets/berlin_000003_000019_leftImg8bit.png
[img9]: assets/cityscapes_validation_loss.png
[img10]: assets/cityscapes_validation_examples.png
