# pipeline comes from https://github.com/tensorflow/models/blob/master/inception/inception/data/build_imagenet_data.py
# python version = 3.5.3
# tensorflow version = 1.1.0
# =====================================================================================================================
# README
# You need to change FLAGS parameters based on your own situation.
# The following parameters work on my own machine. 
# After your decompress the VOCtrainval_11-May-2012.tar file, your parameters may very similar to mine.
# The mode must either be training or validation, and this script will convert the images
# and annotaions to TFRecord respectively.
# =====================================================================================================================

import os
from datetime import datetime
import random

import tensorflow as tf
import numpy as np
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('training_txt',
                           'd:/DataSets/VOC2012/VOC2012/ImageSets/Segmentation/train.txt',
                           'file path to train.txt')
tf.app.flags.DEFINE_string('val_txt',
                           'd:/DataSets/VOC2012/VOC2012/ImageSets/Segmentation/val.txt',
                           'file path to val.txt')
tf.app.flags.DEFINE_string('images_dir',
                           'd:/DataSets/VOC2012/VOC2012/JPEGImages',
                           'file folder to VOC2012 JPEGImages')
tf.app.flags.DEFINE_string('annotations_dir',
                           'd:/DataSets/VOC2012/VOC2012/SegmentationClass',
                           'file folder to VOC2012 SegmentationClass')
tf.app.flags.DEFINE_string('output_dir',
                           'd:/TFRecords/VOC2012',
                           'file folder where this script output tfrecords')
tf.app.flags.DEFINE_string('mode',
                           'training',
                           'either training or validation')


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _convert_to_example(filename, image_buffer, annotation_buffer,
                        height, width):
    """Convert an image to an example
    Args:
        filename: string, name of an image or an annotation, such as 2007_000032.jpg
        image_buffer: bytes, JPEG encoding for a RGB image
        annotation_buffer: bytes, corresponding annotation
        height: integer, image's height
        width: integer, image's width
    Returns:
        an example
    Attention:
        1. all string parameters should be converted to bytes using encode
        2. example.SerializeToString() gets bytes and writes them to the disk
    """

    # convert string to bytes
    filename = filename.encode(encoding='utf-8')

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': _bytes_feature(filename),
        'image/encoded': _bytes_feature(image_buffer),
        'image/annotation': _bytes_feature(annotation_buffer),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width)
    }))
    return example


class ImageDecoder(object):
    def __init__(self):

        # decode RGB JPEG image data
        self._jpeg_data = tf.placeholder(dtype=tf.string)
        self._jpeg_image = tf.image.decode_jpeg(self._jpeg_data, channels=3)

        self._sess = tf.Session()

    def decode_jpeg(self, image_data):
        # image is an numpy array with shape (height, width, 3)
        image = self._sess.run(self._jpeg_image,
                               feed_dict={self._jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _process_image_and_annotation(jpg_file, png_file, decoder):
    """Process a single a image and its corresponding annotation
    Args:
        jpg_file: string, path to an image
        png_file: string, path to the corresponding annotation
        decoder: instance of ImageDecoder to provide image decoding utils
    Returns:
        image_data: bytes, JPEG encoding of the RGB image file
        annotation_data: bytes, corresponding annotation, np.array().tobytes()
        height: integer, image's height
        width: integer, image's width
    """

    # Read the jpg image
    with tf.gfile.FastGFile(jpg_file, mode='rb') as reader:
        image_data = reader.read()

    annotation = np.array(Image.open(png_file))
    # transform the void or unlabelled to background (255 to 0)
    # here annotation [0, 1, 2, ..., 20, 255]
    # more information http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html
    annotation[annotation == 255] = 0
    annotation_data = annotation.tobytes()

    # And get image's height and width
    image = decoder.decode_jpeg(image_data)
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    height = image.shape[0]
    width = image.shape[1]

    # just to make sure
    assert len(annotation.shape) == 2
    assert annotation.shape[0] == height
    assert annotation.shape[1] == width

    return image_data, annotation_data, height, width


def _process_files(jpg_files, png_files, mode):
    """Process and save list of images
    Args:
        jpg_files: list of string, each string is a path a jpg image
        png_files: list of string, each string is a path the corresponding annotation
        mode: string, either training or validation
    """
    decoder = ImageDecoder()

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MkDir(FLAGS.output_dir)
    output_file = os.path.join(FLAGS.output_dir, mode)
    writer = tf.python_io.TFRecordWriter(output_file)

    assert len(jpg_files) == len(png_files)
    num_files = len(jpg_files)

    for i_file in range(num_files):

        jpg_file = jpg_files[i_file]
        png_file = png_files[i_file]

        image_buffer, annotation_buffer, height, width =\
            _process_image_and_annotation(jpg_file, png_file, decoder)

        filename = os.path.basename(jpg_file)
        example = _convert_to_example(filename, image_buffer, annotation_buffer,
                                      height, width)

        # convert example to bytes and write them to output directory
        writer.write(example.SerializeToString())

        if not (i_file + 1) % 100:
            print('%s: Processed %d of %d files.'
                  % (datetime.now(), i_file + 1, num_files))

    writer.close()
    print('%s: Finished writing %d files.' % (datetime.now(), num_files))


def _find_files(mode):
    """Find all images and corresponding annotations
    Args:
        mode: string, either training or validation
    Returns:
        jpg_files: list of string, each string is a path a jpg image
        png_files: list of string, each string is a path the corresponding annotation
    """
    if mode == 'training':
        with open(FLAGS.training_txt, 'r') as fr:
            filenames_raw = fr.readlines()
    elif mode == 'validation':
        with open(FLAGS.val_txt, 'r') as fr:
            filenames_raw = fr.readlines()
    else:
        raise ValueError('mode = either training or validation')

    filenames = []
    for i_raw in filenames_raw:
        filenames.append(i_raw.replace('\n', ''))
    matching_jpg_files = [os.path.join(FLAGS.images_dir, '%s.jpg' % filename)
                          for filename in filenames]
    matching_png_files = [os.path.join(FLAGS.annotations_dir, '%s.png' % filename)
                          for filename in filenames]

    assert len(matching_jpg_files) == len(matching_png_files)

    shuffled_index = list(range(len(matching_jpg_files)))
    random.seed(12345678)
    random.shuffle(shuffled_index)
    random.seed(23456789)
    random.shuffle(shuffled_index)
    random.seed(34567890)
    random.shuffle(shuffled_index)

    jpg_files = [matching_jpg_files[k] for k in shuffled_index]
    png_files = [matching_png_files[k] for k in shuffled_index]

    print('Found %d files' % len(jpg_files))

    return jpg_files, png_files


def _process_dataset(mode):
    jpg_files, png_files = _find_files(mode)
    print('Found %d jpg files for %s' % (len(jpg_files), mode))
    _process_files(jpg_files, png_files, mode)


def main(_):
    _process_dataset(FLAGS.mode)


if __name__ == '__main__':
    tf.app.run()
