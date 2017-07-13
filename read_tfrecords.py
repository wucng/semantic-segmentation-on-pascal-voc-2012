# __author__ = 'dengzelu'
# python version = 3.5.3
# tensorflow version = 1.1.0


import tensorflow as tf


def read_tfrecords(filename_queue):
    """Read a single example from the TFRecord files
    Args:
        filename_queue: queue, a queue of TFRecord files
    Returns：
        result:
        filename: string scalar Tensor, name of the image, such as 2010_001399.jpg
        height: int32 scalar Tensor, image's height
        width: int32 scalar Tensor, image's width
        uint8image: uint8 3-D (height, width, 3) Tensor, jpg image
        annotation: unit8 3-D (height, width, 1) Tensor, corresponding annotation, ranging [0, 20]
    """
    class ImageRecord(object):
        pass

    result = ImageRecord()

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/annotation': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64)
        }
    )

    result.filename = features['image/filename']
    result.uint8image = tf.image.decode_jpeg(features['image/encoded'], channels=3)

    # Attention: bug, reshape must have int32 shape arguments, which can't be int64
    result.height = tf.cast(features['image/height'], tf.int32)
    result.width = tf.cast(features['image/width'], tf.int32)
    result.uint8annotation = tf.reshape(
        tf.decode_raw(features['image/annotation'], tf.uint8),
        tf.stack([result.height, result.width])
    )
    result.uint8annotation = tf.expand_dims(result.uint8annotation, axis=2)
    
    return result
