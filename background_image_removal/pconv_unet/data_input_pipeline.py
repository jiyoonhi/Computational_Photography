import tensorflow as tf
from tensorflow.keras import layers
from functools import partial
import matplotlib.pyplot as plt


def preprocessing_layer(img_size=512):
    return tf.keras.Sequential([
        layers.RandomFlip(mode="horizontal"),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomRotation((-0.1, 0.1)),
        layers.Resizing(img_size, img_size),
        layers.Rescaling(1./255),
    ])


def decode_image(image, image_shape):
    image = tf.io.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*image_shape, 3])
    return image


def read_tfrecord(example):
    tfrecord_format = (
        {"image/encoded": tf.io.FixedLenFeature([], tf.string),
         "image/height": tf.io.FixedLenFeature([], tf.int64),
         "image/width": tf.io.FixedLenFeature([], tf.int64), }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image/encoded"],
                         (example["image/height"], example["image/width"]))

    return image


def load_dataset(filenames, image_size, training=False):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord), num_parallel_calls=tf.data.AUTOTUNE
    )
    preprocess_image = preprocessing_layer(image_size)
    dataset = dataset.map(
        partial(lambda x: preprocess_image(x, training=training)),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset


def get_dataset(filenames, image_size=512, batch_size=4, training=False):
    dataset = load_dataset(filenames, image_size, training)
    if training:
        dataset = dataset.shuffle(2048)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    return dataset
