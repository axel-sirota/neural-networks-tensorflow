import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

data_dir = pathlib.Path('Car_Brand_Logos')
image_count_test = len(list(data_dir.glob('Test/*/*.jpg')))

img_rows, img_cols = 224, 224
class_names = np.array([item.name for item in data_dir.glob('Train/*') if item.name != "LICENSE.txt"])
num_classes = len(class_names)

test_ds = tf.data.Dataset.list_files(str(data_dir / 'Test/*/*'))

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return class_names == parts[-2]


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [img_rows, img_cols])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def format_image(image, label):
    image = tf.image.resize(image, (img_rows, img_cols)) / 255.0
    return image, label


test_examples = test_ds.map(process_path)


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size=128)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


test_examples_dataset = prepare_for_training(test_examples)
loaded = tf.keras.models.load_model('logo_classifier')
loaded.summary()
predictions = loaded.predict(test_examples_dataset.take(1))
print('\nAll Ok!\n')
