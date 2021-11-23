import os
import pathlib

import numpy as np
import tensorflow as tf
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model
from tensorflow.python.data import AUTOTUNE

data_dir = pathlib.Path('Car_Brand_Logos')
image_count_train = len(list(data_dir.glob('Train/*/*.jpg')))
image_count_test = len(list(data_dir.glob('Test/*/*.jpg')))

epochs = 5
train_batch_size = 128
val_batch_size = 6
img_rows, img_cols = 224, 224
steps_per_epoch = np.ceil(image_count_train / train_batch_size)
class_names = np.array([item.name for item in data_dir.glob('Train/*') if item.name != "LICENSE.txt"])
num_classes = len(class_names)

test_ds = tf.data.Dataset.list_files(str(data_dir / 'Test/*/*'))
train_ds = tf.data.Dataset.list_files(str(data_dir / 'Train/*/*'))


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


train_examples = train_ds.map(process_path)
test_examples = test_ds.map(process_path)


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size=train_batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_examples_dataset = prepare_for_training(train_examples)
test_examples_dataset = prepare_for_training(test_examples)

base_model = MobileNet(weights="imagenet", include_top=False)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dense(512, activation="relu")(x)
preds = Dense(8, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()
hist = model.fit(train_examples_dataset,
                 epochs=epochs,
                 steps_per_epoch=image_count_train / train_batch_size,
                 validation_steps=np.floor(image_count_test / train_batch_size),
                 validation_data=test_examples_dataset
                 )
# # save model
model.save("logo_classifier")
