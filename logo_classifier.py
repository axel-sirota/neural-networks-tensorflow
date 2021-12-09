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


# Task 1: Code the following 4 methods to:

# - Create a generator of all the files under the train and test folders
# - Get the label out of the file path (labels go from 0 to 7)
# - Read the image
# - Decode the image from jpeg, convert to float32, resize to 224 x 224 and normalize.

## INSERT TASK 1 CODE HERE

def create_generator(folder):
    generator = ()
    # FILL ME
    return generator


def get_label(file_path):
    label = 0
    # FILL ME
    return label


def decode_img(img):
    decoded_img = None  # FILL ME
    return decoded_img


def process_path(file_path):
    label = get_label(file_path)
    img = None  # FILL to read img
    img = decode_img(img)
    return img, label


## END TASK 1 CODE


test_ds = create_generator('Test')
train_ds = create_generator('Train')
train_examples = train_ds.map(process_path)
test_examples = test_ds.map(process_path)


# This is to create batches out of our generators, it is not key to the learning objectives and can be copied as is
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

## Validation Task 1
## This is for validation only, after you finish the task feel free to remove the prints and the exit command

print(next(iter(test_examples_dataset))[0].shape)
print(next(iter(test_examples))[0].shape)
exit(0)

## End of validation of task 1. (please remove prints and exits after ending it)

# Task 2: Create a model that has the following structure:

#   - As base model MobileNet without the top layer, recall to make it non trainable
#   - 1 Global Average Pooling
#   - 1 Dense Layer of 1024 units
#   - 1 Dense layer of 512 units
#   - A softmax layer to classify one of the 8 brands

## INSERT TASK 2 CODE HERE

base_model = None  # FILL ME

x = base_model.output
# Fill the rest of the model
preds = Dense()(x)  # Fill the kwargs

## END TASK 2 CODE

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

## Validation Task 2
## This is for validation only, after you finish the task feel free to remove the prints and the exit command
print(model.summary())
exit(0)

# Task 3: Train the model for 5 epochs and pass the validation data to be the test_examples_dataset

## INSERT TASK 3 CODE HERE

model.fit()  # Fill the required kwargs here

## END TASK 3 CODE

# # save model
model.save("logo_classifier")
