import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Task 1: Normalize the train and test images to be between 0 and 1. Expand the dimensions as well


# Task 2: Create a model that has the following structure:

#   - 1 Conv2D layer of 10 3x3 filters
#   - 1 MaxPool Layer
#   - 1 Dense layer of 128 units
#   - A softmax layer to classify one of the 10 images

model = None  # FILL ME


print(model.summary())

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Task 3: Train the model for 2 epochs and pass the validation data to be the test images and labels

model.fit()  # Fill the required kwargs here

model.evaluate(test_images, test_labels)

tf.saved_model.save(model, 'mnist_classifier')