import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Task 1: Normalize the train and test images to be between 0 and 1. Expand the dimensions as well

## INSERT TASK 1 CODE HERE

## END TASK 1 CODE

## Validation Task 1
## This is for validation only, after you finish the task please remove the prints and the exit command

print(train_images.shape)
print(np.max(train_images[0]))
exit(0)

## End of validation of task 1. (please remove prints and exits after ending it)

# Task 2: Create a model that has the following structure:

#   - 1 Conv2D layer of 10 3x3 filters
#   - 1 MaxPool Layer
#   - 1 Dense layer of 128 units
#   - A softmax layer to classify one of the 10 images

## INSERT TASK 2 CODE HERE

model = None  # FILL ME

## END TASK 2 CODE


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

## Validation Task 2
## This is for validation only, after you finish the task feel free to remove the prints and the exit command
print(model.summary())
exit(0)

## End of validation of task 2. (please remove prints and exits after ending it)

# Task 3: Train the model for 2 epochs and pass the validation data to be the test images and labels

## INSERT TASK 3 CODE HERE

model.fit()  # Fill the required kwargs here

## END TASK 3 CODE

model.evaluate(test_images, test_labels)

model.save('mnist_classifier')
