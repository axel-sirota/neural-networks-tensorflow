import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

loaded = tf.keras.models.load_model('mnist_classifier')
loaded.summary()
test_images = np.expand_dims(test_images,-1)/255
predictions = loaded.predict(test_images)
print('\nAll Ok!\n')
