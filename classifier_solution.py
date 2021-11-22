import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.expand_dims(train_images,-1)
test_images = np.expand_dims(test_images,-1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=1024, validation_data=(test_images,test_labels))

model.evaluate(test_images, test_labels)

model.save('mnist_classifier')
