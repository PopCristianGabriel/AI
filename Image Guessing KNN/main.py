import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy

data = tf.keras.datasets.fashion_mnist

(trainImages,trainLabels),(testImages,testLabels) = data.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
trainImages = trainImages / 255.0
testImages = testImages / 255.0


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128,activation = "relu"),
    tf.keras.layers.Dense(10,activation = "softmax")
])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(trainImages,trainLabels,epochs=10)
prediction = model.predict(testImages)

for i in range(20):
    plt.grid(False)
    plt.imshow(testImages[i],cmap=plt.cm.binary)
    plt.xlabel("Actual:" + class_names[testLabels[i]])
    plt.title("Prediction:"+class_names[np.argmax(prediction[i])])
    plt.show()
    




