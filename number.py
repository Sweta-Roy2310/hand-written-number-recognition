import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, utils
import os

mnist = datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#============================== Training Starts ==============================
# model = models.Sequential()

# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # (batch, channels, rows, cols)
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten()) # Converting into an 1D array
# model.add(layers.Dense(units = 64, activation='relu')) # 150 layers of neurons with ReLU activation method (doing the below operations on the input)
# model.add(layers.Dense(units=64, activation='softmax')) # making the generated outputs to be a percent type numeration (basically the ouput layer of the neural network)

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# loss, accuracy = model.evaluate(x_test, y_test)

# print("Accuracy: ", accuracy)
# print("Loss: ", loss)

# model.save("numbers2.model")

#============================== Training Ends ==============================
model = models.load_model("numbers2.model")

root = "written-images2/"
imgList = [ os.path.join(root, item) for item in os.listdir(root) ]

for item in imgList:
    img = cv.imread(item)[:,:,0]
    img = np.invert(np.array([img]))
    img = img.reshape(img.shape[0], 28, 28, 1)
    prediction = model.predict([img])
    print("The result is", np.argmax(prediction))
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

#============================== Problems ==============================

# paint-images:
# Accuracy: 5/10
# written-images1:
# Accuracy: 8/10
# written-images2:
# Accuracy: 9/10
# Can not recognize only nine may be due to quality.
# Unsupervised model