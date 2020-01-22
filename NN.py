from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pylab as plt

def load_images(path):
    """
        Initialize the data and labels form the path of the data set given
    :param path: Path to user folders
    :return: array of images and labels respectively
            labels meaning categorical name
    """

    print("[INFO] loading images...")
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    imagePaths = os.listdir(path)   # list of folders in the dataset
    numCal = len(imagePaths)
    random.seed(42)
    random.shuffle(imagePaths)

    # loop over the input images
    for imagePath in imagePaths:
        label = imagePath
        imagePath = path+imagePath+'/'
        for img in os.listdir(imagePath):
            image = cv2.imread(imagePath+img, 0) / 255.0
            print(image.shape)

            #print(image)
            image = image.reshape(*image.shape, 1)
            print(image.shape)
            data.append(image)    # already resized to 30x30
            labels.append(label)
    return data, labels, numCal


# load images for training
data, labels, numCal = load_images('C:/Users/sagar/Desktop/Capstone/images/train/')

(trainX, testX, trainY, testY) = train_test_split(data,	labels, test_size=0.25, random_state=42)

# change to arrays instead of using list for input
trainX = np.array(trainX)
testX = np.array(testX)
trainY = np.array(trainY)
testY = np.array(testY)

# one hot encoding
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)   # fit finds all unique class labels
testY = lb.transform(testY)     # no fit needed as class labels already found


trainY = tf.keras.utils.to_categorical(trainY , numCal)
testY = tf.keras.utils.to_categorical(testY , numCal)

# model architecture code
print("[INFO] building architecture...")

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=(30, 30, 1), name='inputLayer', activation="relu"))
model.add(tf.keras.layers.Conv2D(64, (3,3), name='hiddenLayer1', activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, name='outputLayer', activation="softmax"))

model.summary()

lRate = 0.01
epochs = 10
batch = 10      # size of group of data to pass through the network


# print(trainY)
print("[INFO] training network...")

# TODO: use categorical_crossentropy function as number of users increases
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
# train or fit the model to the data
print(len(trainX), len(testX), len(trainY), len(testY), trainX.shape)

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs)

# evaluate the network
print("[INFO] evaluating network...")

predictions = model.predict(testX, batch_size=batch)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()