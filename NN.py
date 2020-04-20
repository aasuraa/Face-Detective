import pickle
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pylab as plt
import os
import random
import cv2



class FaceNeural:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.here = "here"

    def train(self):
        """
            Only for training
        """
        print("[INFO] initializing train...")
        # load images for training
        path = "./dataset/"
        data, labels, numCal = self.load_images(path)

        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, random_state=42)

        print("[INFO] data split...")
        # change to arrays instead of using list for input
        trainX = np.array(trainX)
        testX = np.array(testX)
        trainY = np.array(trainY)
        testY = np.array(testY)

        print(len(trainX) + len(testX), len(trainX), len(testX), len(trainY), len(testX), trainX.shape)

        # one hot encoding
        lb = LabelBinarizer()
        trainY = lb.fit_transform(trainY)  # fit finds all unique class labels
        testY = lb.transform(testY)  # no fit needed as class labels already found

        # model architecture code
        print("[INFO] building architecture...")
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(1024, input_shape=(30, 30, 1), name='inputLayer', activation="relu"))
        model.add(tf.keras.layers.Dense(512, name='hiddenLayer1', activation="relu"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(len(lb.classes_), name='outputLayer', activation="softmax"))

        model.summary()

        epochs = 1
        print("[INFO] training network...")
        adam = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # rms = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9)
        # sgd = tf.keras.optimizers.SGD(lr=0.0015, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=("categorical_crossentropy" if (len(lb.classes_)) == 2 else "binary_crossentropy"),
                      optimizer=adam, metrics=["accuracy"])

        # train or fit the model to the data using k cross validation
        kf = KFold(n_splits=10, random_state=1)
        print("[INFO] Splitting data into ", kf, "...")
        data = []
        for a, b in kf.split(trainX, trainY):  # takes 10% of data for validation
            H = model.fit(trainX[a], trainY[a], batch_size=32, validation_data=(trainX[b], trainY[b]), epochs=epochs)
            data.append(H)

        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = model.predict(testX, batch_size=5)
        print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

        # plot the training loss and accuracy
        N = np.arange(0, epochs)
        plt.style.use("ggplot")
        plt.figure()
        # for H in data:
        H = data[8]
        plt.plot(N, H.history["loss"], label="train_loss")
        plt.plot(N, H.history["val_loss"], label="val_loss")
        plt.plot(N, H.history["accuracy"], label="train_acc")
        plt.plot(N, H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy (Simple NN)")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(self.dataPath + "plot.png")

        print("[INFO] evaluation done...")

        model.save(self.dataPath + 'newModel.h5')
        self.saveLB(self.dataPath + "lb", lb)
        print("[INFO] serializing network and label binarizer done...")

        self.model = tf.keras.models.load_model(self.dataPath + "newModel.h5")
        self.lb = self.load(self.dataPath + "lb")
        print("[INFO] reloading of model successful...")

    # methods for saving and loading the binarizer
    def saveLB(self, filename, lb):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename + ".pkl", 'wb') as f:
            f.write(pickle.dumps(lb))
            f.close()

    def load(self, filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename + ".pkl", 'rb') as f:
            return pickle.load(f)

    def load_images(self, path):
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
        imagePaths = os.listdir(path)  # list of folders in the dataset
        numCal = len(imagePaths)
        random.seed(42)
        random.shuffle(imagePaths)

        # loop over the input images
        for imagePath in imagePaths:
            label = imagePath
            imagePath = path + imagePath + '/'
            for img in os.listdir(imagePath):
                image = cv2.imread(imagePath + img, 0) / 255.0
                image = image.reshape(*image.shape, 1)
                data.append(image)  # already resized to 30x30
                labels.append(label)
        return data, labels, numCal