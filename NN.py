from __future__ import print_function
import os
import cv2
import random

class FaceNeural():
    def __int__(self):
        self.path = "C:/Users/sagar/Desktop/Capstone/dataset/"      # contains all the folders

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
                image = image.reshape(*image.shape, 1)
                data.append(image)    # already resized to 30x30
                labels.append(label)
        return data, labels, numCal

    def NeuralNetwork(self):
        pass
