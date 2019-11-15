import numpy as np
import math
import SSS.ViolaJones.Integral as intImg
import SSS.ViolaJones.Feature as feature

class ViolaJones:
    def __init__(self, T=10):
        """
          Args:
            T: The number of times to loop and train the selected weak classifiers which should be used
        """
        self.T = T

    def train(self, posImg, negImg):
        """
            Trains the Viola Jones classifier using boosting.

        :param posImg: List of Positive Images, (name, ndarray of images)
        :param negImg: List of Negative Images, (name, ndarray of images)
        :param pos_num: number of positive samples
        :param neg_num: number of negative samples
        :return:
        """

        posNum = len(posImg)
        negNum = len(negImg)
        posIntImages = intImg.powerfulIntegral(posImg, 1)  # array of (name, ii, val)
        negIntImages = intImg.powerfulIntegral(negImg, 0)

        training_data = posIntImages+negIntImages
        weights = np.zeros(len(training_data))
        print("Computing weights of images")
        for x in range(len(training_data)):
            if training_data[x][2] == 1:
                weights[x] = 1.0 / (2 * posNum)
            else:
                weights[x] = 1.0 / (2 * negNum)

        print("Building features")
        features = feature.computeFeatures((10, 10))
        print("Applying features to training examples")
        X, y = features.apply_features(features, training_data)
        print("Selecting best features")

        # TODO: Get the top 10% of features which classified the images with min error to proceed, that's the new feature list
        # take the feature and the whole row value of it corresponding to the image data set


        print("Features selected")

        for t in range(self.T):
            """
            Normalize weights
            Train weak classifiers selected
            get errors, beta value
            get alpha value
            select the classifiers as strong classifiers
            """
            weights = weights / sum(weights)    # normalize weights
            weak_classifiers = self.train_weak(X, y, features, weights)

    def train_weak(self, X, y, features, weights):
        """
            Finds optimal threshold for the classifier
        :param X: Selected features for training
        :param y: actual positive or negative representation of images
        :param features: all the features
        :param weights: weights associated with training data
        :return: array of weak classifiers
        """
        """
            total pos and negative weights
            go through features
            use error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights) as min error
            select feature with this error as best feature
            the feature value is best threshold value
            select polarity based on num of pos and num of neg
            
        """
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w          # sum of positive weight
            else:
                total_neg += w

        classifiers = []        # stores my classifiers
        total_features = X.shape[0]     # no. of selected features
