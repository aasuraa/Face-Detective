import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif
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
        features = feature.computeFeatures((40, 40))
        print("Applying features to training examples")
        X, y = feature.apply_features(features, training_data)
        print("Selecting best features")

        # TODO: Get the top 10% of features which classified the images with min error to proceed, that's the new feature list
        # take the feature and the whole row value of it corresponding to the image data set
        print(X)
        print(X[0])
        print(X[1])
        print(X[0][0])
        print(X[1][0])

        print("Features selected")

        # bases on number of highest scores for each features, select the best % of features
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True) # returns an array
        X = X[indices]  # selecting best features row through returned indices
        features = features[indices]    # new features is best features only

        print("Selected %d potential features" % len(X))
        print("and %d features" % len(features))


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
            print(weak_classifiers)


    def train_weak(self, X, y, features, weights):
        """
            this algorithm is designed to select a single rectangle feature which best separates the positive and negative
        examples with respect to weights of images; using min error to calculate error by feature rather than weighted error
        to compute error at constant time

        :param X: Selected features for training
        :param y: actual positive or negative representation of images
        :param features: all the features
        :param weights: weights associated with training data
        :return: array of weak classifiers
        """
        totalPostW, totalNegW = 0, 0     # total weights
        for w, label in zip(weights, y):        # this has to happen for every training to get weights
            if label == 1:
                totalPostW += w          # sum of positive weight
            else:
                totalNegW += w          # sum of neg weight

        classifiers = []        # stores my classifiers
        total_features = X.shape[0]     # no. of features

        # this loop goes through all the features and selects couple of best p, t with minimum error
        for index, feature in enumerate(X):     # X is 2d array, so we give those values an index to go through it
            if len(classifiers) % 2 == 0 and len(classifiers) != 0:     # print every nth classifier
                print("Trained %d classifiers out of %d" % (len(classifiers), total_features))

            # returns list of tuples
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])  # sorting with respect to feature

            pos_seen, neg_seen = 0, 0   # num of pos or neg seen
            pos_weights, neg_weights = 0, 0     # pos or neg weights seen
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None

            # this loop finds the min error, threshold and polarity of the feature to be selected if has min error
            for w, f, label in applied_feature:     # for this feature 'fi' with weight 'wi' and label 'li'
                error = min(neg_weights + totalPostW - pos_weights, pos_weights + totalNegW - neg_weights)
                if error < min_error:   # at first it's always less, next time it might be or not
                    min_error = error
                    best_feature = features[index]  # feature at the index with pos and neg region
                    best_threshold = f  # feature value at this min error gives best threshold
                    best_polarity = 1 if pos_seen > neg_seen else -1    # if more positive seen, it's positively polar

                if label == 1:      # pos image
                    pos_seen += 1   # num of positive images seen and weight increases
                    pos_weights += w
                else:
                    neg_seen += 1   # neg image, num of neg images seen and neg weight increases
                    neg_weights += w

            clf = feature.WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers