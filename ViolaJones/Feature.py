import SSS.ViolaJones.Regions as region
import numpy as np

class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        """
        This is the actual feature which can also be called a weak classifier.

        :param positive_regions: positively contributing region
        :param negative_regions: negatively contributing region
        :param threshold: threshold
        :param polarity: polarity 1 or -1
        """
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, x):
        """
        Classifies an integral image based on a feature f and the classifiers threshold and polarity

        :param x: the integral image
        :return:
            1 if polarity * feature(x) < polarity * threshold
            0 otherwise
        """
        return 1 if self.polarity * self.computeVal(x) < self.polarity * self.threshold else 0

    def computeVal(self, x):
        """
            Computes the feature value
        :param x: the integral image
        :return: value of the feature by summing positive and subtracting negative region
        """
        return sum([pos.compute_feature(x) for pos in self.positive_regions]) - sum(
            [neg.compute_feature(x) for neg in self.negative_regions])


    def __str__(self):
        return "Feature(WeakClassifier): (threshold=%d, polarity=%d, %s, %s" % (
        self.threshold, self.polarity, str(self.positive_regions), str(self.negative_regions))


def computeFeatures(self, frameSize):
    """
    Builds all possible modified features in frameSize

    :param frameSize: a tuple of form (height, width)
    :return: an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature.
        The second element is an array of rectangle regions negatively contributing to the feature
    """

    height, width = frameSize
    features = []
    for w in range(1, width + 1, 5):
        for h in range(1, height + 1, 5):       # i, j are the positions; w, h are the width and height of feature
            i = 0
            while i + w < width:
                j = 0
                while j + h < height:
                    # 2 rectangle features
                    immediate = region.RectangleRegion(i, j, w, h)
                    right = region.RectangleRegion(i + w, j, w, h)
                    if i + 2 * w < width:  # Horizontally Adjacent
                        features.append(([immediate], [right]))     # positive, negative region to consider

                    bottom = region.RectangleRegion(i, j + h, w, h)
                    if j + 2 * h < height:  # Vertically Adjacent
                        features.append(([immediate], [bottom]))

                    right_2 = region.RectangleRegion(i + 2 * w, j, w, h)
                    # 3 rectangle features
                    if i + 3 * w < width:  # Horizontally Adjacent
                        features.append(([right], [right_2, immediate]))

                    bottom_2 = region.RectangleRegion(i, j + 2 * h, w, h)
                    if j + 3 * h < height:  # Vertically Adjacent
                        features.append(([bottom], [bottom_2, immediate]))

                    # 4 rectangle features
                    bottom_right = region.RectangleRegion(i + w, j + h, w, h)
                    if i + 2 * w < width and j + 2 * h < height:
                        features.append(([right, bottom], [immediate, bottom_right]))

                    j += 5
                i += 5
    print("Computed %d number of features" % (len(features)))
    return np.array(features)


def apply_features(features, training_data):
    """
    :param features:
        An array of tuples [(positive), (negative)].
    :param training_data: Array of tuples [(imageName, intergalImage, classificationValue)].
    :return:
        X: A numpy array of shape (len(features), len(training_data)). Each row represents the value of a single feature for each training example
        y: A numpy array of shape len(training_data). y = training_data[2]
    """
    X = np.zeros((len(features), len(training_data)))
    y = np.array(list(map(lambda data: data[2], training_data)))    # y is only the actual classification of images
    i = 0
    for positive_regions, negative_regions in features:
        feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions]) - sum(
            [neg.compute_feature(ii) for neg in negative_regions])
        # data[0] is training data, feature(data[0]) is where the training data is applied thorough map
        # provide training data to feature function above
        X[i] = list(map(lambda data: feature(data), training_data))
        i += 1
    return X, y
#
# def computeVal(self, x):
#         """
#             Computes the feature value
#         :param x: the integral image
#         :return: value of the feature by summing positive and subtracting negative region
#         """
#         return sum([pos.compute_feature(x) for pos in self.positive_regions]) - sum(
#             [neg.compute_feature(x) for neg in self.negative_regions])