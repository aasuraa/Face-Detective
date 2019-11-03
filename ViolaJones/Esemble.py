import os
import numpy as np
from PIL import Image


def ensemble_vote(int_img, classifiers):
    """
    Classifies given integral image (numpy array) using given classifiers, i.e.
    if the sum of all classifier votes is greater 0, image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_img: Integral image to be classified
    :type int_img: numpy.ndarray
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: 1 iff sum of classifier votes is greater 0, else 0
    :rtype: int
    """
    return 1 if sum([c.get_vote(int_img) for c in classifiers]) >= 0 else 0

def load_images(path):
    """
    Loads images from the path provided
    :param path:
    :return:
    """
    images = []
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            img_arr = np.array(Image.open((os.path.join(path, file))), dtype=np.float64)
            img_arr /= img_arr.max()            #normalize
            images.append(img_arr)
    return images