import numpy as np
import cv2
import pandas as pd

def to_integral_image(image):
    """
    Calculates the integral image based on this instance's original image data
    :param img_arr: Image source data of type np.darray
    :return Integral image for given image of type np.darray with same dimension as of provided image
    """
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            # from vj algorithm paper
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii

def sum_region(ii, top_left, bottom_right):
    """
    Calculates the sum of the rectangle in a feature
    :param integral_img_arr:
    :param top_left: (x, y) of the rectangle's top left corner
    :param bottom_right: (x, y) of the rectangle's bottom right corner
    :return The sum of all pixels in the given rectangle, returns int
    """
    top_left = (top_left[1], top_left[0])
    bottom_right = (bottom_right[1], bottom_right[0])
    if top_left == bottom_right:
        return ii[top_left]
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    return ii[bottom_right] - ii[top_right] - ii[bottom_left] + ii[top_left]

# these are all the basic features used, height by width
allFeatures = [(1, 2), (2, 1), (1, 3), (3, 1)]

class Feature():
    def __init__(self, type, position, w, h):
        """
        :param type: 1, 2, 3, 4 corresconding to basic features
        :param position: top left posltion of harr feature
        :param w: width of feature
        :param h: height of feature
        """
        self.type = type
        self.tlpos= position
        self.w = w
        self.h = h
        self.brpos = (position[0]+h, position[1]+w)
        if self.type == allFeatures[0] or self.type == allFeatures[1]:        # 2 vertical or horizontal
            self.divide = 2
        if self.type == allFeatures[2] or self.type == allFeatures[3]:        # 3 vertical or horizontal
            self.divide = 3

    def out(self):
        if self.type == (1,2):
            name = "2 Horizontal"
        elif self.type == (2,1):
            name = "2 Vertical"
        elif self.type == (3, 1):
            name = "3 Horizontal"
        elif self.type == (1,3):
            name = "3 Vertical"
        else:
            name = "DNE"
        print(name," (height, width): ", self.h, self.w, " position(x, y): ", self.tlpos, " divide val: ", self.divide)

def createFeatures(minFeatureWidth=6, minFeatureHeight=3, maxFeatureWidth=24, maxFeatureHeight=24, frameWidth=60, frameHeight=60):     # intended to work for 60x60
    """
    Creates all the features to be used as weak classifiers for the model
    The combination of best features among these weak classifiers will be used to build strong classifiers
    :param frameWidth:
    :param frameHeight:
    :param minFeatureWidth:
    :param maxFeatureWidth:
    :param minFeatureHeight:
    :param maxFeatureHeight:
    :return:
    """
    features = []
    f=0
    print("..creating haar features..")
    for i in allFeatures:
        startWidth = max(minFeatureWidth, i[1])
        for w in range(startWidth, maxFeatureWidth, i[1]):
            startHeight = max(minFeatureHeight, i[0])
            for h in range(startHeight, maxFeatureHeight, i[1]):
                for x in range(0, frameWidth, 6):          # shifting position of feature by 2 while creation
                    for y in range(0, frameHeight, 6):
                        if(f >= 3):
                            break
                        features.append(Feature(i, (x, y), w, h))
                        f+=1
    print(f)
    return features

all = createFeatures()
for x in all:
    print(x.out())
    print(x.brpos)


# img = cv2.imread("test.jpg",0)
# print(img)

# df = pd.read_csv('../face_labels.csv')
# row = df.loc[df['filename'] == '1.jpg']
# width = row['width']
# print(int(width + 1))

# print(all[0].position[0]+all[0].h, all[0].position[1]+all[0].w)
# print(sum_region(to_integral_image(img), all[0].position, (all[0].position[0]+all[0].w, all[0].position[1]+all[0].h)))
