import pandas as pd

class RectangleRegion:
    """
        The Rectangle region behaves as a part of features.
    """

    def __init__(self, x, y, width, height):
        """
            Initialize the rectangle region.

        :param x: x coordinate of the upper left corner of the rectangle
        :param y: y coordinate of the upper left corner of the rectangle
        :param width: width of the rectangle
        :param height: height of the rectangle
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def computeScore(self, ii):
        """
            Computes the rectangle region value (pos neg val) given the integral image of type name, integralImage, val
        (pos or neg)
        If integral image is positive, we need to refer to csv file else, have default values for negative images.
        From csv file add xmin and ymin value to the height and width of the region to calculate it's value.
        We don't want to save the feature with this height or width because it's different for different images.
        """
        csvFile = pd.read_csv('C:/Users/sagar/Desktop/CSC485/vjtest/face_labels.csv')
        # if val is 1, read the csv file and get min i and j to add to positions
        if ii[2] == 1:  # i is x, j is y which now becomes starting position for faces or non faces
            row = csvFile.loc[csvFile['filename'] == ii[0]]  # go to row with image name
            i = int(row['xmin'])
            j = int(row['ymin'])
        else:   # position somewhere in the middle to get rid of bias where neg images have white background
            i = 20
            j = 20
        return ii[1][self.y + self.height + j][self.x + self.width + i] + ii[1][self.y + j][self.x + i] - (
                    ii[1][self.y + self.height + j][self.x + i] + ii[1][self.y + j][self.x + self.width + i])