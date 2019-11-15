import pandas as pd
import cv2
import SSS.ViolaJones.ViolaJones as VJ
import os

def load_images(path):
    images = []
    for img in os.listdir(path):
        if img.endswith('.jpg'):
            images.append((img, cv2.imread(path+img, 0)))
    return images

# file paths: csv, positive, negative
print("Loading files")
csvFile = pd.read_csv('C:/Users/sagar/Desktop/CSC485/objectDet/face_labels.csv')
posPath = 'C:/Users/sagar/Desktop/CSC485/vjtest/Images/train/positive/'
negPath = 'C:/Users/sagar/Desktop/CSC485/vjtest/Images/train/negative/'
posImg = load_images(posPath)      # has and returns array of (name, read(image))
negImg = load_images(negPath)      # (name, image)


vj = VJ.ViolaJones(5)

vj.train(posImg, negImg)


# features = vj.build_features((60, 60))
# print(type(features))
# print(type(features[3]))
# print(type(features[3][0][0]))
# print(type(features[3][1][0]))
# # print(features[3][0][0])
# print(features[3][0][0].compute_feature(integral_image(cv2.imread("test.jpg",0))))
# print(type(features[3][1][0]))
# print(len(features))

