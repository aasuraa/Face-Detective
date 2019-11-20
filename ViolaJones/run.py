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
# csvFile = pd.read_csv('C:/Users/sagar/Desktop/CSC485/objectDet/face_labels.csv')
posPath = 'C:/Users/sagar/Desktop/CSC485/vjtest/Images/train/positive/'
negPath = 'C:/Users/sagar/Desktop/CSC485/vjtest/Images/train/negative/'
posImg = load_images(posPath)      # has and returns array of (name, read(image))
negImg = load_images(negPath)      # (name, image)


vj = VJ.ViolaJones(5)

vj.train(posImg, negImg)
