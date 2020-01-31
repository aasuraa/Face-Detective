import cv2
import os

path = "./output/"

def load(path):
    print("[INFO] loading images...")

    images = []
    for img in os.listdir(path):
        if img.endswith('.jpg') or img.endswith('.jpeg'):
            images.append((img, cv2.imread(path+img, 0)))
    return images

print(len(load(path)))

