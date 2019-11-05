import cv2
import os

def load_images(path):
    '''
    Loads the images in the provided path with jpg format
    :param path: directory path for images
    :return: array of images
    '''
    images = []
    for img in os.listdir(path):
        if img.endswith('.jpg'):
            images.append(img)
    return images

# image pre-processing; resize, gray-scale
imgPath = 'C:/Users/sagar/Desktop/VJ/Images/test/negative/'                       #image path for input
imgPathOut = 'C:/Users/sagar/Desktop/CSC485/objectDet/Images/test/negative/'         #image path for output
images = load_images(imgPath)                                               #array of images

if not os.path.exists(imgPathOut):
    os.makedirs(imgPathOut)
img_num = 1

for i in images:
    img = cv2.imread(imgPath+i, 0)
    resized_image = cv2.resize(img, (100,100))
    cv2.imwrite(imgPathOut+str(img_num)+'.jpg', resized_image)
    img_num += 1
