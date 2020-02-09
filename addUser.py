import cv2
import os

cam = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

# For each person, enter user's name
uname = input('\nenter user name and press <return> ==>  ')  # user name, string
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# TODO: make false safe by using lowercase for any name input

# Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 						# only for grayscale image

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        # Save the captured image into the datasets folder
        # also create new user folder if one already doesn't exist
        path = "dataset/"+uname+"/"
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + uname + '.' + str(count) + ".jpg", cv2.resize(gray[y:y+h, x:x+w], (30, 30)))
        cv2.imshow('image', img)    # showing video output

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 100: # Take 100 face sample and stop video
         break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()