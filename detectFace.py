import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(1)													# initializing the video capture

while(True):
	ret, frame = vid.read()
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 						# only for grayscale image
	
	# face detection
	faces = faceCascade.detectMultiScale(frame, scaleFactor=1.2,minNeighbors=5, minSize=(20,20))
	
	# rectangular box if a face is detected
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
		roi_color = frame[y:y+h, x:x+w]
	
	# show the image and quit option
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):									# quit with q
		break
	
vid.release()
cv2.destroyAllWindows()
