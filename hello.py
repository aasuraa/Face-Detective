import numpy as np
import cv2

vid = cv2.VideoCapture(0)

while(True):
	ret, frame = vid.read()
	
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 						#only for grayscale image
	
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):									#quit with q
		break
	
vid.release()
cv2.destoryAllWindows()
