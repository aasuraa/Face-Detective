import numpy as np
import cv2
import faceDet.ViolaJones.Integral as ii
import faceDet.ViolaJones.Cascade as cascade

vid = cv2.VideoCapture(1)

clfs = cascade.CascadeClassifier.load("cTrain")

while(True):
	ret, frame = vid.read()			# default shape is (480, 640)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 						# only for grayscale image

	w, h = gray.shape
	print(gray.shape)
	# gray = cv2.resize(gray, (30, 30))
	# frameInt = ii.integral_image(gray)
	# if clfs.classify(frameInt) == 1:
	# 	print("Face Detected")

	for x in range(0, w-30, 100):
		for y in range(0, h-30, 100):
			subFrame = gray[x:x + 30, y:y + 30] 			# selecting sub frame
			subFrameInt = ii.integral_image(subFrame)
			if clfs.classify(subFrameInt) == 1:
				print("Face Detected")
				print("x is %d, y is %d" % (x, y))
				cv2.rectangle(frame, (x, y), (x + 30, y + 30), (0, 0, 255), 2)
				roi_color = frame[y:y + 30, x:x + 30]
			# else:
			# 	print("False")

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):									#quit with q
		break

	
vid.release()
cv2.destroyAllWindows()
