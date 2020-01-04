import numpy as np
import cv2
import RPi.GPIO as GPIO
import time

def moveServoPos(i, p):
    p.ChangeDutyCycle(i)
    #time.sleep(.09)
    return (i+.5)
    
def moveServoNeg(i, p):
    p.ChangeDutyCycle(i)
    #time.sleep(.09)
    return (i-.5)
    
def defaultServoMove(i, horzFlag): 
	'''
		Moves the servo horizontally
	'''   
	
	servoPIN = 22
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)
	GPIO.setup(servoPIN, GPIO.OUT)

	p = GPIO.PWM(servoPIN, 50)
	p.start(i)
	p.ChangeDutyCycle(i)
	try:
		while i<=13.0 and i>=1.0:
			if horzFlag == True:
				i = moveServoPos(i, p)
				if i > 12.0:
					horzFlag = False
			else:
				i = moveServoNeg(i, p)
				if i < 2.0:
					horzFlag = True
			
			# check vidoe for face
			ret, frame = vid.read()
			# TODO: show the video even though face is not in the frame
			# cv2.imshow('frame', frame)
			faces = faceCascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5, minSize=(20,20))
			if(len(faces) != 0):
				return i, horzFlag
	except KeyboardInterrupt:
		p.stop()
		GPIO.cleanup()  
	GPIO.cleanup()

def positionServo(x, y, w, h, i):
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)
	if (x < 0.1*fLen or x+w > 0.9*fLen):
		GPIO.setup(22, GPIO.OUT)
		p = GPIO.PWM(22, 50)
		p.start(i)
		if (x < 0.1*fLen):				# move left, neg movement, r
			return moveServoPos(i, p)
			# print("x<")
		else:							# move right, pos movement, l
			return moveServoNeg(i, p)
			# print("x+w>")
	if (y < 0.1*fWid or y+h > 0.9*fWid):
		GPIO.setup(17, GPIO.OUT)
		p = GPIO.PWM(17, 50)
		p.start(i)
		if (y < 0.1*fWid):				# move down, pos movement, u
			return moveServoPos(i, p)
			# print("y<")
		else:							# move up, neg movement, d
			return moveServoNeg(i, p)
			# print("y+h>")

'''
    GPIO BCM = 17 for vertical movement
    GPIO BCM = 22 for horizontal movement
'''
i = 2.0 	# initial servo position
horzFlag = True 	# initial servo movement direction

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)													#initializing the video capture
global fLen 
global fWid 
fLen = vid.get(3)
fWid = vid.get(4)
	
while(True):
	ret, frame = vid.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 						#only for grayscale image
	
	#face detection
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20,20))
	
	if len(faces) == 0: 									# no face is detected
		i, horzFlag = defaultServoMove(i, horzFlag)	
	else:
		#rectangular box if a face is detected
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
			roi_color = frame[y:y+h, x:x+w]
			if( (x < 0.1*fLen) or (y < 0.1*fWid) or (x+w > 0.9*fLen) or (y+h > 0.9*fWid)):
				if i<=12.0 and i>=2.0:
					i = positionServo(x, y, w, h, i)
	
	cv2.imshow('frame', cv2.flip(frame, 1))
	if cv2.waitKey(1) & 0xFF == ord('q'):									# quit with q
		break
	
vid.release()
cv2.destoryAllWindows()
