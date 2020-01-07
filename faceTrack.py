import numpy as np
import cv2
import RPi.GPIO as GPIO
import time

def moveServoPos(i, p):
    '''
        moves the servo positively to the right or down
        args:
            i:  duty cycle
            p:  pwm
        return:
            i:  duty cycle
    '''
    p.ChangeDutyCycle(i)
    # time.sleep(.5)
    # cv2.waitKey(2000)
    return (i+.5)
    
def moveServoNeg(i, p):
    '''
        moves the servo negatively to the left or up
        args:
            i:  duty cycle
            p:  pwm
        return:
            i:  duty cycle
    '''
    p.ChangeDutyCycle(i)
    # time.sleep(.5)
    # cv2.waitKey(2000)
    return (i-.5)

def verPosition(iVer):
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    servoPIN = 17
    GPIO.setup(servoPIN, GPIO.OUT)
    p = GPIO.PWM(servoPIN, 50)
    p.start(iVer)
    time.sleep(1)			# add time delay to let servo process the code

def position(i, iVer):
    verPosition(iVer)

    servoPIN = 22
    GPIO.setup(servoPIN, GPIO.OUT)
    q = GPIO.PWM(servoPIN, 50)
    q.start(i)
    print("Initial Position Set...")
    time.sleep(1)
    
    GPIO.cleanup()
    
def mapServoFace(x, y, w, h):
    '''
        maps the servo position to face position
        executes only if the face goes out of 90% of the frame boundary
        args:
            x, y:   face top left cordinates
            w, h:   face width, height
            i:      horizontal duty cycle value
        return:
			i:		horizontal duty cycle value to keep track of horizontal movement
    '''
    global i
    global iVer
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    if (x < 0.1*fLen or x+w > 0.9*fLen):    # horizontal movement
        GPIO.setup(22, GPIO.OUT)
        p = GPIO.PWM(22, 50)
        p.start(i)
        if (x < 0.1*fLen):              # move left, neg movement, r
            print("x<")
            if i <= 2.0:
                # print("Horizontal left limit")
                i = 2.0
            else:
                i = moveServoPos(i, p)
        else:                           # move right, pos movement, l
            print("x+w>")
            if i >= 12.0:
                # print("Horizontal right limit")
                i = 12.0
            else:
                i = moveServoNeg(i, p)
    if (y < 0.1*fWid or y+h > 0.9*fWid):    # vertical movement
        GPIO.setup(17, GPIO.OUT)
        p = GPIO.PWM(17, 50)
        p.start(iVer)
        if (y < 0.1*fWid):              # move down, pos movement, u
            print("y<")
            if iVer <= 1.5:
                # print("Vertical up limit")
                iVer = 1.5
            else:
                iVer = moveServoNeg(iVer, p)
        else:                           # move up, neg movement, d
            print("y+h>")
            if iVer >= 4.0:
                # print("Vertical down limit")
                iVer = 4.0
            else:
                iVer = moveServoPos(iVer, p)
    print("Face mapped in frame")
    cv2.waitKey(2000)
    GPIO.cleanup()
    
# global fLen 
# global fWid 

i = 10.0     # initial horizontal servo position
iVer = 1.5	# initial vertical servo position
horzFlag = True     # initial servo movement direction

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)                                                   #initializing the video capture
vid.read()
fLen = vid.get(3)
fWid = vid.get(4)

cv2.waitKey(2000)
position(i, iVer)

while(vid.isOpened()):
	ret, frame = vid.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                      #only for grayscale image
    
    #face detection
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20,20))
    
	for (x, y, w, h) in faces:
		print(x, y, w, h)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
		roi_color = frame[y:y+h, x:x+w]
		if( (x < 0.1*fLen) or (y < 0.1*fWid) or (x+w > 0.9*fLen) or (y+h > 0.9*fWid)):
			# if (i>=2.0 and i<=12.0) or (iVer >= 1.5 and iVer<= 4.0):
			mapServoFace(x, y, w, h)
			# cv2.waitKey(5000)
    
	cv2.imshow('frame', cv2.flip(frame, 1))
	if cv2.waitKey(1) & 0xFF == ord('q'):                                   # quit with q
		break
    
vid.release()
cv2.destoryAllWindows()
