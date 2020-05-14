import RPi.GPIO as GPIO
import time

class ServoControl:
	def __init__(self, l, w):
		self.fLen = l
		self.fWid = w
		self.i = 10.0                                                   # initial horizontal servo position
		self.iVer = 1.5                                                 # initial vertical servo position
		self.horzFlag = True                                            # initial servo movement direction

		self.position(self.i, self.iVer)                                # initial position

	def horizontalScan(self):
		"""
		Scan for a face moving the servos horizontally left to right. Stops when a face is detected.
		"""
		if self.iVer != 1.5:
			self.iVer = 1.5
			self.verPosition(self.iVer)
		servoPIN = 22
		GPIO.setmode(GPIO.BCM)
		GPIO.setwarnings(False)
		GPIO.setup(servoPIN, GPIO.OUT)
		p = GPIO.PWM(servoPIN, 50)
		p.start(self.i)
		
		if self.horzFlag == True:
			self.i += .2
			p.ChangeDutyCycle(self.i)
			time.sleep(.1)
			if self.i > 11.0:
				self.horzFlag = False
		else:
			self.i -= .2
			p.ChangeDutyCycle(self.i)
			time.sleep(.1)
			if self.i < 2.0:
				self.horzFlag = True

		p.stop()
		GPIO.cleanup()
	
	    
	def moveServoPos(self, i, p):
		'''
			moves the servo positively to the right or down
			args:
				i:  duty cycle
				p:  pwm
			return:
				i:  duty cycle
		'''
		p.ChangeDutyCycle(i)
		return (i+.1)
		
	def moveServoNeg(self, i, p):
		'''
			moves the servo negatively to the left or up
			args:
				i:  duty cycle
				p:  pwm
			return:
				i:  duty cycle
		'''
		p.ChangeDutyCycle(i)
		return (i-.1)

	def verPosition(self, iVer):
		'''
			Sets vertical servo positon 
		'''
		GPIO.setmode(GPIO.BCM)
		GPIO.setwarnings(False)

		servoPIN = 17
		GPIO.setup(servoPIN, GPIO.OUT)
		p = GPIO.PWM(servoPIN, 50)
		p.start(iVer)
		time.sleep(1)                                                   # add time delay to let servo process the code
		p.stop()

	def position(self, i, iVer):
		'''
			Initial position of the servo to start
			args:
				i, iVer:    position servo at i and iVer; horizontal and vertical duty cycle
		'''
		self.verPosition(iVer)

		servoPIN = 22
		GPIO.setup(servoPIN, GPIO.OUT)
		q = GPIO.PWM(servoPIN, 50)
		q.start(i)
		print("Initial Position Set...")
		time.sleep(1)
		q.stop()
		
		GPIO.cleanup()
		
	def mapServoFace(self, x, y, w, h):
		'''
			maps the servo position to face position
			executes only if the face goes out of 90% of the frame boundary
			args:
				x, y:   face top left cordinates
				w, h:   face width, height
				i:      horizontal duty cycle value
			return:
				i:      horizontal duty cycle value to keep track of horizontal movement
		'''
		
		GPIO.setmode(GPIO.BCM)
		GPIO.setwarnings(False)
		if (x < 0.1*self.fLen or x+w > 0.9*self.fLen):                  # horizontal movement
			GPIO.setup(22, GPIO.OUT)
			p = GPIO.PWM(22, 50)
			p.start(self.i)
			if (x < 0.1*self.fLen):                                     # move left, neg movement, r
				#print("x<")
				if self.i > 12.0:
					self.i = 12.0
				else:
					self.i = self.moveServoPos(self.i, p)
					time.sleep(.1)
			else:                                                       # move right, pos movement, l
				#print("x+w>")
				if self.i < 2.0:
					self.i = 2.0
				else:
					self.i = self.moveServoNeg(self.i, p)
					time.sleep(.1)
			p.stop()
		if (y < 0.1*self.fWid or y+h > 0.9*self.fWid):                  # vertical movement
			GPIO.setup(17, GPIO.OUT)
			p = GPIO.PWM(17, 50)
			p.start(self.iVer)
			if (y < 0.1*self.fWid):                                     # move down, pos movement, u
				#print("y<")
				if self.iVer < 1.5:
					self.iVer = 1.5
				else:
					self.iVer = self.moveServoNeg(self.iVer, p)
					time.sleep(.1)
			else:                                                       # move up, neg movement, d
				#print("y+h>")
				if self.iVer > 4.0:
					self.iVer = 4.0
				else:
					self.iVer = self.moveServoPos(self.iVer, p)
					time.sleep(.1)
			p.stop()
		#print("Face mapped in frame")
		GPIO.cleanup()
