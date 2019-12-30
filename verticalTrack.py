import RPi.GPIO as GPIO
import time

'''
    GPIO BCM = 17 for horizontal movement
'''
def verOne(i):
    p.ChangeDutyCycle(i)
    time.sleep(.09)
    return (i+.1)
    
def verTwo(i):
    p.ChangeDutyCycle(i)
    time.sleep(.09)
    return (i-.1)
    
servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50)
i = 2.0
p.start(i)
p.ChangeDutyCycle(i)
verFlag = True     # tracks the movement of the Horizontal servo
try:
    while i<=7.0 and i>=1.0:
        if verFlag == True:
            i = verOne(i)
            if i > 6.0:
                verFlag = False
        else:
            i = verTwo(i)
            if i < 2.0:
                verFlag = True
except KeyboardInterrupt:
    p.stop()
    GPIO.cleanup()  
GPIO.cleanup()
