import RPi.GPIO as GPIO
import time

'''
    GPIO BCM = 17 for vertical movement
    GPIO BCM = 22 for horizontal movement
'''
def horOne(i):
    p.ChangeDutyCycle(i)
    time.sleep(.09)
    return (i+.1)
    
def horTwo(i):
    p.ChangeDutyCycle(i)
    time.sleep(.09)
    return (i-.1)
    
servoPIN = 22
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50)
i = 2.0
p.start(i)
p.ChangeDutyCycle(i)
horzFlag = True     # tracks the movement of the Horizontal servo
try:
    while i<=13.0 and i>=1.0:
        if horzFlag == True:
            i = horOne(i)
            if i > 12.0:
                horzFlag = False
        else:
            i = horTwo(i)
            if i < 2.0:
                horzFlag = True
except KeyboardInterrupt:
    p.stop()
    GPIO.cleanup()  
GPIO.cleanup()
