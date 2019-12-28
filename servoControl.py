import RPi.GPIO as GPIO
import time

servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)
GPIO.setwarnings(False)

#p = GPIO.PWM(servoPIN, 50)
i = 0
p = start(i)
p.ChangeDutyCycle(i)

GPIO.cleanup()
