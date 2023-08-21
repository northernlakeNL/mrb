from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from gpiozero import Servo, Button
from gpiozero.pins.pigpio import PiGPIOFactory
import math

## @file
# This script performs object tracking using computer vision techniques and controls servo motors based on the tracked object's position.

## The lines `servo_pin_1 = 22` and `servo_pin_2 = 27` are assigning the GPIO pin numbers to the
# variables `servo_pin_1` and `servo_pin_2`, respectively. These pin numbers are used to initialize
# the servo motors later in the code.
servo_pin_1 = 22
servo_pin_2 = 27

Buttonxl = Button(23)
Buttonxr = Button(24)
Buttonyl = Button(25)
Buttonyr = Button(16)

setpointX = 379
setpointY = 222

factory = PiGPIOFactory()

servo_1 = Servo(22, pin_factory=factory)
servo_2 = Servo(27, pin_factory=factory)

integralx = 0
prev_errorx = 0

integraly = 0
prev_errory = 0

alpha_x = 0.2
alpha_y = 0.2

center_smoothed_y = None
center_smoothed_x = None

## The function calculates the exponential moving average (EMA) by combining the current value with the
# previous EMA using a specified alpha value.
#
# @param actual The actual value at the current time step.
# @param prev_ema The previous exponential moving average (EMA) value.
# @param alpha The alpha parameter is a smoothing factor that determines the weight given to the
# current value compared to the previous exponential moving average (EMA) value.
#
# @return The exponential moving average (EMA) value.
def ema_filter(actual, prev_ema, alpha):
    return alpha * actual + (1 - alpha) * prev_ema

## The function implements a PID controller for controlling the position of a system in the x-axis.
#
# @param actual The actual value of the system or process variable that you are trying to control.
# @param setpoint The setpoint is the desired value or target value that you want the actual value to
# reach or maintain.
#
# @return The output of the PID control algorithm.
def pid_controlx(actual, setpoint):
    global integralx
    global prev_errorx
    error = setpoint - actual
    integralx += error
    derivative = error - prev_errorx
    output = 0.04 * error + 0.0001 * integralx + 0.006 * derivative/0.0053 
    prev_errorx = error
    return output

## The function implements a PID controller to calculate the output based on the difference between the
# actual value and the setpoint.
#
# @param actual The actual value of the system or process variable that you are trying to control.
# @param setpoint The setpoint is the desired value or target value that you want the actual value to
# reach or maintain.
#
# @return The output value, which is the result of the PID control calculation.
def pid_controly(actual, setpoint):
    global integraly
    global prev_errory
    error = setpoint - actual
    integraly += error
    derivative = error - prev_errory
    output = 0.04 * error + 0.0001 * integraly + 0.006 * derivative/0.0053 
    prev_errory = error
    return output

## The code you provided is an infinite loop that performs object tracking using computer vision
# techniques and controls servo motors based on the tracked object's position.
while True:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to file")
    ap.add_argument("-b", "--buffer", type=int , default=64, help="max buffer size")
    args = vars(ap.parse_args())
    orangeLower = (0, 100, 100)
    orangeUpper = (20, 255, 255)

    if not args.get("video", False):
        vs = VideoStream(src=0).start()

    else:
        vs = VideoStream(src=args["video"])

    while True:
        if setpointX > 200 and setpointX < 525: 
            if Buttonxl.is_pressed:
                setpointX -= 1
            elif Buttonxr.is_pressed:
                setpointX += 1
                
        if setpointY > 60 and setpointY < 375:
            if Buttonyl.is_pressed:
                setpointY -= 1
            elif Buttonyr.is_pressed:
                setpointY += 1
  
        frame = vs.read()

        if frame is None:
            break

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        width, height = frame.shape[:2]
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, orangeLower, orangeUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c) 
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if center_smoothed_x is None or center_smoothed_y is None:
            center_smoothed_x = center[0]
            center_smoothed_y = center[1]
        else:
            center_smoothed_x = ema_filter(center[0], center_smoothed_x, alpha_x)
            center_smoothed_y = ema_filter(center[1], center_smoothed_y, alpha_y)

        print(center[0], center[1])

        anglex = pid_controlx(center[0], setpointX)
        angley = pid_controly(center[1], setpointY)

        if anglex > 10:
            anglex = 10
        elif anglex < -10:
            anglex = -10

        if angley > 10:
            angley = 10
        elif angley < -10:
            angley = -10

        servo_1.value = math.sin(math.radians(-3))
        servo_2.value = math.sin(math.radians(20))