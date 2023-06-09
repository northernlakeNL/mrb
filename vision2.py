from collections import deque
import numpy as np
import argparse
import imutils
import cv2 as cv
import time
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to video")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")

args = vars(ap.parse_args())

lowBound = (0,111, 209)
lowBound = (0, 182, 162)

upBound = (255,205,255)
upBound = (255,255,255)
pts = deque(maxlen=args["buffer"])

def fps():
    test = camera.get(cv.CAP_PROP_FPS)
    print("capturing{0} frames per second".format(num_frames))
    start = time.time()

    num_frames = 120
    for i in range(0, num_frames):
        ret, frame = camera.read()

    end = time.time()

    seconds = end - start
    print("Time taken: {0} seconds".format(seconds))

    fps = num_frames / seconds;
    print("Estimated fps is {0}".format(fps))

if not args.get("video", False):
    camera = cv.VideoCapture()
else:
    camera = cv.VideoCapture(args["video"])

i = 0
j = 0
start = time.time()
num_frames_2 = 1200
x_arr = []
y_arr = []

while True:
    (grabbed, frame) = camera.read()

    j += 1

    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=400)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    mask = cv.inRange(hsv, lowBound, upBound)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    cnts= cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv.contourArea)
        ((x,y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

        if radius > 10:
            cv.circle(frame (int(x), int(y)), int(radius), (0,255,255),2)
            cv.circle(frame, center, 5, (0,0,255), -1)

    pts.appendleft(center)

    for i in range(1,len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"]/float(i+1)) *2.5)
        cv.line(frame, pts[i-1], pts[i],(0,0,255), thickness)

    cv.imshow("Frame", frame)

    if center is None:
        str_x, str_y = "x{:03d}".format(center[0]), "y{:03d}".format(center[i])

    x_arr.append(int(str_x[1:]))
    y_arr.append(int(str_y[1:]))

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    end = time.time()
    seconds = end - start
    print("Simulated fps: ", i /seconds)

    x_arr_np = np.array(x_arr)
    x_arr_np = x_arr_np / (i/seconds)
    y_arr_np = np.array(y_arr)
    y_arr_np = y_arr_np / (i/seconds)

    fps = int(i /seconds)

    plt.figure(1)

    plt.subplot(211)

    plt.title(" vert position")

    plt.ylabel("position")
    plt.plot(range((len(x_arr))), x_arr, label="actual")

    fs= 100
    f = 0.8
    x = np.arrange(len(x_arr))

    y = [ 40*np.sin(2*np.pi*f*(i/fs)+2)+273 - x_arr[i] for i in x]

    plt.legend()

    plt.subplot(212)
plt.title('Horizontal Position')


plt.xlabel('Frame Number @ 30FPS') 
plt.ylabel('Position')
# plt.axhline(y=238, linewidth=1, color='r', ls='--', label="reference")


fs = 100
 # sample rate 
f = 0.78 # the frequency of the signal
x = np.arange(len(x_arr))
y = [ 40*np.sin(2*np.pi*f * (i/fs) + np.pi*0.32) + 220 - y_arr[i] for i in x]
plt.plot(range(len(x_arr)), y_arr, label="actual")

plt.legend()

plt.show()

camera.release()
cv.destroyAllWindows()