# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:54:00 2017

@author: IkerVazquezlopez
"""

import cv2
import numpy as np
from PIL import ImageGrab

x_min = 310
y_min = 150
x_max = 1100
y_max = 600

def captureScreen(x_min, y_min, x_max, y_max):
    printscreen_pil = ImageGrab.grab(bbox=(x_min, y_min, x_max, y_max))
    size = (printscreen_pil.size[1], printscreen_pil.size[0], 3)
    printscreen_numpy = np.reshape(np.array(printscreen_pil.getdata(), dtype='uint8'), size)
    return printscreen_numpy

def equal_color(p0, p1):
    if p0[0] == p1[0] and p0[1] == p1[1] and p0[2] == p1[2]:
        return True
    else:
        return False
    
def get_activated_pixels(flow, thresh=10):
    activated = np.zeros((flow.shape[0],flow.shape[1]), dtype=np.uint8)
    activated[flow[...,2] > thresh] = 255
    return activated

def compute_flow(prev, next):
    hsv = np.zeros((prev.shape[0], prev.shape[1], 3), dtype=np.float32)
    hsv[...,1] = 255
    flow = cv2.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 200 , 5, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    return hsv

def CCL(bin_flow, connectivity):
    ret, labels = cv2.connectedComponents(bin_flow, connectivity)
    return labels

def compute_bounding_boxes(img): # img is a binary image
    x_min = img.shape[1]; x_max = 0;
    y_min = img.shape[0]; y_max = 0;
    for x in range(0, img.shape[1]):
        for y in range(0, img.shape[0]):
            if img[y,x ] == 1:
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y 
    return x_min, y_min, x_max, y_max
        


cap = cv2.VideoCapture("test_video.mp4")
frame_count = int( cap.get(cv2.CAP_PROP_FRAME_COUNT) ) - 100
frame_heigth = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )

print( "Frame count: " + str(frame_count))
print( "Frame heigth: " + str(frame_heigth))
print( "Frame width: " + str(frame_width))

# Load the video
video = []
for i in range(0,frame_count):
    _, frame = cap.read()
    video.append(frame)

    
# Compute optical flow
flow = []
for i in range(1,frame_count):
    prev = cv2.cvtColor(video[i-1], cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
    f = compute_flow(prev, curr)
    flow.append(f)



## Detect the when a pixel changes its color for the last time (not worth)
#foreground = [[[0] for i in range(frame_width)] for j in range(frame_heigth)]
#for j in range(0, frame_width):
#    for i in range(0, frame_heigth):
#        arr = []
#        for f in range(1, frame_count-1):
#            prev_pixel = video[f-1][i,j]
#            curr_pixel = video[f][i,j]
#            if np.isnan(flow[f][i,j][2]): continue
#            #print(str(flow[f][i,j][1]) + ' // ' + str(flow[f][i,j][2]))
#            if not equal_color(prev_pixel, curr_pixel) and flow[f][i,j][2] > 0.5:
#                foreground[i][j].append(f)


# Create data in tuples for each frame
data = []
for f in range(0, frame_count-1):
    activated = get_activated_pixels(flow[f], 10)
    data.append(
                (activated,
                 flow[f],
                 CCL(activated, 8)
                )
               )
    

# Get the bbox for each object (connected flow) in the frame
for f in range(0, frame_count-1):
    blobs = []
    bboxes_points = []
    bboxes_imgs = []
    ccl = data[f][2]
    for i in range(1, np.max(ccl)+1):
            m = np.uint8(ccl == i)
            canvas = np.zeros((frame_heigth,frame_width), dtype=np.uint8)
            xmin, ymin, xmax, ymax = compute_bounding_boxes(m)
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
            blobs.append(np.uint8(m)*255)
            bboxes_imgs.append(canvas)
            bboxes_points.append((xmin, ymin, xmax, ymax))
    data[f] = (data[f][0], data[f][1], data[f][2], bboxes_points, bboxes_imgs, blobs)
    
for f in range(0,frame_count-1):
    cv2.imshow('bbox', data[f][4][0])
    cv2.imshow('activated', data[f][0])
    cv2.waitKey(50)
cv2.destroyAllWindows()
#cv2.imshow("foreground", foreground)
#cv2.waitKey(0)
#cv2.imwrite('foreground.png', foreground)


cap.release()
cv2.destroyAllWindows()








