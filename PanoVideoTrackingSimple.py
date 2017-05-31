# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:54:00 2017

@author: IkerVazquezlopez
"""

import cv2
import numpy as np
from scipy import stats
from PIL import ImageGrab



def captureScreen(x_min, y_min, x_max, y_max):
    printscreen_pil = ImageGrab.grab(bbox=(x_min, y_min, x_max, y_max))
    size = (printscreen_pil.size[1], printscreen_pil.size[0], 3)
    printscreen_numpy = np.reshape(np.array(printscreen_pil.getdata(), dtype='uint8'), size)
    return printscreen_numpy

x_min = 310
y_min = 150
x_max = 1100
y_max = 600

cap = cv2.VideoCapture("test_video.mp4")
ret, frame1 = cap.read()
#frame1 = captureScreen(x_min,y_min,x_max,y_max)
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

ret, frame2 = cap.read()
#frame2 = captureScreen(x_min,y_min,x_max,y_max)
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)
hsv[...,1] = 255



def compute_flow(prev, next):
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 200 , 5, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR), ang, mag

def get_activated_pixels(flow):
    activated = np.zeros((flow.shape[0],flow.shape[1]), dtype=np.uint8)
    activated[flow[...,1] > 10] = 255
    activated[flow[...,2] > 10] = 255
    return activated


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

def CCL(bin_flow, connectivity):
    ret, labels = cv2.connectedComponents(bin_flow, connectivity)
    return labels





flow, flow_ang, flow_mag = compute_flow(prvs, next)
binary = get_activated_pixels(flow)
prev_ccl = CCL(binary, 8)


prvs = next

video = []
while(1):
    # Capture the image and compute the flow
    ret, frame2 = cap.read()
    if ret == None or ret == -1:
        break
    #frame2 = captureScreen(x_min,y_min,x_max,y_max)
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    
    flow, flow_ang, flow_mag = compute_flow(prvs, next)
    
#    cv2.imshow('0', flow[...,0].reshape((flow.shape[0],flow.shape[1])))
#    cv2.imshow('1', flow[...,1].reshape((flow.shape[0],flow.shape[1])))
#    cv2.imshow('2', flow[...,2].reshape((flow.shape[0],flow.shape[1])))
    
    result = cv2.addWeighted(cv2.cvtColor(next, cv2.COLOR_GRAY2BGR), 0.7, flow, 0.3, 0.3)
    cv2.imshow('output', result)
    cv2.moveWindow('output',-1000,100)
#    
    
    binary = get_activated_pixels(flow)
    ccl = CCL(binary, 8)
#    cv2.imshow('bin', binary)
    #cv2.imshow('ccl', np.uint8(ccl)*100)
#    print(np.max(ccl))
    
    
    d = np.zeros((frame1.shape[0], frame1.shape[1]), dtype=np.uint8)
    # Get the mask for each object
    for i in range(1, np.max(ccl)+1):
        m = np.uint8(ccl == i)
        xmin, ymin, xmax, ymax = compute_bounding_boxes(m)
        cv2.rectangle(m, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        d = cv2.add(d, m)
    
    d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
    video.append(cv2.add(result, d))
    
#    cv2.imshow('bbox', d)
#    cv2.moveWindow('bbox',-1800,100)
    
    k = cv2.waitKey(30) & 0xff
    if k == 32:
        break
    prvs = next
    
fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
print(len(video))
output = cv2.VideoWriter('result.mp4', fourcc, 30.0, (video[0].shape[0], video[0].shape[1]), True)
for f in video:
    output.write(f)
output.release()
cap.release()
cv2.destroyAllWindows()










