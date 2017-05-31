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
frame1 = captureScreen(x_min,y_min,x_max,y_max)
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

ret, frame2 = cap.read()
frame2 = captureScreen(x_min,y_min,x_max,y_max)
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
    activated[flow[...,0] > 10] = 255
    return activated


def compute_bounding_boxes(flow):
    flow = cv2.cvtColor(flow, cv2.COLOR_HSV2BGR)
    flow = cv2.cvtColor(flow, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(flow, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1)
    connectivity = 8
    ret, labels = cv2.connectedComponents(binary, connectivity)
    cv2.imshow('ccl',labels*2000)

def CCL(bin_flow, connectivity):
    ret, labels = cv2.connectedComponents(bin_flow, connectivity)
    return labels

def get_first_nonzero_pixel_coords(image):
    non_zero = np.nonzero(image)
    return non_zero[0][0], non_zero[1][0]



MAX_OBJECTS = 255
objects = []
for i in range(0,MAX_OBJECTS):
    objects.append(np.zeros((frame1.shape[0], frame1.shape[1]), dtype=np.uint8))
print(len(objects)) 

indices = np.zeros(MAX_OBJECTS, dtype=np.uint8 )


def find_new_idx():
    for idx in range(1, len(indices)):
        if indices[idx] == 0:
            indices[idx] = 1
            return idx



flow, flow_ang, flow_mag = compute_flow(prvs, next)
binary = get_activated_pixels(flow)
prev_ccl = CCL(binary, 8)


prvs = next

while(1):
    # Capture the image and compute the flow
    ret, frame2 = cap.read()
    frame2 = captureScreen(x_min,y_min,x_max,y_max)
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    
    flow, flow_ang, flow_mag = compute_flow(prvs, next)
    
    
    result = cv2.addWeighted(cv2.cvtColor(next, cv2.COLOR_GRAY2BGR), 0.7, flow, 0.3, 0.3)
    cv2.imshow('output', result)
    cv2.moveWindow('output',-1000,100)
    
    
    binary = get_activated_pixels(flow)
    ccl = CCL(binary, 8)
    #cv2.imshow('bin', binary)
    
    tracked_ids = np.zeros(MAX_OBJECTS, dtype=np.uint8 )
    # Get the mask for each object
    for i in range(1, np.max(ccl)): 
        m = np.uint8(ccl == i)
        
        blob = cv2.bitwise_and(prev_ccl, prev_ccl, None, m.reshape(binary.shape))
        b = blob[np.nonzero(blob)]
        mode = stats.mode(b, axis=None)
        if len(mode[0]) == 0:
            idx = find_new_idx()
        else:
            idx = mode[0][0]
            print(mode[0][0])
        tracked_ids[idx] = 1
        y, x = get_first_nonzero_pixel_coords(m) #OpenCV uses transposed image
        
        flood_mask = np.zeros((m.shape[0]+2, m.shape[1]+2), dtype=np.uint8)
        print(flood_mask.shape)
        flood_mask[0:m.shape[0], 0:m.shape[1]] = m
        ret, m, _, _ = cv2.floodFill(m, flood_mask, (x,y), int(idx)) # with inverted image
        cv2.imshow('m', m*122)
        objects[idx] = m.reshape(binary.shape)
    
    prev_ccl = np.zeros((frame1.shape[0], frame1.shape[1]), dtype=np.uint8)
    for i in range(1,len(objects)):
        if tracked_ids[i] == 0:
            objects[i] = np.zeros((frame1.shape[0], frame1.shape[1]), dtype=np.uint8)
            indices[i] = 0
        else:
            prev_ccl = cv2.add(prev_ccl, objects[i]*i)

    
    cv2.imshow('prev_ccl', prev_ccl)
    
    k = cv2.waitKey(30) & 0xff
    if k == 32:
        break
    prvs = next

cap.release()
cv2.destroyAllWindows()










