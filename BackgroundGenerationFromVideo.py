# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:54:00 2017

@author: IkerVazquezlopez
"""

import cv2
import numpy as np
import pickle
from scipy import stats




#%% FUNCTION DEFINITION

## Video io functions  ----------------------------------------------------
def load_video(argument_filename):
    video = []
    cap = cv2.VideoCapture(argument_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    for _ in range(0, frame_count):
        _, frame = cap.read()
        video.append(frame)
    return video, frame_count, frame_heigth, frame_width

def write_video(filename, v):
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    isColor = False
    if len(v[0].shape) == 3:
        isColor = v[0].shape[2] == 3
    out = cv2.VideoWriter(filename, fourcc, 20.0, (v[0].shape[1],v[0].shape[0]), isColor)
    for f in range(0, len(v)):
        out.write(v[f])
    out.release()
    print("Video saved to: " + filename)



## Image processing fuctions  --------------------------------------------

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
            if img[y,x] == 1:
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y 
    return x_min, y_min, x_max, y_max


#%% LOAD THE VIDEO

video, frame_count, _, _ = load_video('test_video.mp4')

#%% OPTICAL FLOW COMPUTATION

# Compute optical flow
flow = []
for i in range(1,frame_count):
    prev = cv2.cvtColor(video[i-1], cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
    f = compute_flow(prev, curr)
    flow.append(f)

write_video("opticalFlow.mp4", np.uint8(flow))

flow, _, _, _ = load_video("opticalFlow.mp4")


#%% GENERATE DATA

# Create data in tuples for each frame
ccl = []
for f in range(0, frame_count-1):
    act = get_activated_pixels(flow[f], 20)
    ccl.append(np.uint8(CCL(act, 8)))
    



# Get the bbox for each object (connected flow) in the frame
bboxes = []
for f in range(0, frame_count-1):
    frame_bboxes = []
    print(str(f) + ' / ' + str(frame_count-1))
    for i in range(1, np.max(ccl[f])+1):
            m = np.uint8(ccl[f] == i)
            xmin, ymin, xmax, ymax = compute_bounding_boxes(m)
            frame_bboxes.append((ymin, xmin, ymax, xmax))
    bboxes.append(frame_bboxes)
    
# Load data if it is already computed
with open('bboxes.pickle', 'wb') as f:
    pickle.dump(bboxes, f)
with open('bboxes.pickle', 'rb') as f:
    bboxes = pickle.load(f)
    



#%% GENERATE BACKGROUND
height, width, _ = video[0].shape
background = np.zeros(video[0].shape, dtype=np.float32)


for i in range(0,height):
    print(str(i) + ' / ' + str(height))
    for j in range(0, width):
        for f in range(0, len(bboxes)):
            outside_bbox = True
            pixels = []
            for box in bboxes[f]:
                y_min, x_min, y_max, x_max = box
                if (i < y_min or i > y_max) and (j < x_min or j > x_max):
                    pixels.append(video[f][i,j])
            if len(pixels) == 0: continue
            background[i][j] = np.array(list(stats.mode(pixels)[0][0]))



#%% SHOW RESULTS

cv2.imwrite('background.png', np.uint8(background))
