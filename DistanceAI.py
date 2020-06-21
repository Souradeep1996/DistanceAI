#import torch, torchvision
#print(torch.__version__, torch.cuda.is_available())

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

#import miscellaneous
import time
import os
import datetime
import pandas as pd
import os.path
from os import path
import store

store.df.to_csv("Violations_DIP.csv")

cfg = get_cfg()
cfg.MODEL.DEVICE="cpu"
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
predictor = DefaultPredictor(cfg)

#define a function which returns the bottom center of every bbox
def mid_point(img,person,idx):
    #get the coordinates
    x1,y1,x2,y2 = person[idx]
    _ = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
  
    #compute bottom center of bbox
    x_mid = int((x1+x2)/2)
    y_mid = int(y2)
    mid   = (x_mid,y_mid)
  
    _ = cv2.circle(img, mid, 5, (255, 0, 0), -1)
    cv2.putText(img, str(idx), mid, cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2, cv2.LINE_AA)
  
    return mid

from scipy.spatial import distance
def compute_distance(midpoints,num):
    dist = np.zeros((num,num))
    for i in range(num):
        for j in range(i+1,num):
              if i!=j:
                dst = distance.euclidean(midpoints[i], midpoints[j])
                dist[i][j]=dst
    return dist

def change_2_red(img,person,p1,p2):
    risky = np.unique(p1+p2)
    for i in risky:
        x1,y1,x2,y2 = person[i]
        _ = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  
    return img

def find_closest(dist,num,thresh):
    p1 =[]
    p2 =[]
    d =[]
    for i in range(num):
        for j in range(i,num):
            if( (i!=j) & (dist[i][j]<=thresh)):
                p1.append(i)
                p2.append(j)
                d.append(dist[i][j])
    return p1,p2,d

def find_closest_people(date, name, img, thresh, location):
    #img = cv2.imread(name)
    outputs = predictor(img)
    classes=outputs['instances'].pred_classes.cpu().numpy()
    bbox=outputs['instances'].pred_boxes.tensor.cpu().numpy()
    ind = np.where(classes==0)[0]
    person=bbox[ind]
    midpoints = [mid_point(img,person,i) for i in range(len(person))]
    num = len(midpoints)
    dist= compute_distance(midpoints,num)   
    p1,p2,d=find_closest(dist,num,thresh)
    if p1 != []:
        img = change_2_red(img,person,p1,p2)
        cv2.imwrite(name, img)
        store.df.at[date, location] += len(p1)+len(p2)
        os.remove("Violations_DIP.csv")
        store.df.to_csv("Violations_DIP.csv")
    return 0

#specify path to video
video_1 = "rtsp://admin:admin123@192.168.1.112:554/cam/realmonitor?channel=1&subtype=0"
video_2 = "rtsp://admin:admin123@192.168.1.181:554/cam/realmonitor?channel=1&subtype=0"

#capture video
#cap_1 = cv2.VideoCapture(video_1)
#cap_2 = cv2.VideoCapture(video_2)
thresh = 100
cnt=0
while (1):
    now = datetime.datetime.now()

    # Directory 
    date_time = now.strftime("%d-%m-%Y/")
  
    # Parent Directory path 
    parent_dir_1 = "DIP Gate"
    parent_dir_2 = "Main Gate"
  
    # Path 
    path_1 = os.path.join(parent_dir_1+'/', date_time)
    path_2 = os.path.join(parent_dir_2+'/', date_time)
    #print(path_1)
    #print(path_2)
  
    # Create the directory 
    if not os.path.exists(path_1):
        os.makedirs(path_1)
    if not os.path.exists(path_2):
        os.makedirs(path_2)
    
    #capture video
    cap_1 = cv2.VideoCapture(video_1)
    cap_2 = cv2.VideoCapture(video_2)
    
    #get timestamp
    ts = datetime.datetime.now()
    ts = ts.strftime('%d-%m-%Y %H-%M-%S')
    
    # Capture frame-by-frame
    ret_1, frame_1 = cap_1.read()
    ret_2, frame_2 = cap_2.read()
    #print (ret_1, ret_2)
     
    if (ret_1 == True):
        #save each frame to folder        
        #cv2.imwrite(path_1+ts+'.png', frame_1)        
        _ = [find_closest_people(date_time, path_1+ts+'.png',frame_1, thresh, parent_dir_1)]
     
    if (ret_2 == True):
    	#cv2.imwrite(path_2+ts+'.png', frame_2)
    	_ = [find_closest_people(date_time, path_2+ts+'.png',frame_2, thresh, parent_dir_2)]
    	
    if (cnt>=24*60*6):
        break
    
    time.sleep(10)
    
