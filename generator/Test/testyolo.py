# code inspired by:
# https://github.com/ivder/YoloBBoxChecker/blob/master/main.py

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


def convert(size,x,y,w,h):
    box = np.zeros(4)
    dw = 1./size[0]
    dh = 1./size[1]
    x = x/dw
    w = w/dw
    y = y/dh
    h = h/dh
    box[0] = x-(w/2.0)
    box[1] = x+(w/2.0)
    box[2] = y-(h/2.0)
    box[3] = y+(h/2.0)

    return (box)
    

""" Configure Paths"""   
dir_path = ''


all_files = os.listdir(dir_path + 'labels')
text_files = [file for file in all_files if file.endswith(".txt")]
for text_file in text_files:
    img_path = dir_path + 'images\\' + text_file.rstrip(".txt") + ".jpg"
    img = cv2.imread(img_path)
    txt_file = open(dir_path + 'labels\\' + text_file, "r")
   

    """ Convert YOLO format to get xmin,ymin,xmax,ymax """ 
    lines = txt_file.read().splitlines()  
    for idx, line in enumerate(lines):
        value = line.split()
        x=y=w=h=cls= None
        cls = value[0]
        x = float(value[1])
        y = float(value[2])
        w = float(value[3])
        h = float(value[4])
	
        img_h, img_w = img.shape[:2]
        bb = convert((img_w, img_h), x,y,w,h)
        cv2.rectangle(img, (int(round(bb[0])),int(round(bb[2]))),(int(round(bb[1])),int(round(bb[3]))),           
            (55, 255, 155),
            5
        )
        cv2.putText(img, str(cls), (int(round(bb[0])),int(round(bb[2]))-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (55, 255, 155), 3)
       
    plt.figure()
    plt.imshow(img)
    plt.show()
