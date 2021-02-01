# -*- coding=utf-8 -*-
#!/usr/bin/python
# 此代码仅仅用于单类显示
import os
import numpy as np
from pycocotools.coco import COCO
import cv2
import json
from ipdb import set_trace



json_path = './json/'
img_dir = './tif/'
output = './show/'
label = 'tank'

for json_file in os.listdir(json_path):
    print(json_file)
    # read_json and save seg
    with open(json_path + json_file) as f:
        json_info = json.load(f)
    shapes = json_info['shapes']
    bboxes = []
    polygons = []
    scores = []
    for shape in shapes:
        point = shape['points']
        polygon = np.array(point, dtype=np.int).reshape(-1,2)
        xmax = np.array(polygon)[:,0].max()
        xmin = np.array(polygon)[:,0].min()
        ymax = np.array(polygon)[:,1].max()
        ymin = np.array(polygon)[:,1].min()
        bbox = [xmin,ymin,xmax,ymax]
        bbox = np.array(bbox, dtype=np.int)
        polygons.append(polygon)
        bboxes.append(bbox)
        score = shape['score']
        scores.append(score)
    # plot picture
    
    im = cv2.imread(img_dir + json_file.replace('json','png'))
    for i in range(len(bboxes)):
        cv2.polylines(im, [polygons[i]], True, (0,255,255), 2)
        cv2.rectangle(im,(int(bboxes[i][0]),int(bboxes[i][1])),(int(bboxes[i][2]),int(bboxes[i][3])),(0,0,255),2)
        cv2.putText(im, label + ':' + str(round(scores[i],3)), (int(bboxes[i][0]),int(bboxes[i][1]-4)), cv2.FONT_HERSHEY_PLAIN,1, (255,0,0), 2, cv2.LINE_AA)
    cv2.imwrite(output + json_file.replace('json','png'), im)
    # set_trace()


    