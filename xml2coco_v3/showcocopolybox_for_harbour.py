# -*- coding=utf-8 -*-
#!/usr/bin/python

import os
import numpy as np
from pycocotools.coco import COCO
import cv2
import ipdb
anno_file = 'instances_train2017.json'
img_dir = './JPEGImages/'
output = './train_annotationJpeg/'
coco = COCO(anno_file)
categoryes = ['heat_engine_plant', 'SteelPlant', 'cement_plant']
catIds = coco.getCatIds(catNms=categoryes)

#imgIds = coco.getImgIds(catIds=catIds)
annoIds = coco.getAnnIds(catIds=catIds)
# image = coco.loadImgs(imgIds[0])[0]
#ipdb.set_trace()
for root, dirs, files in os.walk( img_dir ):
    n = 1
    for fn in files:
        img_file = root + '/' + fn
        img_name = fn.split('.')[0]
        im = cv2.imread(img_file)
        polygons = []
        bboxes = []
        label = []
        for i in range(len(annoIds)):
            annotation = coco.loadAnns(annoIds[i])[0]
            index = annotation['image_id']
            image = coco.loadImgs(index)[0]
            # print(image['file_name'])
            # print(fn)
            if image['file_name'] == fn:
                
                # segmentation = annotation['segmentation']
                # polygon = np.array(segmentation, dtype=np.int).reshape(-1,2)
                #ipdb.set_trace()
                bbox_info = annotation['bbox'] #x,y,w,h(左上角坐标和宽高)
                bbox = [bbox_info[0], bbox_info[1], bbox_info[0]+bbox_info[2], bbox_info[1]+bbox_info[3]] #xmin,ymin,xmax,ymax
                bbox = np.array(bbox, dtype=np.int)
                # print(bbox)
                # ipdb.set_trace()
                # polygons.append(polygon)
                bboxes.append(bbox)
                # ipdb.set_trace()
                label.append(categoryes[annotation['category_id']-1])

        
        for i in range(len(bboxes)):
            # cv2.polylines(im, [polygons[i]], True, (0,255,255), 2)
            cv2.rectangle(im,(int(bboxes[i][0]),int(bboxes[i][1])),(int(bboxes[i][2]),int(bboxes[i][3])),(0,0,255),2)
            cv2.putText(im, label[i], (int(bboxes[i][0]),int(bboxes[i][1]-4)), cv2.FONT_HERSHEY_PLAIN,1, (255,0,0), 2, cv2.LINE_AA)
        print(img_name)
        cv2.imwrite(output + img_name + '.jpg', im)


