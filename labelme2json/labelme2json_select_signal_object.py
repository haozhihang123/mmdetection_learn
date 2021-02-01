# -*- coding=utf-8 -*-
#!/usr/bin/python
# coco的bbox格式为(x,y,w,h)左上角坐标和宽高
import sys
import os
import shutil
import numpy as np
import json
import xml.etree.ElementTree as ET
import ipdb
import cv2
# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
START_IMAGE_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
# If necessary, pre-define category and its id
# PRE_DEFINE_CATEGORIES = {"Storage_tank": 1, "ship": 2, "Container": 3, "resources": 4,
#                           "Crane":5, 'Small_Wharf':6}
PRE_DEFINE_CATEGORIES = {"Storage_tank": 1}

#json_dir = './all_json/'
#json_dir = './test_points_5/'
# json_dir = './train2017/'
json_dir = 'json'
tif_dir = 'tif'
json_dict = {"images": [],
             "type": "instances",
             "annotations": [],
             "categories": []}
categories = PRE_DEFINE_CATEGORIES
image_id = START_IMAGE_ID
bnd_id = START_BOUNDING_BOX_ID

#for json_file in os.listdir(json_dir):
for root, dirs, files in os.walk( json_dir ):
    for fn in files:
        print(fn)
        json_file = root + '/' + fn
        json_name = fn.split('.')[0]
        images_name = json_name + '.tif'
        
        # ipdb.set_trace()
        with open(json_file) as f:
            json_info = json.load(f)
        shapes = json_info['shapes']
        # 判断是否包含单类目标
        flg = 0
        for shp in shapes:
            if shp['label'] == 'Storage_tank':
                flg = 1
                break
        if flg == 0:
            continue


        # read tif
        img = cv2.imread(tif_dir + '/' + json_name + '.tif' ,1)
        img_h = img.shape[0]
        img_w = img.shape[1]
        image = {'file_name': images_name,
                 'height': img_h,
                 'width': img_w,
                 'id': image_id}
        image_id = image_id + 1
        json_dict['images'].append(image)

        for shape in shapes:
            label = shape['label']
            if label != 'Storage_tank':
                continue
            # ipdb.set_trace()
            categories_id = categories[label]
            polygon = shape['points']
            segmentation = np.array(polygon).reshape(1, -1).tolist()
            xmax = np.array(polygon)[:,0].max()
            xmin = np.array(polygon)[:,0].min()
            ymax = np.array(polygon)[:,1].max()
            ymin = np.array(polygon)[:,1].min()
            # x,y,w,h = (xmax+xmin)/2, (ymax+ymin)/2, xmax-xmin, ymax-ymin 
            #x, y, w, h = cv2.boundingRect(np.array(polygon))
            x,y,w,h = xmin, ymin, xmax-xmin, ymax-ymin 
            # x, y, w, h = cv2.boundingRect(polygon)
            # ipdb.set_trace()
            annotations = dict()
            annotations['area'] = w * h
            annotations['iscrowd'] = 0
            annotations['image_id'] = image_id - 1
            annotations['bbox'] = [x, y, w, h]
            annotations['category_id'] = categories_id
            annotations['id'] = bnd_id
            annotations['ignore'] = 0
            annotations['segmentation'] = segmentation
            json_dict['annotations'].append(annotations)

            bnd_id = bnd_id + 1
            # ipdb.set_trace()
        

for cate, cid in categories.items():
    cat = {'supercategory': 'none', 'id': cid, 'name': cate}
    json_dict['categories'].append(cat)

# json_file = './annotations/train.json'
json_file = 'instances_val2017.json'
json_fp = open(json_file, 'w')
json_str = json.dumps(json_dict)
json_fp.write(json_str)
json_fp.close()
print('convert finish')


#ipdb.set_trace()
 

