# -*- coding=utf-8 -*-
#!/usr/bin/python

import sys
import os
import shutil
import numpy as np
import json
import xml.etree.ElementTree as ET
import ipdb
import cv2
from lxml import etree
import collections
import shutil

def parseXmlFiles(xml_path):
    tree  = etree.parse(xml_path)
    root = tree.getroot() 
    objectes = root.findall('.//object')
    bnd_box = []
    for object in objectes:
        name = object.find("name").text
        
        bndbox = object.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        xmax = float(bndbox.find("xmax").text)
        ymin = float(bndbox.find("ymin").text)
        ymax = float(bndbox.find("ymax").text)
        
        bnd_box.append([name, xmin, xmax, ymin, ymax])
    return bnd_box


# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
START_IMAGE_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {"heat_engine_plant": 1, "SteelPlant": 2,  "cement_plant": 3}
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}
xml_dir = 'Annotations/'
categories = PRE_DEFINE_CATEGORIES
image_id = START_IMAGE_ID
bnd_id = START_BOUNDING_BOX_ID

#for json_file in os.listdir(json_dir):
for root, dirs, files in os.walk(xml_dir):
    # 训练部分
    # ipdb.set_trace()
    json_dict=collections.OrderedDict()  #将普通字典转换为有序字典
    json_dict['images']=[]
    json_dict['type']='instances'
    json_dict['annotations']=[]
    json_dict['categories']=[]    
    box_id = 0
    for fn in files:
        xml_file = root + '/' + fn
        xml_name = fn.split('.')[0]
        images_name = xml_name + '.jpg'
        # shutil.copy(JPEGImages_dir + '/' + images_name, jpg_train_dir + '/')
        # ipdb.set_trace()
        #image_id = images_name.split('_')[-1].split('.')[0]
        image = {'file_name': images_name,
                 'height': 2500,
                 'width': 2500,
                 'id': image_id}
        image_id = image_id + 1
        json_dict['images'].append(image)
        
        # 读取json，获取分割信息
        bnd_box = parseXmlFiles(xml_file)
        
        # with open(json_file) as f:
        #     json_info = json.load(f)
        # shapes = json_info['shapes']

        for bnd_id, bbox in enumerate(bnd_box):
            
            categories_id = categories[bbox[0]]
            xmin = bbox[1]
            xmax = bbox[2]
            ymin = bbox[3]
            ymax = bbox[4]
            x,y,w,h = xmin, ymin, xmax-xmin, ymax-ymin 
            
            # print(xmin,ymin,xmax,ymax)
            annotations = dict()
            annotations['area'] = w * h
            annotations['iscrowd'] = 0
            annotations['image_id'] = image_id - 1
            annotations['bbox'] = [x, y, w, h]
            annotations['category_id'] = categories_id
            annotations['id'] = box_id + 1
            # print(annotations['id'])
            box_id += 1
            annotations['ignore'] = 0
            # ipdb.set_trace()
            json_dict['annotations'].append(annotations)
        # ipdb.set_trace()
    # 保存训练json结果
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    # ipdb.set_trace()
    json_file = 'instances_train2017.json'
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print('train.json finish!')

    
