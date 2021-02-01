# !/usr/bin/env python
# encoding: utf-8
from __future__ import print_function
import sys
import os
import numpy as np
import gdal
import xml.dom.minidom
import shapefile 
import math
import json
from ipdb import set_trace
import sys
import shutil

def check_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def webmercator_to_lonlat(x, y):
    _x = x / 20037508.342787001 * 180
    _y = y / 20037508.342787001 * 180
    lon = _x
    lat = (180.0 / math.pi) * 2.0 * (math.atan(math.exp(_y * (math.pi / 180.0))) - math.pi / 4.0)
    return lon, lat

def readjson(filename):
    with open(filename) as f:
        json_info = json.load(f)
    shapes = json_info['shapes']
    bboxes = []
    polygons = []
    for shape in shapes:
        point = shape['points']
        polygon = np.array(point, dtype=np.int).reshape(-1,2)
        xmax = np.array(polygon)[:,0].max()
        xmin = np.array(polygon)[:,0].min()
        ymax = np.array(polygon)[:,1].max()
        ymin = np.array(polygon)[:,1].min()
        bbox = [xmin,ymin,xmax,ymax]
        bbox = np.array(bbox, dtype=np.int)
        bboxes.append(bbox)
        polygons.append(polygon)
    return bboxes, polygons

def read_shp(filename):
    filename = filename.replace('json','tif')
    dataset=gdal.Open(filename)
    if dataset is None:
        print('FATAL: GDAL open file failed. [%s]'%filename)
        sys.exit(1)
    img_coord=dataset.GetGeoTransform()
    return img_coord

def main():
    json_dir,src_shp_dir,out_shp_dir,filename = process_arguments(sys.argv)
    check_path(out_shp_dir)
    list_box = []
    box_name = []
    polygon_news = []
    for json_file in os.listdir(json_dir):
        print(json_file)
        img_coord = read_shp(src_shp_dir + json_file)
        boxs, polygons = readjson(json_dir + json_file)
        # 保存矩形到shp
        for i, item in enumerate(boxs):  
            xmin,ymin,xmax,ymax = item
            xmin = img_coord[0] + img_coord[1]*xmin
            ymin = img_coord[3] + img_coord[5]*ymin
            xmax = img_coord[0] + img_coord[1]*xmax
            ymax = img_coord[3] + img_coord[5]*ymax
            # 魔卡托坐标转换
            xmin,ymin = webmercator_to_lonlat(xmin,ymin)
            xmax,ymax = webmercator_to_lonlat(xmax,ymax)            
            name=b'resource'
            # set_trace()
            list_box.append([xmin,ymin,xmax,ymax,name])
            box_name.append(json_file.split('.')[0] + '__' + str(i))
        # 保存多边形到shp
        for i, item in enumerate(polygons):  
            polygon_new = []
            for k in item:
                x = img_coord[0] + img_coord[1]*k[0]
                y = img_coord[3] + img_coord[5]*k[1]
                # 魔卡托坐标转换
                x,y = webmercator_to_lonlat(x,y) 
                polygon_new.append([x,y])
            polygon_news.append(polygon_new)    
            name=b'resource'
            list_box.append([xmin,ymin,xmax,ymax,name])
            box_name.append(json_file.split('.')[0] + '__' + str(i))
    # 创建矩形框shp
    sf=shapefile.Writer(out_shp_dir + '/box_' + filename, shapeType=5)
    sf.field('FID','C','40')
    sf.field('shape','C','40')
    sf.field('name','C','40')
    for i, item in enumerate(list_box):
        # set_trace()
        xmin,ymin,xmax,ymax,name=item[0],item[1],item[2],item[3],item[4]
        sf.poly([[[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax],[xmin,ymin]]])
        sf.record(box_name[i],'Polygon',name)    
    sf.close()

    # 创建多边形框shp
    sf=shapefile.Writer(out_shp_dir + '/polygon_' + filename, shapeType=5)
    sf.field('FID','C','40')
    sf.field('shape','C','40')
    sf.field('name','C','40')
    for i, item in enumerate(polygon_news):
        # set_trace()
        sf.poly([item])
        sf.record(box_name[i],'Polygon',name)    
    sf.close()
    

def process_arguments(argv):
    if len(argv) < 5:
        help()
    json_dir = argv[1]
    src_shp_dir = argv[2]
    out_shp_dir = argv[3]
    filename = argv[4]
    return json_dir,src_shp_dir,out_shp_dir,filename

def help():
    print('Usage: python json_shp_GE_WS.py json_dir src_shp_dir out_shp_dir item\n')
    exit()

if __name__ == '__main__':
    main()

                






    
