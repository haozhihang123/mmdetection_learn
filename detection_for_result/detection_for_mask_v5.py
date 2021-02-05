'''
功能：针对mmdetection将预测的结果存放到txt，xml，json中
版本：v3
新增：运行程序时先格式化之前的数据
'''
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os
import ipdb
import sys
from lxml import etree
import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import shutil
import time
import ipdb
import json
from PIL import Image
from skimage import measure
import base64
from ipdb import set_trace

# 模型文件
# root =  '/opt/netdisk/192.168.0.36/d/object_detection/project/harbor/results/ship/sample8/cascade_mask_rcnn_r50_fpn_20e_attention_dcn_coco_ship/'
# config_file = root + 'cascade_mask_rcnn_r50_fpn_20e_attention_dcn_coco_ship.py'
# checkpoint_file =root + './epoch_20.pth'


# 测试文件路径
# test_pic_dir = '/opt/netdisk/192.168.0.36/d/object_detection/project/harbor/samples/harbour/old/caofeidian_harbour/dst_path/'
# test_pic_dir = '/opt/netdisk/192.168.0.36/d/object_detection/project/harbor/samples/sample6/resources/test/tif/'
# test_pic_dir = '/opt/netdisk/192.168.0.36/d/object_detection/project/harbor/samples/sample7/Storage_tank/test/tif/'
# test_pic_dir = '/opt/netdisk/192.168.0.36/d/object_detection/project/harbor/data/harbour/new/tianjin/dst_path/'
# test_pic_dir = '/opt/netdisk/192.168.0.36/d/object_detection/project/harbor/data/harbour/new/tianjin_new_youpian/dst/'
def process_arguments(argv):
    if len(argv) < 5:
        print('please input test_pic_dir, save_dir, config_file, checkpoint_file')
        exit()
    test_pic_dir = argv[1]
    save_dir = argv[2]
    config_file = argv[3]
    checkpoint_file = argv[4]
    return test_pic_dir, save_dir, config_file, checkpoint_file
test_pic_dir, save_dir, config_file, checkpoint_file= process_arguments(sys.argv)


xml_paht = save_dir + '/result_pic/'
save_jpg_path =save_dir + '/result_pic/'
save_row_pic =save_dir + '/result_pic/'

save_seg_json_dir =save_dir + '/seg_json/'
save_seg_tif_dir =save_dir + '/seg_tif/'
# 检查文件夹是否存在
def check_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(path + 'has been reset!')
    else:
        print(path + 'creat new folder!')
    os.makedirs(path)

check_path(xml_paht)
check_path(save_seg_json_dir)
check_path(save_seg_tif_dir)


# 初始化模型
model = init_detector(config_file, checkpoint_file)





# 保存检测结果到txt
def save_txt(f,result):
        n=len(result[0])
        save.append(f[0:-4]+'\n')
        save.append(str(n)+'\n')
        print(n)
        if n>0:
                for i in range(n):       
                        save.append(result[0][i][0].__str__()+'\t')
                        save.append(result[0][i][1].__str__()+'\t')
                        save.append(result[0][i][2].__str__()+'\t')
                        save.append(result[0][i][3].__str__()+'\t')
                        save.append(result[0][i][4].__str__()+'\n')

class GEN_Annotations:
    def __init__(self, filename):
        self.root = etree.Element("annotation")
        child1 = etree.SubElement(self.root, "folder")
        child1.text = "VOC2007"
        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename
        child3 = etree.SubElement(self.root, "source")
        child4 = etree.SubElement(child3, "annotation")
        child4.text = "PASCAL VOC2007"
        child5 = etree.SubElement(child3, "database")
        child5.text = "Unknown"
        child6 = etree.SubElement(child3, "image")
        child6.text = "flickr"
        #child7 = etree.SubElement(child3, "flickrid")
        #child7.text = "35435"

    def set_size(self,witdh,height,channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)

    def savefile(self,filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

    def add_pic_attr(self,label,score,xmin,ymin,xmax,ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        pose = etree.SubElement(object, "pose")
        pose.text = str(0)
        scores = etree.SubElement(object, "score")
        scores.text = str(score)
        truncated = etree.SubElement(object, "truncated")
        truncated.text = str(0)
        difficult = etree.SubElement(object, "difficult")
        difficult.text = str(0)
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)
    def genvoc(filename,class_,width,height,depth,xmin,ymin,xmax,ymax,savedir):
        anno = GEN_Annotations(filename)
        anno.set_size(width,height,depth)
        anno.add_pic_attr("pos", xmin, ymin, xmax, ymax)
        anno.savefile(savedir)

# 保存检测结果到xml
def save_xml(file_name,img_shape,results):
        anno= GEN_Annotations(file_name)
        anno.set_size(*img_shape)
        for result in results[0]:
                xmin = result[0]
                ymin = result[1]
                xmax = result[2]
                ymax = result[3]
                score = result[4]
                anno.add_pic_attr("ship",score,xmin,ymin,xmax,ymax)
                anno.savefile(xml_paht + file_name.split('.')[0] + '.xml')

# 保存检测结果bbox到json中
def save_labelme_box_json(file_name,results):
    labelme_format = {
    "version":"3.6.16",
    "flags":{},
    "imagePath":file_name,
    "imageHeight":1024,
    "imageWidth":1024
    }
    # ipdb.set_trace()
    labelme_format["imageData"] = 'None'
    for i,boxs in enumerate(results):
        label = model.CLASSES[i]
        if len(boxs)==0:
            continue
        shapes = []
        for box in boxs:
            s = {"label":label,"shape_type":"rectangle"}
            points = [
                [int(box[0]),int(box[1])],# xmin,ymin
                [int(box[2]),int(box[3])] # xmax,ymax
                ]
            s["points"] = points
            shapes.append(s)
    labelme_format["shapes"] = shapes
    json.dump(labelme_format,open("%s/%s"%(save_json_dir,file_name.replace(".tif",".json")),"w"),ensure_ascii=False, indent=2)    

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    

    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    # 如果上一步找到了多个不同的结果，分别保存就行了，
    # contours = np.subtract(contours, 1)

    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def save_labelme_segm_json(file_name,results):
    labelme_format = {
    "version":"3.6.16",
    "flags":{},
    "imagePath":file_name,
    "imageHeight":1024,
    "imageWidth":1024
    }
    with open(test_pic_dir+file_name,"rb") as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')
    labelme_format["imageData"] = imageData
    # 遍历segms
    result_segms = []
    labels = []
    scores = []
    for i, segms in enumerate(results[1]):
        if len(segms):
            for k, segm in enumerate(segms):
                # 可能转换成多个多边形
                polygons = binary_mask_to_polygon(segm+0,1)

                # 需要增加
                for polygon in polygons:
                    result_segms.append(polygon)
                    
                    # labels.append(model.CLASSES[i]) # 多类预测
                    labels.append(model.CLASSES)  # 单类预测
                    scores.append(results[0][i][k][4])
    shapes = []
    for seg, label,score in zip(result_segms, labels, scores):
        if score<0.1:
            continue
        s = {"label":label,"shape_type":"polygon","flags":{}}
        # for a in range(len(seg)):
        #     seg[a] = int(seg[a])

        points = np.array(seg).reshape(-1,2).tolist()
        s["points"] = points
        s["label"] = label
        s["score"] = score.item()
        shapes.append(s)
    labelme_format["shapes"] = shapes
    # ipdb.set_trace()
    if len(shapes)>0:
        shutil.copyfile(test_pic_dir + file_name,save_seg_tif_dir + file_name)
        json.dump(labelme_format,open("%s/%s"%(save_seg_json_dir,file_name.split('.')[0]+'.json'),"w"),ensure_ascii=False, indent=2)    

# 检测所有测试图像
def detec_all_pic():
    # get all img
    font = cv2.FONT_HERSHEY_SIMPLEX
    n = 0
    for f in os.listdir(test_pic_dir):
        if f == 'Thumbs.db':
            print('Thumbs.db')
            continue
        # test a single image and show the results
        img = test_pic_dir + f
        # ipdb.set_trace()
        print(img)
        result_row = inference_detector(model, img)
        result = result_row
        object_num = 0
        class_num = len(result[0])
        for i in range(class_num):
            object_num += len(result[0][i])
        print('file:{}    detect num:{}'.format(img,object_num))
        if object_num <1:
            continue

        # 针对单类目标，mask
        for i in range(len(result[0][0])-1,-1,-1):
            
            if result[0][0][i][0]== result[0][0][i][2] or  result[0][0][i][1]== result[0][0][i][3]:
                result[0][0] = np.delete(result[0][0],i,0)
                result[1][0] = np.delete(result[1][0],i,0)
        object_num = 0
        class_num = len(result[0])
        for i in range(class_num):
            object_num += len(result[0][i])
        print('file:{}    detect num:{}'.format(img,object_num))
        if object_num <1:
            continue
        # 保存box信息到json,直接传递bbox信息
        # save_labelme_box_json(f,result[0])    # result[0]是分割模型的所有矩形框检测结果
        # 模型输出的结果都传进去（包括bbox和segm）
        save_labelme_segm_json(f,result)

        # 可视化并保存--自己编写,问题解决
        # shutil.copyfile(img,save_row_pic + f)
        # image = cv2.imread(img)
        # for i,resul in enumerate(result[0]):
        #     if len(resul) == 0:
        #         continue 
        #     for result_1 in resul:
        #         xmin = int(result_1[0])
        #         ymin = int(result_1[1])
        #         xmax = int(result_1[2])
        #         ymax = int(result_1[3])
        #         score = result_1[4]
        #         if score > 0.1:
        #             cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),2)
        #             cv2.putText(image,model.CLASSES[i]+str(score),(xmin,ymin),font,1.0,(0,255,0),2)
        # cv2.imwrite(save_jpg_path + f, image)
        # visualize the results in a new window
        # show_result(img, result, model.CLASSES)
        # or save the visualization results to image file
        im = show_result_pyplot(model, img, result, score_thr=0.1, fig_size=(15, 15))
        im = Image.fromarray(mmcv.bgr2rgb(im))
        im.save(save_dir + '/result_pic/'+f)
        # ipdb.set_trace()

detec_all_pic()
