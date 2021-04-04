# -*- coding:utf-8 -*-
from ipdb import set_trace
import io
import os
import cv2
import mmcv
import json
import argparse
import numpy as np
import base64
import shutil
from skimage import measure
import pycocotools.mask as maskUtils
from mmdet.apis import init_detector, inference_detector

# 目前这个版本只能预测单类目标

# 设置预测结果为mask还是bbox
mask2json = 1
bbox2json = 1 - mask2json

 
def reference_labelme_json():
    ref_json_path = './reference_labelme.json'
    data = json.load(open(ref_json_path))
    return data
 
import colorsys
import random
def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors
 
def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors
 
def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)
 
def searchDirFile(rootDir, path_list, img_end):
    for dir_or_file in os.listdir(rootDir):
        filePath = os.path.join(rootDir, dir_or_file)
        # 判断是否为文件
        if os.path.isfile(filePath):
            # 如果是文件再判断是否以.jpg结尾，不是则跳过本次循环
            if os.path.basename(filePath).endswith(img_end):
                subname = filePath.split('/')[-1]
                path_list.append(subname)
            else:
                continue
        # 如果是个dir，则再次调用此函数，传入当前目录，递归处理。
        elif os.path.isdir(filePath):
            searchDirFile(filePath, path_list, img_end)
        else:
            print('not file and dir '+os.path.basename(filePath))
            exit()
 
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




def main(args):
 
    #rgb--
    cnum = 8
    self_color = ncolors(cnum)
    colorbar_vis = np.zeros((cnum*30, 100, 3), dtype=np.uint8)
    for ind,colo in enumerate(self_color):
        k_tm = np.ones((30, 100, 3), dtype=np.uint8) * np.array([colo[-1], colo[-2], colo[-3]])
        colorbar_vis[ind*30:(ind+1)*30, 0:100] = k_tm
    # cv2.imwrite('./colorbar_vis.png', colorbar_vis)
 
    mkdir_os(args.output_folder)
    mkdir_os(args.output_vis)
    mkdir_os(args.row_pic)
 
    score_thr = 0.5
    model = init_detector(args.input_config_file, args.input_checkpoint_file, device='cuda:0')
    
    trainimg = []
    searchDirFile(args.input_folder, trainimg, '.tif')
    for ind, val in enumerate(trainimg):
        print(ind, '/', len(trainimg))
        subname = trainimg[ind]
        suffix = subname.split('.')[1]
        name = os.path.join(args.input_folder, subname)
        
        result = inference_detector(model, name)
 
        ori_img = mmcv.imread(name)
        img = ori_img.copy()
        height, width = img.shape[:2]
 
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
 
        #这里注意inference_detector的结果顺序在mmdetectionrc1.0中
        #与训练时候的categories顺序相同,并不是categories的id顺序,所以训练时候注意json文件
        data_labelme = {}
        data_labelme['version'] = '4.2.9'
        data_labelme['flags'] = {}
        data_labelme['imagePath'] = subname
        with open(name,"rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
        data_labelme['imageData'] = imageData
        data_labelme['imageHeight'] = height
        data_labelme['imageWidth'] = width
 
        shapes = []
        thickness = 2
        # 检测
        if mask2json == 0 and bbox2json == 1:
            for label in range(len(bbox_result)):
                bbox = bbox_result[label]
                for i in range(bbox.shape[0]):
                    shape = {}
                    if bbox[i][4] > score_thr:
                        #颜色---rgb2bgr---imwrite
                        #self_color[0]是ignore的专属
                        #其他颜色和categories中id对应
                        id = label+1
                        cur_color = self_color[id][::-1]
                        # 单类预测
                        shape['label'] = model.CLASSES
                        shape['points'] = []
                        shape['shape_type'] = "rectangle"
                        shape['score'] = str(bbox[i][4])
                        shape['flags'] = {}
                        # labelme是x1y1x2y2
                        shape['points'].append([int(bbox[i][0]), int(bbox[i][1])])
                        shape['points'].append([int(bbox[i][2]), int(bbox[i][3])])
                        shapes.append(shape)
    
                        cv2.rectangle(img, (int(bbox[i][0]), int(bbox[i][1])), (int(bbox[i][2]), int(bbox[i][3])),
                                    (cur_color[0], cur_color[1], cur_color[2]),
                                    thickness)
                        cv2.putText(img, model.CLASSES + ':' + str(bbox[i][4])[:5], (int(bbox[i][0]), int(bbox[i][1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (cur_color[0], cur_color[1], cur_color[2]), 2)
    
            data_labelme['shapes'] = shapes
            with io.open(os.path.join(args.output_folder, subname.replace(suffix, 'json')), 'w',
                        encoding="utf-8") as outfile:
                my_json_str = json.dumps(data_labelme, ensure_ascii=False, indent=1)
                outfile.write(my_json_str)
            shutil.copyfile(name, os.path.join(args.row_pic, subname))
            cv2.imwrite(os.path.join(args.output_vis, subname), img)




        # 分割
        if mask2json == 1 and bbox2json == 0:
            bboxes = np.vstack(bbox_result)
            result_segms = []
            labels = []
            scores = []
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                np.random.seed(42)
                color_masks = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for i in inds:
                    
                    polygon = binary_mask_to_polygon(segms[i]+0,1)
                    if len(polygon)>1:
                        print('mask2polygon error!  place check!')
                        exit()
                    result_segms.append(polygon)
                    # labels.append(model.CLASSES[i]) # 多类预测
                    labels.append(model.CLASSES)  # 单类预测
                    scores.append(bboxes[i][-1])

                shapes = []
                for seg, label, score in zip(result_segms, labels, scores):
                    if score<0.1:
                        continue
                    # 绘制结果图
                    seg_np = np.array(seg[0],np.int32).reshape((-1,1,2))
                    cv2.polylines(img, [seg_np], True, self_color[0], 2)
                    cv2.putText(img, label + ':' + str(score)[:5], (seg_np[:,:,0].min(), seg_np[:,:,1].min()),cv2.FONT_HERSHEY_SIMPLEX, 0.8,  self_color[1], 2)
                    s = {"label":label,"shape_type":"polygon","flags":{}}
                    points = np.array(seg).reshape(-1,2).tolist()
                    s["points"] = points
                    s["label"] = label
                    s["score"] = score.item()
                    shapes.append(s)
                data_labelme["shapes"] = shapes
                
                if len(shapes)>0:
                    cv2.imwrite(os.path.join(args.output_vis, subname), img)
                    shutil.copyfile(name, os.path.join(args.row_pic, subname))
                    json.dump(data_labelme,open(os.path.join(args.output_folder, subname.replace(suffix, 'json')),"w"),ensure_ascii=False, indent=2)

            
            

 
 
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(
        description=
        "mmdetection_inference_result2labelme-json")
    parser.add_argument('-icf',
                        "--input_config_file",
                        default='/nfs/private/mmdetection/work_dirs/sample9_harbor/cascade_mask_rcnn_r50_fpn_20e_coco_strank/cascade_mask_rcnn_r50_fpn_20e_coco_strank.py',
                        help="set input folder1")
    parser.add_argument('-jcf',
                        "--input_checkpoint_file",
                        default='/nfs/private/mmdetection/work_dirs/sample9_harbor/cascade_mask_rcnn_r50_fpn_20e_coco_strank/max_epoch_24.pth',
                        help="set input folder2")
    parser.add_argument('-if',
                        "--input_folder",
                        default='/nfs/private/netdisk/192.168.0.36/object_detection/project/harbor/samples/sample9/storage_tank/tif/',
                        help="set input folder2")
    parser.add_argument('-of',
                        "--output_folder",
                        default='output_folder',
                        help="set output folder")
    parser.add_argument('-ov',
                        "--output_vis",
                        default='./vis_mmcv/',
                        help="set output folder")
    parser.add_argument('-rp',
                        "--row_pic",
                        default='./row_tif/',
                        help="set output folder")                        
    args = parser.parse_args()
 
    if args.input_config_file is None:
        parser.print_help()
        exit()
 
    main(args)