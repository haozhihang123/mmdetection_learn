# -*- coding: utf-8 -*-
import glob
import xml.etree.ElementTree as ET
import numpy as np
from ipdb import set_trace
import cv2

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y 
    box_area = box[0] * box[1] 
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)

def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters

    return clusters

def draw_plot(img_dir, save_dir,w_h):
    im = cv2.imread(img_dir)
    img_h, img_w, _ = im.shape
    bboxes = []
    for box in w_h:
        xmin = int((img_w-box[0])/2)
        ymin = int((img_h-box[1])/2)
        xmax = int((img_w+box[0])/2)
        ymax = int((img_h+box[1])/2)
        bboxes.append([xmin,ymin,xmax,ymax])
    for i in range(len(bboxes)):
        cv2.rectangle(im,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]),(0,0,255),2)
        cv2.putText(im, str(w_h[i]), (int(bboxes[i][0]),int(bboxes[i][1]-4)), cv2.FONT_HERSHEY_PLAIN,1, (255,0,0), 2, cv2.LINE_AA)
    cv2.imwrite(save_dir, im)



ANNOTATIONS_PATH = "xml"
CLUSTERS = 9

def load_dataset(path):
	dataset = []
	for xml_file in glob.glob("{}/train/*xml".format(path))+glob.glob("{}/val/*xml".format(path)):
		print(xml_file)
		tree = ET.parse(xml_file)

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			xmin = int(float(obj.findtext("bndbox/xmin"))) / width
			ymin = int(float(obj.findtext("bndbox/ymin"))) / height
			xmax = int(float(obj.findtext("bndbox/xmax"))) / width
			ymax = int(float(obj.findtext("bndbox/ymax"))) / height
			dataset.append([xmax - xmin, ymax - ymin])
	return np.array(dataset)
data = load_dataset(ANNOTATIONS_PATH)
print('data shape is {}'.format(data.shape))
out = kmeans(data, k=CLUSTERS)
yolov3clusters = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
yolov3out= np.array(yolov3clusters)/416.0

print("self data Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("yolov3 Accuracy: {:.2f}%".format(avg_iou(data, yolov3out) * 100))
# print("Boxes:\n {}-{}".format(out[:, 0]*416, out[:, 1]*416))
out_true = np.array(out[:,:]*416,dtype=int)
print("Boxes:\n {}".format(out_true))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))

# 将生成的矩形框画在图中
test_pic = 'test.jpg'
save_dir = 'kmeans.jpg'
draw_plot(test_pic, save_dir,out_true)

# 将原始yolo锚框画在图中
test_pic = 'test.jpg'
save_dir = 'yolo.jpg'
draw_plot(test_pic, save_dir,yolov3clusters)