import glob
import json
from ipdb import set_trace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# ANNOTATIONS_PATH = "./data/pascalvoc07-annotations"
ANNOTATIONS_PATH = "./json"
CLUSTERS = 9
BBOX_NORMALIZE = False

def show_cluster(data, cluster, max_points=2000):
	'''
	Display bouding box's size distribution and anchor generated in scatter.
	'''
	if len(data) > max_points:
		idx = np.random.choice(len(data), max_points)
		data = data[idx]
	plt.scatter(data[:,0], data[:,1], s=5, c='lavender')
	plt.scatter(cluster[:,0], cluster[:, 1], c='red', s=100, marker="^")
	plt.xlabel("Width")
	plt.ylabel("Height")
	plt.title("Bounding and anchor distribution")
	plt.savefig("cluster.png")
	plt.show()

def show_width_height(data, cluster, bins=50):
	'''
	Display bouding box distribution with histgram.
	'''
	if data.dtype != np.float32:
		data = data.astype(np.float32)

	width = data[:, 0]
	height = data[:, 1]
	ratio = height / width
	area = ((data[:, 0]*data[:, 1])**0.5).astype(np.int32)

	plt.figure(1,figsize=(20, 10))
	plt.subplot(221)
	plt.hist(width, bins=bins, color='green')
	plt.xlabel('width')
	plt.ylabel('number')
	plt.title('Distribution of Width')

	plt.subplot(222)
	plt.hist(height,bins=bins, color='blue')
	plt.xlabel('Height')
	plt.ylabel('Number')
	plt.title('Distribution of Height')

	plt.subplot(223)
	plt.hist(ratio, bins=bins,  color='magenta')
	plt.xlabel('Height / Width')
	plt.ylabel('number')
	plt.title('Distribution of aspect ratio(Height / Width)')

	plt.subplot(224)
	ax1 = plt.gca()
	ax1.set_xlim([0,400])#重点是对获取到的axes对象进行操作
	ax1.xaxis.set_ticks([i*50 for i in range(9)])
	plt.hist(area, bins=bins,  color='red')
	plt.xlabel('area**0.5')
	plt.ylabel('number')
	plt.title('Distribution of area')

	x_major_locator=MultipleLocator(1)
	#把x轴的刻度间隔设置为1，并存在变量里
	y_major_locator=MultipleLocator(10)
	#把y轴的刻度间隔设置为10，并存在变量里



	plt.savefig("shape-distribution.png")
	plt.show()
	
def sort_cluster(cluster):
	'''
	Sort the cluster to with area small to big.
	'''
	if cluster.dtype != np.float32:
		cluster = cluster.astype(np.float32)
	area = cluster[:, 0] * cluster[:, 1]
	cluster = cluster[area.argsort()]
	ratio = cluster[:,1:2] / cluster[:, 0:1]
	return np.concatenate([cluster, ratio], axis=-1)

def read_json(json_file):
	with open(json_file) as f:
        	json_info = json.load(f)
	shapes = json_info['shapes']
	height = json_info['imageHeight']
	width = json_info['imageWidth']
	return height, width, shapes
	
def load_dataset(path, normalized=True):
	'''
	load dataset from pasvoc formatl xml files
	'''
	dataset = []
	for json_file in glob.glob("{}/*json".format(path)):
		print(json_file)
		height, width, shapes = read_json(json_file)
		for shape in shapes:
			point = shape['points']
			polygon = np.array(point, dtype=np.int).reshape(-1,2)
			xmax = np.array(polygon)[:,0].max()
			xmin = np.array(polygon)[:,0].min()
			ymax = np.array(polygon)[:,1].max()
			ymin = np.array(polygon)[:,1].min()
			if normalized:
				xmin = xmin / float(width)
				ymin = ymin / float(height)
				xmax = xmax / float(width)
				ymax = ymax / float(height)
			if (xmax - xmin) == 0 or (ymax - ymin) == 0:
				continue # to avoid divded by zero error.
			dataset.append([xmax - xmin, ymax - ymin])

	return np.array(dataset)

def iou(boxes, clusters):
    """
    Calculates the Intersection over Union (IoU) between N boxes and K clusters.
    :param boxes: numpy array of shape (n, 2) where n is the number of box, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (n, k) where k is the number of clusters
    """
    N = boxes.shape[0]
    K = clusters.shape[0]
    iw = np.minimum(
        np.broadcast_to(boxes[:, np.newaxis, 0], (N, K)),    # (N, 1) -> (N, K)
        np.broadcast_to(clusters[np.newaxis, :, 0], (N, K))  # (1, K) -> (N, K)
    )
    ih = np.minimum(
        np.broadcast_to(boxes[:, np.newaxis, 1], (N, K)),
        np.broadcast_to(clusters[np.newaxis, :, 1], (N, K))
    )
    if np.count_nonzero(iw == 0) > 0 or np.count_nonzero(ih == 0) > 0:
        raise ValueError("Some box has no area")

    intersection = iw * ih   # (N, K)
    boxes_area = np.broadcast_to((boxes[:, np.newaxis, 0] * boxes[:, np.newaxis, 1]), (N, K))
    clusters_area = np.broadcast_to((clusters[np.newaxis, :, 0] * clusters[np.newaxis, :, 1]), (N, K))

    iou_ = intersection / (boxes_area + clusters_area - intersection + 1e-7)

    return iou_

def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean(np.max(iou(boxes, clusters), axis=1))

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

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    iter_num = 1
    while True:
        print("Iteration: %d" % iter_num)
        iter_num += 1

        distances = 1 - iou(boxes, clusters)
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            if len(boxes[nearest_clusters == cluster]) == 0:
                print("Cluster %d is zero size" % cluster)
                # to avoid empty cluster
                clusters[cluster] = boxes[np.random.choice(rows, 1, replace=False)]
                continue

            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

print("Start to load data annotations on: %s" % ANNOTATIONS_PATH)
data = load_dataset(ANNOTATIONS_PATH, normalized=BBOX_NORMALIZE)

print("Start to do kmeans, please wait for a moment.")
out = kmeans(data, k=CLUSTERS)

out_sorted = sort_cluster(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))

show_cluster(data, out, max_points=2000)

if out.dtype != np.float32:
	out = out.astype(np.float32)

print("Recommanded aspect ratios(width/height)")
print("Width    Height   Width/Height")
for i in range(len(out_sorted)):
	print("%.3f      %.3f     %.1f" % (out_sorted[i,0], out_sorted[i,1], out_sorted[i,0]/out_sorted[i,1]))
show_width_height(data, out, bins=50)
