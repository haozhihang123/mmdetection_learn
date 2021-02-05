from argparse import ArgumentParser
import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace
import os
import csv



def main():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image dir')
    parser.add_argument('config_dir', help='Config dir')
    parser.add_argument('checkpoint_dir', help='Checkpoint dir')
    parser.add_argument('result_dir', help='result dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.9, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config_dir, args.checkpoint_dir, device=args.device)
    
    # test some img
    select_result = []
    for img_file in os.listdir(args.img_dir):
        print(img_file)
        img = args.img_dir + img_file
        result = inference_detector(model, img)

        # show the results
        result_pic = show_result_pyplot(model, img, result, score_thr=args.score_thr)
        plt.figure(figsize=(15, 10))
        plt.imshow(mmcv.bgr2rgb(result_pic))
        plt.title('result')
        plt.tight_layout()
        plt.savefig(args.result_dir + img_file)

        # save result to csv
        # for box in result[0]:
        #     box = box.tolist()
        #     if box[-1]>args.score_thr:
        #         select_result.append([img_file.split('.')[0] ,box[-1], int(box[0]), int(box[1]), int(box[2])-int(box[0]), int(box[3])-int(box[1])])
    # set_trace()
    # with open("restlt.csv", 'a', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file, delimiter=',')
    #     for res in select_result:
    #         writer.writerow((res[0], str(res[1])[:5] + ' ' + str(res[2]) + ' ' + str(res[3]) + ' ' + str(res[4]) + ' ' + str(res[5])))
    # print('asda')


if __name__ == '__main__':
    main()
