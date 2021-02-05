from argparse import ArgumentParser
import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace
def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('result_path', help='result path')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)

    # show the results
    # img = show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    # plt.figure(figsize=(15, 10))
    # plt.imshow(mmcv.bgr2rgb(img))
    # plt.title('result')
    # plt.tight_layout()
    # plt.savefig(args.result_path + args.img.split('/')[-1])


if __name__ == '__main__':
    main()