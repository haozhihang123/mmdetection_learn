import json
import matplotlib.pyplot as plt
import sys
import os
from collections import OrderedDict
from ipdb import set_trace
class visualize_mmdetection():
    def __init__(self, path):
        self.log = open(path)
        self.dict_list = list()
        self.loss = list()
        self.val_mAP = list()

    def load_data(self):
        for line in self.log:
            info = json.loads(line)
            self.dict_list.append(info)

        for i in range(1, len(self.dict_list)):
            for value, key in dict(self.dict_list[i]).items():
                # ------------find key for every iter-------------------#
                if 'loss' in dict(self.dict_list[i]):
                    loss_value = dict(self.dict_list[i])['loss']
                    self.loss.append(loss_value)
                if 'mAP' in dict(self.dict_list[i]):
                    mAP_value = dict(self.dict_list[i])['mAP']
                    self.val_mAP.append(mAP_value)
        self.loss = list(OrderedDict.fromkeys(self.loss))
        self.val_mAP = list(OrderedDict.fromkeys(self.val_mAP))
        return self.loss, self.val_mAP

    def show_chart(self, draw_data):
        # 数据规整
        loss_len_min = 0
        map_len_min = 0
        for key, value in draw_data.items():
            if loss_len_min == 0:
                loss_len_min = len(value[0])
                map_len_min = len(value[1])
            else:
                loss_len_min = min(loss_len_min, len(value[0]))  
        plt.rcdefaults()
        plt.rcParams.update({'font.size': 15})
        plt.figure(0,figsize=(20, 20))
        # 绘制loss图
        for key, value in draw_data.items():
            plt.plot(value[0][:loss_len_min], label='loss_'+key)
            # map_xlabel = [i for i in range(0,loss_len_min,int(loss_len_min/map_len_min))]
            # plt.plot(map_xlabel[1:],value[1][:map_len_min], label='map_'+key)
        plt.legend()
        plt.savefig(('result_loss.png'))
        plt.close(0)
        # 绘制map图
        plt.figure(1,figsize=(20, 20))
        # 绘制loss图
        for key, value in draw_data.items():
            plt.plot(value[1][:map_len_min], label='map_'+key)
        plt.legend()
        plt.savefig(('result_map.png'))
        plt.close(1)



if __name__ == '__main__':
    jsons = 'jsons/'
    draw_data = dict()
    for json_file in os.listdir(jsons):
        x = visualize_mmdetection(jsons+json_file)
        loss, val_mAP = x.load_data()
        draw_data[json_file] = [loss, val_mAP]
    x.show_chart(draw_data)