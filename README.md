# mmdetection_learn
1. 可视化输入网络的图像和标签（数据增强后的图像和标签）
程序位置：train.py
结果格式：show_train_pic

2. 可视化RPN锚框产生及选择出来的正负样本
程序位置：anchor_head.py(mmdetection/mmdet/models/dense_heads/anchor_head.py)
结果格式：show_rpn

3. 可视化backbone输出的特征图
程序位置：two_stage.py（draw_features函数）
结果格式：features_show

4. 批量预测结果并存储为json格式（labelme能够读取）
程序位置：detection_for_mask_v5.py

5. 将预测结果josn转换为shp矢量格式
程序位置：json_2_shp_mokatuo_v3_poly.py

6. 样本制作（读取json显示在原图上）
程序位置：json2show/label_json_show_ann.py

7. 样本制作（labelme转coco）
程序位置：labelme2json/

8. 样本制作（xml转coco）
程序位置：xml2coco_v3/

9. kmeans聚类出anchor的长宽比
程序位置：anchor_kmeans.py

