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

10. 根据mmdetection产生的work_dir中的json文件绘制loss和map折线图
程序位置：visualize_workdir_json/visualize_workdir_json.py

11. mmdetection预测脚本（生成结果图，生成每张测试图片对应的xml，json文件）
程序位置：detection_for_result/*（detection_for_mask_v5.py：预测并保存json，image_demo.py：单张图片预测，  image_demo_batch_test.py：批量图片预测）

12. kmeans聚类锚框（读取训练集xml标记聚类锚框）
程序位置：kmeans/anchor_kmeans.py

13. 阅读DetNet之后编写的DetNet的backbone
程序位置：detnet_backbone.py

14. 读取labelme标记的json文件，使用kmeans聚类锚框
程序位置：read_json_for_kmeans_anchor
结果格式：keymeans_result/

15. 修改预测脚本，将输出结果存入json（bbox,mask均可）
程序位置：mmdetection2labelme(单类bbox,mask).py
结果格式：产生三个文件夹（预测时指定），分别是原图，json，带结果的图
