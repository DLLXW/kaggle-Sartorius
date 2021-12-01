### kaggle-Sartorius
[Sartorius - Cell Instance Segmentation Detect](https://www.kaggle.com/c/sartorius-cell-instance-segmentation)

我线下训练所用的环境:
- mmcv-full                 1.3.17
- torch                     1.7.1+cu101    
- torchvision               0.8.2+cu101    

### code
- mmdetection-neuron-inference.ipynb :用于mmdet2.18的推理代码
- swin-transformer-for-detection.ipynb:用于swin-transformer-for-detection的推理代码
- mmdet-218/ 训练代码
- swin-mmdet/ 训练代码

由于线上kaggle是cuda11.0，所以线上提交需要安装torch和mmcv-full都需要使用cuda11.0的版本（在我share的notebook:/input中全部包含完毕了，可直接使用）

### summary
- 数据集:[coco-train/val](https://www.kaggle.com/vexxingbanana/sartorius-coco-dataset-notebook)
- 方案一：最新版的[mmdet-2.18.0](https://github.com/open-mmlab/mmdetection)
  
    - mask-rcnn-swin-small LB:30.6; CV(segm-mAP):22.6 配置文件在:qyl_config/swin_s.py
    
- 方案二：[Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)
    - cascade-mask-rcnn-swin-small LB:30.1 CV(segm-mAP):24  配置文件在:qyl_config/swin_s.py
    - cascade-mask-rcnn-swin-base LB:30.1 CV(segm-mAP):24.5  配置文件在:qyl_config/swin_b.py

train模型(单卡):
```
python tools/train.py ./qyl_config/swin_s.py --gpu-ids 3
```

### To do

下一步要做的事情罗列：
- 不同epoch波动大，所以需要改变验证策略
- 线下验证集:使用官方验证集，[参考](https://www.kaggle.com/theoviel/competition-metric-map-iou/notebook),需要想办法将这个iou-mAP集成到mmdet的验证中
- 五折训练，参考code区5-fold-coco数据集制作，但是五个模型的预测结果融合需要进一步思考(NMS or WBF ,能够针对MASK吗)
- 调参：目前只是在默认配置上更改了训练尺度和一丁点的数据增强，感觉这两个方面还需呀进一步探索，根据dicussuion区讨论，mmdet的模型单折到32+是完全没问题的


