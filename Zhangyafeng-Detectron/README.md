该工程主要记录迁移学习的过程和配置:  
=============================
* 原生数据(LIVECell数据集)介绍[参考这个](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/285384)。
* [数据集获取仓库](https://github.com/sartorius-research/LIVECell)
* 迁移学习主要[参考这个](https://www.kaggle.com/markunys/sartorius-transfer-learning-train-with-livecell)。其中原数据图片是tif格式，需要转换成png格式，具体代码见trainTif2Png.py。
* 数据增强主要[参考这个](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/294006)。使用该帖子数据增强配置，在原生数据上面，效果一般。5k次迭代，MAP: 0.11284。7w次迭代, MAP：0.21(具体见权重[model_0073443.pth](https://pan.baidu.com/s/1A8Tpnxr8RlCB7au-iW4_OA), 提取码: 07i1)。
* 比赛数据集上面, 使用model_0073443.pth模型权重进行finetune，在479iter时候，MAP：0.262669，详见train_net_finetune.py。在3629次迭代的时候, MAP：0.28367.

