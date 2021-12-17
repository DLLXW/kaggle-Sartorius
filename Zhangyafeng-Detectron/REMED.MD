该工程主要记录迁移学习的过程和配置:  
=============================
* 原生数据(LIVECell数据集)介绍[参考这个](https://www.kaggle.com/markunys/sartorius-transfer-learning-train-with-livecell)。
* [数据集获取仓库](https://github.com/sartorius-research/LIVECell)
* 迁移学习主要[参考这个](https://www.kaggle.com/markunys/sartorius-transfer-learning-train-with-livecell)。其中原数据图片是tif格式，需要转换成png格式，具体代码见trainTif2Png.py。
* 数据增强主要[参考这个](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/294006)。使用该帖子数据增强配置，在原生数据上面，效果一般。5k次迭代，MAP: 0.11284。


   
