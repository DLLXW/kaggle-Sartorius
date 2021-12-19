# 主要记录数据和后处理方面:

## 数据:
  * 数据标注存在Broken masks, 详细参考[这个](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/278801)。
  * 对于Broken masks, 进行easy fix, fixed之后, 细胞形态会有差异, 详细参考[这个](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/279488)。
  * clean astro mask, 详细参考[这个](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/291371)。
  * experiment with cleaned astro masks, 清洗数据和未清洗数据, CV榜单有差距, LB榜单没有差距，详细参考[这个](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/291639)。
## 后处理:
  * MMDetection Neuron Inference NMS Improvement, 详细参考[这个](https://www.kaggle.com/zzhnku/mmdetection-neuron-inference-nms-improvement)。
  
