# Auto Train With Fitlog

## 简介

**该工程主要方便设置超参自动训练，同时借助fitlog记录详细实验结果来分析哪个超参对实验结果起到积极作用**



## 使用方式

- **修改train.py文件，建议直接拷贝该文件。该文件在原train.py文件中把参数定义部分剥离，同时定义了fitlog**
- **原train.py文件中的参数定义，放在了globalconfi/globalconfig.py文件中定义，同时可以在上上面定义一些自己想改变的超参，最终这些超参会传递给模型配置文件(我这里hanson_config/cascade_mask_rcnn_r50_fpn.py)**
- **最后训练使用main.py，在main.py里面可以设置自己的超参组**
- **result保存在logfile下面的文件夹下**
- **如果想查看结果，可以在更目录使用fitlog log logfile命令**

## 补充

**在utils文件夹下面有划为数据集的文件**
