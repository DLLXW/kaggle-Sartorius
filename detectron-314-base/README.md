### 训练/验证数据
大家可以使用[这个](https://www.kaggle.com/slawekbiel/positive-score-with-detectron-2-3-training/data)
这样我们可以保持一致，而且这个验证集和线上一致性很好
- cv: 27 LB: 304
- cv: 28 LB: 311 test short-edge:800
- cv: 28  LB: 314  test short-edge:1040

**关于预训练数据**

大家可以参考使用[这个](https://www.kaggle.com/markunys/sartorius-transfer-learning-train-with-livecell/data)，预训练得多迭代一些才能看到明显的效果
### 训练配置
- 多尺度训练:
```python
cfg.INPUT.MIN_SIZE_TRAIN = (650, 780, 910, 1040)
cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
```

- train.py: 该脚本可以放到detectron2/tools/ 目录下
```bash
#因为我是远程服务器，所以nohup这样可以保证程序在后台运行，输出结果保存在out_1215.out文件里面，可随时打开查看训练情况
nohup python train.py >out_1215.out 2>&1 & 
```
- train_livecell.py 预训练livecell数据集的脚本

### 核心的配置
```python
cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
cfg.OUTPUT_DIR="output_1215/"
cfg.DATASETS.TRAIN = ("sartorius_train",)
cfg.DATASETS.TEST = ("sartorius_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = "./output_livecell_x1/model_0006503.pth"#livecell预训练的模型

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.INPUT.CROP.ENABLED = True
# 开启半精度训练
# cfg.SOLVER.AMP.ENABLED = True

cfg.SOLVER.BASE_LR = 0.005
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = (5000, 7500, 9000)
# cfg.SOLVER.GAMMA = 0.9
# # 优化器功能、权重衰减、学习率衰减
# cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 0.01
# 训练之前，会做一个热身运动， 学习率慢慢增加初始学习率
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
cfg.SOLVER.WARMUP_ITERS = 200
#多尺度训练
cfg.INPUT.MIN_SIZE_TRAIN = (650, 780, 910, 1040)
cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
cfg.TEST.AUG.ENABLED=True
cfg.TEST.AUG.FLIP=True
cfg.TEST.AUG.MAX_SIZE=1408
cfg.TEST.AUG.MIN_SIZES=(650, 780, 910, 1040)
#
#cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
## Options are: "smooth_l1", "giou", "diou", "ciou"
#cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.35

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
cfg.SOLVER.CHECKPOINT_PERIOD = len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch
cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch
cfg.TEST.AUG.FLIP = True
```