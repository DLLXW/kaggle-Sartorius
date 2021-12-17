#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import detectron2
from pathlib import Path
import random, cv2, os
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader, DatasetMapper
import detectron2.data.transforms as T

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.structures import polygons_to_bitmask
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
# from detectron2.engine import BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer

dataDir=Path('/tmp/pycharm_project_957/sartorius-cell-instance-segmentation/LIVECell_dataset_2021/images/')


def polygon_to_rle(polygon, shape=(520, 704)):
    # print(polygon)
    mask = polygons_to_bitmask([np.asarray(polygon) + 0.25], shape[0], shape[1])

    rle = mask_util.encode(np.asfortranarray(mask))
    return rle

# Taken from https://www.kaggle.com/theoviel/competition-metric-map-iou
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)


def score(pred, targ):
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x: x['segmentation'], targ))
    enc_targs = [polygon_to_rle(enc_targ[0]) for enc_targ in enc_targs]
    ious = mask_util.iou(enc_preds, enc_targs, [0] * len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)


class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']: item['annotations'] for item in dataset_dicts}

    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, targ))

    def evaluate(self):
        return {"MaP IoU": np.mean(self.scores)}


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)

    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg,
                                            mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                                                T.RandomBrightness(0.9, 1.1),
                                                T.RandomContrast(0.9, 1.1),
                                                T.RandomSaturation(0.9, 1.1),
                                                T.RandomLighting(0.9),
                                                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                                                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                                            ]))
    # def build_hooks(self):
    #     # copy of cfg
    #     cfg = self.cfg.clone()
    #
    #     # build the original model hooks
    #     hooks = super().build_hooks()
    #
    #     # add the best checkpointer hook
    #     hooks.insert(-1, BestCheckpointer(cfg.TEST.EVAL_PERIOD,
    #                                       DetectionCheckpointer(self.model, cfg.OUTPUT_DIR),
    #                                       "MaP IoU",
    #                                       "max",
    #                                       ))
    #     return hooks

def setup(cfg):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    register_coco_instances('sartorius_train', {}, '/tmp/pycharm_project_957/livecell_annotations_train.json', dataDir)
    register_coco_instances('sartorius_val', {}, '/tmp/pycharm_project_957/livecell_annotations_val.json', dataDir)
    register_coco_instances('sartorius_test', {}, '/tmp/pycharm_project_957/livecell_annotations_test.json', dataDir)
    metadata = MetadataCatalog.get('sartorius_train')
    print("metadata:", metadata)
    train_ds = DatasetCatalog.get('sartorius_train')

    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
    cfg.DATASETS.TRAIN = ("sartorius_train", "sartorius_test")
    cfg.DATASETS.TEST = ("sartorius_val",)
    # 输出权重
    cfg.OUTPUT_DIR = "output_livecell/"
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.AMP.ENABLED = True
    cfg.INPUT.MIN_SIZE_TRAIN = (680, )
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.SOLVER.MAX_ITER = 80000
    cfg.SOLVER.STEPS = (50000, 70000, 76000)
    cfg.SOLVER.CHECKPOINT_PERIOD = (len(DatasetCatalog.get('sartorius_train')) + len(
        DatasetCatalog.get('sartorius_test'))) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
    cfg.TEST.EVAL_PERIOD = (len(DatasetCatalog.get('sartorius_train')) + len(
        DatasetCatalog.get('sartorius_test'))) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch

    # cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('sartorius_train1')) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch

    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()
    # default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.num_gpus = 4
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
