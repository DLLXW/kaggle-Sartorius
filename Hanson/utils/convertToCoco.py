import pycocotools
from pycocotools.coco import COCO
import cv2
import os
import numpy as np
import torch
baseImgPath = '/home/data1/hanfeng/datasets/kaggle/Sartorius'

coco = COCO('0_new_train_dataset.json')
ids = list(coco.imgToAnns.keys())
img_id = ids[0]
ann_ids = coco.getAnnIds(imgIds=img_id)
target = coco.loadAnns(ann_ids)
file_name = coco.loadImgs(img_id)[0]['file_name']
file_name = os.path.join(baseImgPath,file_name[file_name.find('/train') + 1:])
img = cv2.imread(file_name)
height, width, _ = img.shape
masks = [coco.annToMask(obj).reshape(-1) for obj in target]
masks = np.vstack(masks)
masks = masks.reshape(-1, height, width)
mask = torch.from_numpy(masks)
for m in range(masks.shape[0]):
    mask = masks[m]
    color = np.random.randint(0, 255)
    channel = np.random.randint(0, 3)
    y, x = np.where(mask == 1)
    img[y, x, channel] = color
cv2.imwrite('result.jpg',img)