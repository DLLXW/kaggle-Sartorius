import os
from pickle import FALSE
import numpy as np
import random
import pynvml
import time
from tqdm import tqdm

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

pynvml.nvmlInit()
master_port = 5686
folds = [0]

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(3)
is_run = False

pbr = tqdm(folds)
for fold in pbr:
    if fold > 0:
        break
    while True:
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if is_run:
            time.sleep(100)
        else:
            time.sleep(10)
        if meminfo.used / 1000000. <= 1000:
            if is_run:
                time.sleep(100)
            else:
                time.sleep(10)
            if meminfo.used / 1000000. <= 1000:
                pbr.set_description(str(pbr))
                cmd = f"python -m torch.distributed.launch --master_port {master_port} --nproc_per_node=4 train.py \
                        --launcher pytorch --resize-w 960 --resize-h 720 --samples-per-gpu 1 --workers-per-gpu 1 &"
                os.system(cmd)
                is_run = True
                time.sleep(100)
                break