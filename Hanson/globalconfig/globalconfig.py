import argparse
import os 
from mmcv import Config, DictAction

parser = argparse.ArgumentParser(description='Sartorius')
##basic params
parser.add_argument('--config', type = str,default='hanson_config/cascade_mask_rcnn_r50_fpn.py',help='train config file path')
parser.add_argument('--work-dir', type = str,default= 'logfile',help='the dir to save logs and models')
parser.add_argument(
    '--resume-from', help='the checkpoint file to resume from')
parser.add_argument(
    '--no-validate',
    action='store_true',
    help='whether not to evaluate the checkpoint during training')
group_gpus = parser.add_mutually_exclusive_group()
group_gpus.add_argument(
    '--gpus',
    type=int,
    help='number of gpus to use '
    '(only applicable to non-distributed training)')
group_gpus.add_argument(
    '--gpu-ids',
    type=int,
    nargs='+',
    help='ids of gpus to use '
    '(only applicable to non-distributed training)')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument(
    '--deterministic',
    action='store_true',
    help='whether to set deterministic options for CUDNN backend.')
parser.add_argument(
    '--options',
    nargs='+',
    action=DictAction,
    help='override some settings in the used config, the key-value pair '
    'in xxx=yyy format will be merged into config file (deprecate), '
    'change to --cfg-options instead.')
parser.add_argument(
    '--cfg-options',
    nargs='+',
    action=DictAction,
    help='override some settings in the used config, the key-value pair '
    'in xxx=yyy format will be merged into config file. If the value to '
    'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    'Note that the quotation marks are necessary and that no white space '
    'is allowed.')
parser.add_argument(
    '--launcher',
    choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none',
    help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)

##dataset
parser.add_argument('--resize-w', type=int, default=640)
parser.add_argument('--resize-h', type=int, default=480)
parser.add_argument('--samples-per-gpu', type=int, default=2)
parser.add_argument('--workers-per-gpu', type=int, default=2)

args = parser.parse_args()