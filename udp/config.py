# encoding: utf-8
"""
@author: Yuanhao Cai
@date:  2020.03
"""

import os, getpass
import os.path as osp
import argparse
import sys

from easydict import EasyDict as edict
from dataset.attribute import load_dataset
from yacs.config import CfgNode as CN

class Config:
    # -------- Directoy Config -------- #
    USER = getpass.getuser()
    # ROOT_DIR = os.environ['RSN_HOME']
    ROOT_DIR = 'D:/anaconda3/envs/ak/UDP-RSN/'
    OUTPUT_DIR = osp.join(ROOT_DIR, 'model_logs',
            osp.split(osp.split(osp.realpath(__file__))[0])[1]) #RSNCOCO
    TEST_DIR = osp.join(OUTPUT_DIR, 'test_dir')
    DEMO_DIR = osp.join(OUTPUT_DIR, 'demo_dir')
    TENSORBOARD_DIR = osp.join(OUTPUT_DIR, 'tb_dir')

    # -------- Data Config -------- #
    DATALOADER = edict()
    DATALOADER.NUM_WORKERS = 0
    DATALOADER.ASPECT_RATIO_GROUPING = False
    DATALOADER.SIZE_DIVISIBILITY = 0

    DATASET = edict()
    DATASET.NAME = 'Demo'
    dataset = load_dataset(DATASET.NAME)
    DATASET.KEYPOINT = dataset.KEYPOINT

    INPUT = edict()
    INPUT.NORMALIZE = True
    INPUT.MEANS = [0.406, 0.456, 0.485] # bgr
    INPUT.STDS = [0.225, 0.224, 0.229]

    # edict will automatcally convert tuple to list, so ..
    INPUT_SHAPE = dataset.INPUT_SHAPE
    OUTPUT_SHAPE = dataset.OUTPUT_SHAPE

    # -------- Model Config -------- #
    MODEL = edict()
    MODEL.BACKBONE = 'Res-50'
    MODEL.UPSAMPLE_CHANNEL_NUM = 256
    MODEL.STAGE_NUM = 1
    MODEL.OUTPUT_NUM = DATASET.KEYPOINT.NUM
    MODEL.DEVICE = 'cuda'
    MODEL.WEIGHT = None
    # -------- Loss Config -------- #
    LOSS = edict()
    LOSS.OHKM = True
    LOSS.TOPK = 8
    LOSS.COARSE_TO_FINE = True
    RUN_EFFICIENT = False
    # -------- Test Config -------- #
    TEST = dataset.TEST
    TEST.IMS_PER_GPU = 32

    # -------- Demo Config -------- #

    # -------- Cudnn Config -------- #
    CUDNN = CN()
    CUDNN.BENCHMARK = True
    CUDNN.DETERMINISTIC = False
    CUDNN.ENABLED = True


config = Config()
cfg = config



def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-log', '--linklog', default=False, action='store_true')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if args.linklog:
        link_log_dir()
