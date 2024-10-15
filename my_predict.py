# import os
import argparse

# from glob import glob
# from collections import defaultdict

# import cv2
# import torch
# import joblib
# import numpy as np
# from loguru import logger
# from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.models import build_network, build_body_model

# from lib.models.preproc.detector import DetectionModel
# from lib.models.preproc.extractor import FeatureExtractor
# from lib.data.datasets import CustomDataset
# from lib.utils.imutils import avg_preds
# from lib.models.smplify import TemporalSMPLify
# from lib.utils.transforms import matrix_to_axis_angle

parser = argparse.ArgumentParser()

parser.add_argument(
    "--video",
    type=str,
    default="videos/madfit.mp4",
    help="input video path or youtube link",
)

parser.add_argument(
    "--output_pth",
    type=str,
    default="output/demo",
    help="output folder to write results",
)

parser.add_argument(
    "--calib", type=str, default=None, help="Camera calibration file path"
)

parser.add_argument(
    "--estimate_local_only",
    action="store_true",
    help="Only estimate motion in camera coordinate if True",
)

parser.add_argument(
    "--visualize", action="store_true", help="Visualize the output mesh if True"
)

parser.add_argument("--save_pkl", action="store_true", help="Save output as pkl file")

parser.add_argument(
    "--run_smplify",
    action="store_true",
    help="Run Temporal SMPLify for post processing",
)

# Namespace(video='videos/madfit.mp4', output_pth='output/demo', calib=None, estimate_local_only=False, visualize=False, save_pkl=False, run_smplify=False)
args = parser.parse_args()

cfg = get_cfg_defaults()
cfg.merge_from_file("configs/yamls/demo.yaml")
"""
cfg as below:
DATASET:
  RATIO: [1.0, 0, 0, 0, 0]
  SEQLEN: 81
DEBUG: False
DEVICE: cuda
EVAL: False
EXP_NAME: demo
FLIP_EVAL: True
LOGDIR:
LOSS:
  CAMERA_LOSS_SKIP_EPOCH: 5
  CAMERA_LOSS_WEIGHT: 0.04
  CASCADED_LOSS_WEIGHT: 0.0
  CONTACT_LOSS_WEIGHT: 0.04
  JOINT2D_LOSS_WEIGHT: 5.0
  JOINT3D_LOSS_WEIGHT: 5.0
  LOSS_WEIGHT: 60.0
  POSE_LOSS_WEIGHT: 1.0
  ROOT_POSE_LOSS_WEIGHT: 0.4
  ROOT_VEL_LOSS_WEIGHT: 0.001
  SHAPE_LOSS_WEIGHT: 0.001
  SLIDING_LOSS_WEIGHT: 0.5
  VERTS3D_LOSS_WEIGHT: 1.0
MODEL:
  BACKBONE: vit
MODEL_CONFIG: configs/yamls/model_base.yaml
NUM_WORKERS: 0
OUTPUT_DIR: experiments/
RESUME: False
SEED_VALUE: -1
SUMMARY_ITER: 50
TITLE: default
TRAIN:
  BATCH_SIZE: 64
  CHECKPOINT: checkpoints/wham_vit_bedlam_w_3dpw.pth.tar
  DATASET_EVAL: 3dpw
  END_EPOCH: 999
  LR: 0.0003
  LR_DECAY_RATIO: 0.1
  LR_FINETUNE: 5e-05
  LR_PATIENCE: 5
  MILESTONES: [50, 70]
  MOMENTUM: 0.9
  OPTIM: Adam
  STAGE: stage2
  START_EPOCH: 0
  WD: 0.0
"""

# default `smpl_batch_size` is 64 * 81 = 5184
smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
smpl = build_body_model(cfg.DEVICE, smpl_batch_size)

print(smpl)

# network = build_network(cfg, smpl)
# network.eval()
