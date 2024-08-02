import os
import argparse
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.models import build_network, build_body_model

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

args = parser.parse_args()

cfg = get_cfg_defaults()
cfg.merge_from_file("configs/yamls/demo.yaml")

print(cfg)

print(args)
