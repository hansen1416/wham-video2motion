import os
import argparse
from collections import defaultdict

import cv2
import torch
import joblib
import numpy as np
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.models.smplify import TemporalSMPLify
from lib.utils.transforms import matrix_to_axis_angle


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
