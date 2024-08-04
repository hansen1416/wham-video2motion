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
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor

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

# print(cfg)
# print(args)

smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
network = build_network(cfg, smpl)
network.eval()

# print(network)

video = os.path.join("videos", "madfit1.mp4")

cap = cv2.VideoCapture(video)
assert cap.isOpened(), f"Faild to load video file {video}"

fps = cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

with torch.no_grad():

    # `cfg.DEVICE.lower()` cuda
    detector = DetectionModel(cfg.DEVICE.lower())
    # `cfg.DEVICE.lower()` cuda, `cfg.FLIP_EVAL` True
    extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)

    slam = None

    bar = Bar("Preprocess: 2D detection and SLAM", fill="#", max=length)

    while cap.isOpened():
        flag, img = cap.read()
        if not flag:
            break

        # 2D detection and tracking
        detector.track(img, fps, length)

        # SLAM
        if slam is not None:
            slam.track()

        bar.next()

    tracking_results = detector.process(fps)

    if slam is not None:
        slam_results = slam.process()
    else:
        slam_results = np.zeros((length, 7))
        slam_results[:, 3] = 1.0  # Unit quaternion

    # Extract image features
    # TODO: Merge this into the previous while loop with an online bbox smoothing.
    tracking_results = extractor.run(video, tracking_results)

    print(tracking_results)
