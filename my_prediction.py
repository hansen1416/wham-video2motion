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


# # Namespace(video='videos/madfit.mp4', output_pth='output/demo', calib=None, estimate_local_only=False, visualize=False, save_pkl=False, run_smplify=False)
# args = argparse.Namespace(
#     video="videos/madfit1.mp4",
#     output_pth="output/demo",
#     calib=None,
#     estimate_local_only=False,
#     visualize=False,
#     save_pkl=False,
#     run_smplify=False,
# )


cfg = get_cfg_defaults()
cfg.merge_from_file("configs/yamls/demo.yaml")

if torch.cuda.is_available():
    cfg.DEVICE = "cuda"
else:
    cfg.DEVICE = "cpu"

# print(cfg)

video_name = "madfit1.mp4"
video_path = os.path.join("videos", video_name)

output_pth = os.path.join(".", "output", video_name)

if not os.path.exists(output_pth):
    os.makedirs(output_pth)


cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), f"Faild to load video file {video_path}"

fps = cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


with torch.no_grad():

    # `cfg.DEVICE.lower()` cuda
    detector = DetectionModel(cfg.DEVICE.lower())
    # `cfg.DEVICE.lower()` cuda, `cfg.FLIP_EVAL` True
    extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)

    print(detector)
    print(extractor)
