import os
import argparse

# from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import numpy as np

# from loguru import logger
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

# check if cuda is available
if not torch.cuda.is_available():
    cfg.DEVICE = "cpu"

# default `smpl_batch_size` is 64 * 81 = 5184
smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
# SMPL(
#   Gender: NEUTRAL
#   Number of joints: 24
#   Betas: 10
#   (vertex_joint_selector): VertexJointSelector()
# )
smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
# here load the checkpoint
network = build_network(cfg, smpl)
network.eval()


video_name = "madfit"

video = os.path.join("videos", f"{video_name}.mp4")

cap = cv2.VideoCapture(video)
assert cap.isOpened(), f"Faild to load video file {video}"

fps = cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

output_pth = os.path.join(".", "output", video_name)

with torch.no_grad():

    if not (
        os.path.exists(os.path.join(output_pth, "tracking_results.pth"))
        and os.path.exists(os.path.join(output_pth, "slam_results.pth"))
    ):

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

        joblib.dump(tracking_results, os.path.join(output_pth, "tracking_results.pth"))
        joblib.dump(slam_results, os.path.join(output_pth, "slam_results.pth"))

    else:

        tracking_results = joblib.load(os.path.join(output_pth, "tracking_results.pth"))
        slam_results = joblib.load(os.path.join(output_pth, "slam_results.pth"))
        print(f"Already processed data exists at {output_pth} ! Load the data .")

    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)

    # run WHAM
    results = defaultdict(dict)

    print(f"lenth of dataset: {len(dataset)}")

    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                (
                    _id,
                    x,
                    inits,
                    features,
                    mask,
                    init_root,
                    cam_angvel,
                    frame_id,
                    kwargs,
                ) = flipped_batch
                flipped_pred = network(
                    x,
                    inits,
                    features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=True,
                    **kwargs,
                )

                # Forward pass with normal input
                batch = dataset.load_data(subj)
                (
                    _id,
                    x,
                    inits,
                    features,
                    mask,
                    init_root,
                    cam_angvel,
                    frame_id,
                    kwargs,
                ) = batch
                pred = network(
                    x,
                    inits,
                    features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=True,
                    **kwargs,
                )

                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred["pose"].squeeze(
                    0
                ), flipped_pred["betas"].squeeze(0)
                pose, shape = pred["pose"].squeeze(0), pred["betas"].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(
                    -1, 24, 6
                )
                avg_pose, avg_shape = avg_preds(
                    pose, shape, flipped_pose, flipped_shape
                )
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (
                    flipped_pred["contact"][..., [2, 3, 0, 1]] + pred["contact"]
                ) / 2

                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

            else:
                # data
                batch = dataset.load_data(subj)
                (
                    _id,
                    x,
                    inits,
                    features,
                    mask,
                    init_root,
                    cam_angvel,
                    frame_id,
                    kwargs,
                ) = batch

                # inference
                pred = network(
                    x,
                    inits,
                    features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=True,
                    **kwargs,
                )

        print(f"pred: {pred}")

        # if False:
        if args.run_smplify:
            smplify = TemporalSMPLify(
                smpl, img_w=width, img_h=height, device=cfg.DEVICE
            )
            input_keypoints = dataset.tracking_results[_id]["keypoints"]
            pred = smplify.fit(pred, input_keypoints, **kwargs)

            with torch.no_grad():
                network.pred_pose = pred["pose"]
                network.pred_shape = pred["betas"]
                network.pred_cam = pred["cam"]
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

        # ========= Store results ========= #
        pred_body_pose = (
            matrix_to_axis_angle(pred["poses_body"]).cpu().numpy().reshape(-1, 69)
        )
        pred_root = (
            matrix_to_axis_angle(pred["poses_root_cam"]).cpu().numpy().reshape(-1, 3)
        )
        pred_root_world = (
            matrix_to_axis_angle(pred["poses_root_world"]).cpu().numpy().reshape(-1, 3)
        )
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = (pred["trans_cam"] - network.output.offset).cpu().numpy()

        results[_id]["pose"] = pred_pose
        results[_id]["trans"] = pred_trans
        results[_id]["pose_world"] = pred_pose_world
        results[_id]["trans_world"] = pred["trans_world"].cpu().squeeze(0).numpy()
        results[_id]["betas"] = pred["betas"].cpu().squeeze(0).numpy()
        results[_id]["verts"] = (
            (pred["verts_cam"] + pred["trans_cam"].unsqueeze(1)).cpu().numpy()
        )
        results[_id]["frame_ids"] = frame_id

    if True:
        joblib.dump(results, os.path.join(output_pth, "wham_output.pkl"))
