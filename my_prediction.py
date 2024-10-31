import os
import argparse
from collections import defaultdict
from typing import Tuple
import struct

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
args = argparse.Namespace(
    video="videos/madfit1.mp4",
    output_pth="output/demo",
    calib=None,
    estimate_local_only=False,
    visualize=False,
    save_pkl=False,
    run_smplify=False,
)


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


def get_tracks_slam_results(
    video_path, output_pth, cfg, video_length, video_capture, video_fps
) -> Tuple[defaultdict, np.ndarray]:

    if os.path.exists(
        os.path.join(output_pth, "tracking_results.pth")
    ) and os.path.exists(os.path.join(output_pth, "slam_results.pth")):

        tracking_results = joblib.load(os.path.join(output_pth, "tracking_results.pth"))
        slam_results = joblib.load(os.path.join(output_pth, "slam_results.pth"))
        print(f"Already processed data exists at {output_pth} ! Load the data .")

        return tracking_results, slam_results

    with torch.no_grad():

        # `cfg.DEVICE.lower()` cuda
        detector = DetectionModel(cfg.DEVICE.lower())
        # `cfg.DEVICE.lower()` cuda, `cfg.FLIP_EVAL` True
        extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)

        slam = None

        bar = Bar("Preprocess: 2D detection and SLAM", fill="#", max=video_length)

        while video_capture.isOpened():
            flag, img = video_capture.read()
            if not flag:
                break

            # 2D detection and tracking
            detector.track(img, video_fps, video_length)

            # SLAM
            if slam is not None:
                slam.track()

            bar.next()

        tracking_results = detector.process(video_fps)

        if slam is not None:
            slam_results = slam.process()
        else:
            slam_results = np.zeros((video_length, 7))
            slam_results[:, 3] = 1.0  # Unit quaternion

        # Extract image features
        # TODO: Merge this into the previous while loop with an online bbox smoothing.
        tracking_results = extractor.run(video_path, tracking_results)

        # print(tracking_results)

        joblib.dump(tracking_results, os.path.join(output_pth, "tracking_results.pth"))
        joblib.dump(slam_results, os.path.join(output_pth, "slam_results.pth"))

        return tracking_results, slam_results


video_capture = cv2.VideoCapture(video_path)
assert video_capture.isOpened(), f"Faild to load video file {video_path}"

video_fps = video_capture.get(cv2.CAP_PROP_FPS)
video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
video_width, video_height = video_capture.get(
    cv2.CAP_PROP_FRAME_WIDTH
), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)


tracking_results, slam_results = get_tracks_slam_results(
    video_path, output_pth, cfg, video_length, video_capture, video_fps
)


def motion_prediction(cfg, network, dataset, output_pth):

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

        # print(f"pred: {pred}")

        # if args.run_smplify:
        #     smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
        #     input_keypoints = dataset.tracking_results[_id]["keypoints"]
        #     pred = smplify.fit(pred, input_keypoints, **kwargs)

        #     with torch.no_grad():
        #         network.pred_pose = pred["pose"]
        #         network.pred_shape = pred["betas"]
        #         network.pred_cam = pred["cam"]
        #         output = network.forward_smpl(**kwargs)
        #         pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

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

    joblib.dump(results, os.path.join(output_pth, "wham_output.pkl"))

    return results


smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
network = build_network(cfg, smpl)
network.eval()


if not os.path.exists(os.path.join(output_pth, "wham_output.pkl")):

    # print(tracking_results[0].keys())
    # print(slam_results.shape)

    # Build dataset
    dataset = CustomDataset(
        cfg, tracking_results, slam_results, video_width, video_height, video_fps
    )

    # `results` is a dictionary, the 1st layer is number of subjects,
    # the 2nd layer include keys ('pose', 'trans', 'pose_world', 'trans_world', 'betas', 'verts', 'frame_ids')
    # `pose`: (n_frames, 72), `trans`: (n_frames, 3), `pose_world`: (n_frames, 72), `trans_world`: (n_frames, 3)
    # `betas`: (n_frames, 10), `verts`: (n_frames, 6890, 3), `frame_ids`: (n_frames,)
    # the 6890 in `verts` is the number of vertices in SMPL model
    results = motion_prediction(cfg, network, dataset, output_pth)

    # save smpl.faces to local file
    np.save(os.path.join(output_pth, "faces.npy"), smpl.faces)

else:
    results = joblib.load(os.path.join(output_pth, "wham_output.pkl"))


# pose = results[0]["pose"]
# trans = results[0]["trans"]
# pose_world = results[0]["pose_world"]
# trans_world = results[0]["trans_world"]
# betas = results[0]["betas"]
# all_vertices = results[0]["verts"]
# frame_ids = results[0]["frame_ids"]

# print(
#     f"pose: {pose.shape}, trans: {trans.shape}, pose_world: {pose_world.shape}, trans_world: {trans_world.shape}, \
#       betas: {betas.shape}, verts: {all_vertices.shape}, frame_ids: {frame_ids.shape}"
# )

# # print(all_vertices)

# all_vertices_data = all_vertices.flatten()

# file_path = os.path.join(
#     os.path.join(os.environ["USERPROFILE"], "Documents"), "all_vertices.bin"
# )

# # Open a binary file for writing
# with open(file_path, "wb") as binary_file:
#     for value in all_vertices_data:
#         # Pack the float64 value into binary format
#         packed_data = struct.pack("d", value)
#         # Write the packed data to the file
#         binary_file.write(packed_data)

# print(f"Data saved to {file_path}")

faces = np.array(smpl.faces)

# (13776, 3)
# print(faces.flatten().shape)

faces_filepath = os.path.join(
    os.path.join(os.environ["USERPROFILE"], "Documents"), "faces.bin"
)

with open(faces_filepath, "wb") as binary_file:
    for value in faces.flatten():
        # Pack the float64 value into binary format
        packed_data = struct.pack("L", value)
        # Write the packed data to the file
        binary_file.write(packed_data)

print(f"Faces saved to {faces_filepath}")

# # List to hold the read float64 values
# read_data = []

# # Open the binary file for reading
# with open(file_path, "rb") as binary_file:
#     while True:
#         # Read 8 bytes (size of float64)
#         bytes_data = binary_file.read(8)
#         if not bytes_data:
#             break  # End of file
#         # Unpack the bytes back to float64
#         value = struct.unpack("d", bytes_data)[0]
#         read_data.append(value)

# print("Data read from file:", read_data)

# # visual each frame from results
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# mesh = Poly3DCollection(verts[0][smpl.faces], alpha=0.9)
# face_color = (1.0, 1.0, 0.9)
# edge_color = (0.5, 0.5, 0)
# mesh.set_edgecolor(edge_color)
# mesh.set_facecolor(face_color)
# ax.add_collection3d(mesh)
# # ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color="r")

# # if plot_joints:
# #     ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
# plt.show()


# vertex_colors = np.ones([verts[0].shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

# tri_mesh = trimesh.Trimesh(verts[0], smpl.faces, vertex_colors=vertex_colors)

# mesh = pyrender.Mesh.from_trimesh(tri_mesh)

# scene = pyrender.Scene()
# scene.add(mesh)

# pyrender.Viewer(scene, use_raymond_lighting=True)

# for i in range(verts.shape[0]):
#     frame_verts = verts[i]

#     mesh.vertices = frame_verts

#     scene = pyrender.Scene()

#     scene.add(mesh)

#     pyrender.Viewer(scene, use_raymond_lighting=True)
