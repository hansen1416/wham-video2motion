import joblib
import os

import numpy as np


def axis_angle_to_quaternion(axis_angle):

    # print(axis_angle)

    axis = axis_angle / np.linalg.norm(axis_angle)
    angle = np.linalg.norm(axis_angle)
    half_angle = angle / 2

    w = np.cos(half_angle)
    x, y, z = axis * np.sin(half_angle)

    return np.array([x, y, z, w])


output_pth = os.path.join(".", "output")

# Assuming output_pth is defined
output_file = os.path.join(output_pth, "madfit1", "wham_output.pkl")
loaded_results = joblib.load(output_file)

pose = loaded_results[0]["pose"]
trans = loaded_results[0]["trans"]
pose_world = loaded_results[0]["pose_world"]
trans_world = loaded_results[0]["trans_world"]
betas = loaded_results[0]["betas"]
verts = loaded_results[0]["verts"]
frame_ids = loaded_results[0]["frame_ids"]


for pose_frame in pose:
    # print(pose.shape)

    axis_angles = pose_frame.reshape((24, 3))

    print(axis_angles.shape)

    quaternions = np.apply_along_axis(axis_angle_to_quaternion, axis=1, arr=axis_angles)

    print(quaternions)
