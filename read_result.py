import joblib
import os

import numpy as np

video_name = "madfit3"

output_pth = os.path.join(".", "output", video_name)

# Assuming output_pth is defined
output_file = os.path.join(output_pth, "wham_output.pkl")
loaded_results = joblib.load(output_file)

# dict_keys(['pose', 'trans', 'pose_world', 'trans_world', 'betas', 'verts', 'frame_ids'])
# print(loaded_results[0].keys())

pose = loaded_results[0]["pose"]
trans = loaded_results[0]["trans"]
pose_world = loaded_results[0]["pose_world"]
trans_world = loaded_results[0]["trans_world"]
betas = loaded_results[0]["betas"]
verts = loaded_results[0]["verts"]
frame_ids = loaded_results[0]["frame_ids"]

print(loaded_results)

print(f"pose", pose.shape)
print(f"trans", trans.shape)
print(f"pose_world", pose_world.shape)
print("trans_world", trans_world.shape)
print("betas", betas.shape)
print("verts", verts.shape)
print("frame_ids", frame_ids.shape)


def save2bin(data, filename):

    dirname = os.path.dirname(filename)

    # if dir does not exist, create it
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    data = np.array(data, dtype=np.float32)
    with open(filename, "wb") as f:
        f.write(data.tobytes())


# save 'pose', 'trans', 'pose_world', 'trans_world' to files
save2bin(pose, os.path.join("results", video_name, "pose.bin"))
save2bin(trans, os.path.join("results", video_name, "trans.bin"))
save2bin(pose_world, os.path.join("results", video_name, "pose_world.bin"))
save2bin(trans_world, os.path.join("results", video_name, "trans_world.bin"))


"""
'Pelvis', 0
'L_Hip', 1
'R_Hip', 2
'Spine1', 3
'L_Knee', 4
'R_Knee', 5
'Spine2', 6
'L_Ankle', 7
'R_Ankle', 8
'Spine3', 9
'L_Foot', 10
'R_Foot', 11
'Neck', 12
'L_Collar', 13
'R_Collar', 14
'Head', 15
'L_Shoulder', 16
'R_Shoulder', 17
'L_Elbow', 18
'R_Elbow', 19
'L_Wrist', 20
'R_Wrist', 21
'L_Hand', 22
'R_Hand', 23
"""

joints_mapping = {
    0: "Pelvis",
    1: "L_Hip",
    2: "R_Hip",
    3: "Spine1",
    4: "L_Knee",
    5: "R_Knee",
    6: "Spine2",
    7: "L_Ankle",
    8: "R_Ankle",
    9: "Spine3",
    10: "L_Foot",
    11: "R_Foot",
    12: "Neck",
    13: "L_Collar",
    14: "R_Collar",
    15: "Head",
    16: "L_Shoulder",
    17: "R_Shoulder",
    18: "L_Elbow",
    19: "R_Elbow",
    20: "L_Wrist",
    21: "R_Wrist",
    22: "L_Hand",
    23: "R_Hand",
}
