import os
import subprocess

target_folder = os.path.join("/tram", "videos")
results_folder = os.path.join("/tram", "results")

# list all files in `target_folder`
files = os.listdir(target_folder)

print(f"got {len(files)} files")

for f in files:

    file_name = f.split(".")[0]

    # check if the results already exists
    if os.path.exists(os.path.join(results_folder, file_name)):
        print(f"{f} already processed")
        continue
    else:
        print(f"{f} in process")

    video_path = os.path.join(target_folder, f)

    # List of commands to run
    command = (
        [
            "docker",
            "run",
            "-v",
            ".:/code/",
            "--rm",
            "yusun9/wham-vitpose-dpvo-cuda11.3-python3.9",
            "python",
            "demo.py",
            "--video",
            video_path,
        ],
    )

    subprocess.run(command, check=True)
