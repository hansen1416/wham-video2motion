import os
import sys
import shutil

import oss2
from oss2 import SizedFileAdapter, determine_part_size
from oss2.models import PartInfo
from oss2.credentials import EnvironmentVariableCredentialsProvider
from dotenv import load_dotenv

load_dotenv()


def percentage(consumed_bytes, total_bytes):
    if total_bytes:
        rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
        # rate表示下载进度。
        print("\r{0}% ".format(rate), end="")

        sys.stdout.flush()


def folder_download_sync(bucket_name, oss_endpoint, target_path="./"):
    # create a bucket
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, oss_endpoint, bucket_name, connect_timeout=30)

    # 列举Bucket下的所有文件。
    for obj in oss2.ObjectIterator(bucket):
        # object_stream = bucket.get_object(obj.key)

        oss_path_arr = obj.key.split("/")

        local_path = os.path.join(target_path, *oss_path_arr)

        # check if the file already exists in oss
        if os.path.exists(local_path):
            print(f"{local_path} already exists in local")
            continue

        # create the folder if it does not exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # with open(local_path, "wb") as local_fileobj:
        #     shutil.copyfileobj(object_stream, local_fileobj)

        print(f"downloading {obj.key} to {local_path}")

        # download the file
        bucket.get_object_to_file(obj.key, local_path, progress_callback=percentage)


folder_download_sync(
    "workout-videos",
    "oss-ap-southeast-1.aliyuncs.com",
    "videos",
)
