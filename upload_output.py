import os
import sys
import pathlib
from typing import List

import oss2
from oss2 import SizedFileAdapter, determine_part_size
from oss2.models import PartInfo
from oss2.credentials import EnvironmentVariableCredentialsProvider
from dotenv import load_dotenv

load_dotenv()


def folder_uploader_sync(folder_path, bucket_name, oss_endpoint):

    folder_path = os.path.abspath(folder_path)

    if not os.path.exists(folder_path):
        print(f"{folder_path} does not exist")
        sys.exit(1)

    path_delimiter = os.path.sep
    all_oos_path = []

    # get all files in the folder recursively
    all_files: List[str] = [
        str(f) for f in pathlib.Path(folder_path).rglob("*") if f.is_file()
    ]

    for filepath in all_files:

        # remove the prefix `folder_path` from the file path
        sub_path = filepath[len(folder_path) + 1 :]

        all_oos_path.append("/".join(sub_path.split(path_delimiter)))

    # create a bucket
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, oss_endpoint, bucket_name, connect_timeout=30)

    # upload all files
    for i, filepath in enumerate(all_files):

        if not os.path.isfile(filepath):
            continue

        target_path = all_oos_path[i]

        # check if the file already exists in oss
        if bucket.object_exists(target_path):
            # print(f"{self.process_idx}: {target_path} already exists in oss")
            continue

        total_size = os.path.getsize(filepath)
        # Use the determine_part_size method to determine the part size.
        part_size = determine_part_size(total_size, preferred_size=30 * 1024 * 1024)

        upload_id = bucket.init_multipart_upload(target_path).upload_id

        parts = []

        # Upload the parts.
        with open(filepath, "rb") as fileobj:
            part_number = 1
            offset = 0
            while offset < total_size:
                num_to_upload = min(part_size, total_size - offset)
                # Use the SizedFileAdapter(fileobj, size) method to generate a new object and recalculate the position from which the append operation starts.
                result = bucket.upload_part(
                    target_path,
                    upload_id,
                    part_number,
                    SizedFileAdapter(fileobj, num_to_upload),
                )
                parts.append(PartInfo(part_number, result.etag))

                offset += num_to_upload
                part_number += 1

                # show progress
                print(
                    f"\r{i+1}/{len(all_files)}: {target_path} -> {bucket_name} {offset}/{total_size}",
                    end="",
                )

            print()

        headers = dict()

        bucket.complete_multipart_upload(target_path, upload_id, parts, headers=headers)


folder_uploader_sync(
    os.path.join("/root", "wham-video2motion", "output"),
    "wham-results",
    "oss-ap-southeast-1.aliyuncs.com",
)
