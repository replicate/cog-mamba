import subprocess
import glob
import os.path as osp

def download_and_extract(url, dest):
    try:
        print(f"Downloading {url}...")
        output = subprocess.check_output(["pget", "-x", url, dest], close_fds=False)
    except subprocess.CalledProcessError as e:
        # If download fails, clean up and re-raise exception
        raise e

