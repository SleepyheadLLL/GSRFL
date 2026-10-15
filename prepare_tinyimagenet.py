"""Downloads Tiny-ImageNet-200 and sorts its validation images into per-class folders."""

import argparse
import os
import shutil
import urllib.request
import zipfile

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def prepare(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    root = os.path.join(data_dir, "tiny-imagenet-200")
    archive = os.path.join(data_dir, "tiny-imagenet-200.zip")
    if not os.path.isdir(root):
        if not os.path.isfile(archive):
            urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_dir)
    val_sorted = os.path.join(root, "val_sorted")
    if os.path.isdir(val_sorted) and os.listdir(val_sorted):
        return
    with open(os.path.join(root, "val", "val_annotations.txt")) as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            dst = os.path.join(val_sorted, parts[1])
            os.makedirs(dst, exist_ok=True)
            src = os.path.join(root, "val", "images", parts[0])
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst, parts[0]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="./data")
    prepare(ap.parse_args().data_dir)
