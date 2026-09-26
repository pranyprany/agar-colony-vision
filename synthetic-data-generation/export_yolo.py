"""Convert generated synthetic dishes (grow_colonies.py output) into a YOLO detection dataset."""
import argparse
import json
import os
import random
import shutil

import numpy as np
from PIL import Image

# AGAR class id -> index in the detector's class list
# ["S.aureus", "B.subtilis", "P.aeruginosa", "C.albicans", "E.coli"]
AGAR_TO_YOLO = {"1": 0, "2": 1, "3": 2, "6": 3, "4": 4}
MIN_SIDE_PX = 12
MIN_MASK_COVER = 0.10


def convert_boxes(labels, mask, w, h):
    """Generated JSON stores x/width along image rows and y/height along columns; swap to standard."""
    lines = []
    for b in labels:
        cls = AGAR_TO_YOLO.get(str(b["class"]))
        if cls is None:
            continue
        x1, y1 = b["y"], b["x"]
        x2, y2 = x1 + b["height"], y1 + b["width"]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
            continue
        if mask is not None and (mask[y1:y2, x1:x2] > 0).mean() < MIN_MASK_COVER:
            continue  # box without a visible colony
        lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--generated_dir", required=True)
    ap.add_argument("-o", "--output_dir", required=True)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stems = sorted(f[:-5] for f in os.listdir(args.generated_dir) if f.endswith(".json"))
    dishes = sorted({s.rsplit("_", 2)[0] for s in stems})  # split by source dish, not by crop
    random.Random(args.seed).shuffle(dishes)
    n_val = max(1, int(len(dishes) * args.val_fraction)) if len(dishes) > 1 else 0
    val_dishes = set(dishes[:n_val])

    for split in ("train", "valid"):
        os.makedirs(os.path.join(args.output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "labels"), exist_ok=True)

    kept = dropped = 0
    for stem in stems:
        png = os.path.join(args.generated_dir, stem + ".png")
        if not os.path.exists(png):
            continue
        with open(os.path.join(args.generated_dir, stem + ".json")) as f:
            labels = json.load(f)["labels"]
        npy = os.path.join(args.generated_dir, stem + ".npy")
        mask = np.load(npy) if os.path.exists(npy) else None
        w, h = Image.open(png).size
        lines = convert_boxes(labels, mask, w, h)
        dropped += len(labels) - len(lines)
        kept += len(lines)
        split = "valid" if stem.rsplit("_", 2)[0] in val_dishes else "train"
        shutil.copy(png, os.path.join(args.output_dir, split, "images", stem + ".png"))
        with open(os.path.join(args.output_dir, split, "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(lines))

    print(f"boxes kept: {kept}, dropped (tiny/empty): {dropped}; dishes: {len(dishes)} ({n_val} to valid)")


if __name__ == "__main__":
    main()
