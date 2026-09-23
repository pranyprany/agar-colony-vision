"""Convert AGAR's per-image JSON annotations into YOLO-format .txt labels."""
import argparse
import json
import os

from PIL import Image

CLASSES = ["S.aureus", "B.subtilis", "P.aeruginosa", "C.albicans", "E.coli"]


def convert_image(json_path, img_path, txt_path):
    with Image.open(img_path) as img:
        w, h = img.size
    with open(json_path, "r") as f:
        labels = json.load(f)["labels"]

    lines = []
    for item in labels:
        try:
            class_no = CLASSES.index(item["class"])
        except ValueError:
            continue  # label outside the 5 tracked species (e.g. defects/contamination)

        x1, y1 = item["x"], item["y"]
        x2, y2 = x1 + item["width"], y1 + item["height"]
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width_r = (x2 - x1) / w
        height_r = (y2 - y1) / h
        lines.append(f"{class_no} {x_center} {y_center} {width_r} {height_r}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input_dir", required=True, help="directory of AGAR <id>.jpg/<id>.json pairs")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to write YOLO <id>.txt labels")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    json_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".json"))

    for json_name in json_files:
        image_id = os.path.splitext(json_name)[0]
        json_path = os.path.join(args.input_dir, json_name)
        img_path = os.path.join(args.input_dir, image_id + ".jpg")
        if not os.path.exists(img_path):
            continue
        txt_path = os.path.join(args.output_dir, image_id + ".txt")
        convert_image(json_path, img_path, txt_path)
        print(f"processed {image_id}")

    print("done")


if __name__ == "__main__":
    main()
