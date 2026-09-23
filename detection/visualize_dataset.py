"""Draw AGAR JSON annotations on top of their source image, for sanity-checking labels."""
import argparse
import json
import os

from PIL import Image, ImageDraw

CLASSES = ["S.aureus", "B.subtilis", "P.aeruginosa", "C.albicans", "E.coli"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_id", help="AGAR image id, e.g. 1287")
    parser.add_argument("-d", "--dataset_dir", default="dataset", help="directory containing <id>.jpg/<id>.json")
    parser.add_argument("-o", "--output", help="save annotated image here instead of showing it")
    args = parser.parse_args()

    img_path = os.path.join(args.dataset_dir, f"{args.image_id}.jpg")
    json_path = os.path.join(args.dataset_dir, f"{args.image_id}.json")

    img = Image.open(img_path)
    with open(json_path) as f:
        labels = json.load(f)["labels"]

    draw = ImageDraw.Draw(img)
    for item in labels:
        x, y, w, h = item["x"], item["y"], item["width"], item["height"]
        draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
        draw.text((x, y - 10), item["class"])

    if args.output:
        img.save(args.output)
        print(f"saved to {args.output}")
    else:
        img.show()


if __name__ == "__main__":
    main()
