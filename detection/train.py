"""Train a YOLOv8 detector on the AGAR-derived colony dataset."""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--data", default="data.yaml", help="path to YOLO data.yaml")
    parser.add_argument("-m", "--model", default="yolov8n.pt", help="base model or checkpoint to start from")
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
