"""Run a trained colony detector live off a webcam feed."""
import argparse

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--weights", default="weights/best.pt", help="path to trained YOLO weights")
    parser.add_argument("-c", "--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--imgsz", type=int, default=480)
    args = parser.parse_args()

    model = YOLO(args.weights)
    print(model.names)
    cam = cv2.VideoCapture(args.camera)

    while True:
        success, frame = cam.read()
        if not success:
            break

        results = model.track(frame, conf=args.conf, imgsz=args.imgsz)
        cv2.putText(frame, f"Total: {len(results[0].boxes)}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Live Camera", results[0].plot())

        if cv2.waitKey(1) == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
