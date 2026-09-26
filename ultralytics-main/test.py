"""
YOLO model test script - verify model generalization ability

Usage:
python ultralytics-main/test.py --conf 0.76 --weights runs/detect/train5/weights/best.pt --data ultralytics-main/hgc27-mgc803.yaml --img 640

Parameter descriptions:
--weights: trained model weights path
--conf: confidence threshold
--data: dataset config file path
--img: input image size
"""

import argparse
from ultralytics import YOLO
import os

def main():
    # Argument config
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='model weights path (required)')
    parser.add_argument('--conf', type=float, default=0.76, help='confidence threshold')
    parser.add_argument('--data', type=str, default='ultralytics-main/hgc27-mgc803.yaml', help='dataset config file path')
    parser.add_argument('--img', type=int, default=640, help='input image size')

    args = parser.parse_args()

    # Load the model
    model = YOLO(args.weights)

    # Test the model
    results = model.val(
        data=args.data,
        conf=args.conf,
        imgsz=args.img,
        split='test',  # use the test set to verify generalization
        verbose=True   # output detailed evaluation metrics
    )

    # Print key evaluation metrics
    print(f"\nModel generalization evaluation results:")
    print(f"mAP@0.5: {float(results.box.map):.4f}")
    print(f"mAP@0.5:0.95: {float(results.box.map):.4f}")
    #print(f"Precision: {float(results.box.p):.4f}")
    #print(f"Recall: {float(results.box.r):.4f}")

if __name__ == '__main__':
    main()