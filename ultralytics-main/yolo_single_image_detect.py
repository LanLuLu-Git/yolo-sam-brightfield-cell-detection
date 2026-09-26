import os
import cv2
import numpy as np
from ultralytics import YOLO


class YOLOSingleDetector:
    def __init__(self, weights_path, conf_thres=0.25, iou_thres=0.5, device='cuda'):
        self.weights_path = weights_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device if device and 'cuda' in device and __import__('torch').cuda.is_available() else 'cpu'
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights file does not exist: {self.weights_path}")
        model = YOLO(self.weights_path)
        print(f"Model loaded, using device: {model.device}")
        return model

    def detect(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        results = self.model(image,
                             conf=self.conf_thres,
                             iou=self.iou_thres,
                             device=self.device)

        return results[0], image

    def save_yolo_label(self, result, image_shape, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        height, width = image_shape[:2]
        labels = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height

                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        with open(save_path, 'w') as f:
            f.write('\n'.join(labels))

        print(f"YOLO label file saved: {save_path}")

    def save_detected_image(self, image, result, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        result_image = result.plot()

        cv2.imwrite(save_path, result_image)
        print(f"Detection result image saved: {save_path}")

    def run(self, image_path, output_dir='./output'):
        print(f"Start detecting image: {image_path}")

        result, image = self.detect(image_path)

        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        ext = os.path.splitext(image_name)[1]

        label_path = os.path.join(output_dir, base_name + '.txt')
        image_save_path = os.path.join(output_dir, base_name + '_detected' + ext)

        self.save_yolo_label(result, image.shape, label_path)
        self.save_detected_image(image, result, image_save_path)

        if result.boxes is not None and len(result.boxes) > 0:
            print(f"Detection complete! Detected {len(result.boxes)} objects in total")
            for i, box in enumerate(result.boxes):
                class_name = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"  Object {i+1}: {class_name} (confidence: {conf:.4f})")
        else:
            print("Detection complete! No objects detected")


if __name__ == '__main__':
    config = {
        'weights': r'runs\260721前\260314-YOLOv8FEB-C2C12-293-400\weights\best.pt',
        'image_path': r'D:\Data\实验结果汇总\识别对比图\C2C12-HEK293\带探针\251223-C2C12-HEK293-95.tif',
        'output_dir': r'D:\Data\实验结果汇总\识别对比图\C2C12-HEK293\带探针',
        'conf_thres': 0.2,
        'iou_thres': 0.5,
        'device': 'cuda:0'
    }

    print("=" * 50)
    print("YOLO single-image detection tool")
    print("=" * 50)
    print(f"Weights file: {config['weights']}")
    print(f"Detection image: {config['image_path']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Confidence threshold: {config['conf_thres']}")
    print(f"IoU threshold: {config['iou_thres']}")
    print(f"Device: {config['device']}")
    print("=" * 50)

    try:
        detector = YOLOSingleDetector(
            weights_path=config['weights'],
            conf_thres=config['conf_thres'],
            iou_thres=config['iou_thres'],
            device=config['device']
        )
        detector.run(config['image_path'], config['output_dir'])
    except Exception as e:
        print(f"Detection failed: {str(e)}")