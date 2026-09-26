#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO batch evaluation script — batch-detect a test image folder using trained weights,
and compute metrics such as Precision, Recall, F1-score, mAP, and FPS.

============================================================
⚠️ Important notes (please read first)
============================================================
 Precision, Recall, and F1-score require comparing predictions against ground truth
 annotations in order to be computed. This script supports two modes:

 Mode A (with annotations): images + annotation files
   → Supports two annotation formats:
     • YOLO format (.txt):  class_id cx cy w h (normalized coordinates)
     • Pascal VOC format (.xml): <annotation> → <object> → <name> + <bndbox>
     Specify via the labels_format parameter: 'txt', 'xml', or 'auto' (auto-detect)
   → Outputs full metrics: P, R, F1, mAP50, mAP50-95, confusion matrix, FPS

 Mode B (without annotations): images only
   → Outputs: FPS benchmark + per-class detection count statistics

 The script auto-detects the label folder. Detection order:
   1. Path specified by the 'labels' parameter in CONFIG
   2. {source}/labels/ subfolder (YOLO format)
   3. {source}/annotations/ subfolder (Pascal VOC format)
   4. The source folder itself (images and labels mixed together)

============================================================
Usage
============================================================

 Method 1 — Edit the CONFIG dictionary directly, then run:
    python batch_evaluate.py

 Method 2 — Command-line arguments (override the corresponding CONFIG settings):
    python batch_evaluate.py --weights best.pt --source Data/test/ --conf 0.5

============================================================
Annotation format description
============================================================

 YOLO format (.txt):
   One object per line: class_id cx cy w h
     - class_id: class index (integer, starting from 0)
     - cx, cy:   bounding-box center coordinates (normalized, 0~1)
     - w, h:     bounding-box width and height (normalized, 0~1)
   Filename matches the image name, extension .txt
   e.g. image_001.jpg corresponds to image_001.txt

 Pascal VOC format (.xml):
   <annotation>
     <filename>image_001.jpg</filename>
     <size><width>640</width><height>480</height></size>
     <object>
       <name>class_name</name>
       <bndbox>
         <xmin>100</xmin><ymin>200</ymin>
         <xmax>300</xmax><ymax>400</ymax>
       </bndbox>
     </object>
   </annotation>
   Filename matches the image name, extension .xml
   e.g. image_001.jpg corresponds to image_001.xml
============================================================
"""

import argparse
import csv
import glob
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils.metrics import (
    ConfusionMatrix,
    DetMetrics,
    ap_per_class,
    box_iou,
)

# ============================================================================
# Default config — modify parameters here to run, no command line needed
# ============================================================================

DEFAULT_CONFIG = {
    # --- Model ---
    'weights': R'runs\detect\260726-MCF7-MCF10A-FEB\weights\best.pt',   # Trained model weights path (.pt)
    # --- Data ---
    'source': R'D:\Data\徐庭臣训练数据集\mcf7-mcf10a-260414\val\images',  # Test image folder path
    'labels': R'D:\Data\徐庭臣训练数据集\mcf7-mcf10a-260414\val\labels',  # Label folder path (None=auto-detect)
    'labels_format': 'auto',                               # Annotation format: 'txt', 'xml', 'auto' (auto-detect)
    # --- Detection parameters ---
    'conf_thres': 0.3,                                 # Confidence threshold (0~1)
    'iou_thres': 0.5,                                  # NMS IoU threshold (0~1)
    'img_size': 640,                                   # Input image size
    'batch_size': 4,                                   # Batch size
    'device': '',                                      # Device: ''=auto, 'cpu', 'cuda:0'
    # --- Output ---
    'output_dir': 'Data\测试对比',            # Result save directory
    'save_annotated': False,                           # Whether to save images with detection boxes
    'save_format': 'csv',                              # 'csv', 'json', 'both'
    'output_prefix': 'MCF7-MCF10A-FEB-',                      # Output filename prefix (e.g. 'yolov8n_' → 'yolov8n_evaluation_metrics.csv')
    'warmup_iterations': 5,                            # GPU warmup iteration count
}

# Supported image formats
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ============================================================================
# Helper functions
# ============================================================================

def get_image_paths(source_path: str) -> List[str]:
    """Get all image paths in a folder, searching subfolders recursively.

    Args:
        source_path: image folder path

    Returns:
        sorted list of image paths
    """
    source_path = os.path.abspath(source_path)
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Image folder does not exist: {source_path}")

    image_paths = []
    for root, _, files in os.walk(source_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        raise FileNotFoundError(f"No supported images found in folder ({source_path})")

    return sorted(image_paths)


def find_labels_folder(source_dir: str, labels_config: Optional[str]) -> Optional[str]:
    """Auto-detect the location of the label folder.

    Detection order:
        1. Path specified by labels_config
        2. {source_dir}/labels/ subfolder
        3. {source_dir}/annotations/ subfolder (Pascal VOC format)
        4. source_dir itself

    Args:
        source_dir: image source folder
        labels_config: user-configured labels parameter

    Returns:
        the found label folder path, or None
    """
    # 1. Path explicitly specified by the user
    if labels_config and os.path.isdir(labels_config):
        return os.path.abspath(labels_config)
    if labels_config:
        print(f"  Warning: the specified label folder does not exist: {labels_config}")

    # 2. source/labels/ subfolder (YOLO format)
    labels_subdir = os.path.join(source_dir, 'labels')
    if os.path.isdir(labels_subdir):
        return labels_subdir

    # 3. source/annotations/ subfolder (Pascal VOC format)
    annotations_subdir = os.path.join(source_dir, 'annotations')
    if os.path.isdir(annotations_subdir):
        return annotations_subdir

    # 4. source itself (images and labels mixed together)
    # Check whether there are .txt or .xml annotation files under source
    txt_files = glob.glob(os.path.join(source_dir, '*.txt'))
    xml_files = glob.glob(os.path.join(source_dir, '*.xml'))
    if txt_files or xml_files:
        return source_dir

    return None


def detect_label_format(labels_dir: str, labels_format_config: Optional[str] = None) -> Optional[str]:
    """Detect the annotation format in the annotation folder.

    Args:
        labels_dir: label folder path
        labels_format_config: user-configured format ('txt', 'xml', 'auto', None)

    Returns:
        'txt' — YOLO format annotations
        'xml' — Pascal VOC format annotations
        None — unable to detect any annotation files
    """
    # Format explicitly specified by the user
    if labels_format_config and labels_format_config != 'auto':
        fmt = labels_format_config.lower()
        if fmt in ('txt', 'xml'):
            # Verify that annotation files of the specified format exist
            pattern = os.path.join(labels_dir, f'*.{fmt}')
            if glob.glob(pattern):
                return fmt
            print(f"  Warning: specified format '{fmt}' but no .{fmt} files were found in {labels_dir}")
        else:
            print(f"  Warning: unknown labels_format '{labels_format_config}', will auto-detect")

    # Auto-detect
    txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    xml_files = glob.glob(os.path.join(labels_dir, '*.xml'))

    if txt_files and xml_files:
        # When both exist, prefer the format with more files
        if len(txt_files) >= len(xml_files):
            return 'txt'
        else:
            return 'xml'
    elif txt_files:
        return 'txt'
    elif xml_files:
        return 'xml'

    return None


def load_yolo_labels(label_path: str, img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load a YOLO format annotation file and convert to absolute coordinates (xyxy).

    YOLO format: class_id cx cy w h  (normalized coordinates)

    Args:
        label_path: .txt annotation file path
        img_width: original image width
        img_height: original image height

    Returns:
        (classes, boxes_xyxy):
            classes:  shape (M,),  dtype=int — class IDs
            boxes:    shape (M, 4), dtype=float32 — absolute coordinates [x1, y1, x2, y2]
    """
    if not os.path.isfile(label_path):
        return np.array([], dtype=int), np.empty((0, 4), dtype=np.float32)

    cls_list, boxes_list = [], []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx = float(parts[1]) * img_width
            cy = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            cls_list.append(cls_id)
            boxes_list.append([x1, y1, x2, y2])

    if not cls_list:
        return np.array([], dtype=int), np.empty((0, 4), dtype=np.float32)

    return np.array(cls_list, dtype=int), np.array(boxes_list, dtype=np.float32)


def load_xml_labels(
    label_path: str,
    class_name_to_id: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a Pascal VOC XML format annotation file and return absolute coordinates (xyxy).

    XML format:
      <annotation>
        <object>
          <name>class_name</name>
          <bndbox>
            <xmin>100</xmin><ymin>200</ymin>
            <xmax>300</xmax><ymax>400</ymax>
          </bndbox>
        </object>
      </annotation>

    The coordinates are already absolute pixel coordinates, no conversion needed.

    Args:
        label_path: .xml annotation file path
        class_name_to_id: mapping from class name to ID, e.g. {'cell': 0, 'nucleus': 1}

    Returns:
        (classes, boxes_xyxy):
            classes:  shape (M,),  dtype=int — class IDs
            boxes:    shape (M, 4), dtype=float32 — absolute coordinates [x1, y1, x2, y2]
    """
    if not os.path.isfile(label_path):
        return np.array([], dtype=int), np.empty((0, 4), dtype=np.float32)

    cls_list, boxes_list = [], []
    try:
        tree = ET.parse(label_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"  Warning: XML parsing failed: {label_path}")
        return np.array([], dtype=int), np.empty((0, 4), dtype=np.float32)

    for obj in root.iter('object'):
        # Skip difficult samples (difficult=1)
        diff = obj.find('difficult')
        if diff is not None and diff.text is not None and diff.text.strip() == '1':
            continue

        # Class name
        name_el = obj.find('name')
        if name_el is None or name_el.text is None:
            continue
        cls_name = name_el.text.strip()

        # Map the class name to an ID
        if cls_name not in class_name_to_id:
            # Try fuzzy matching (remove spaces, case-insensitive)
            cls_name_lower = cls_name.lower().replace(' ', '')
            matched = False
            for known_name, known_id in class_name_to_id.items():
                if known_name.lower().replace(' ', '') == cls_name_lower:
                    cls_id = known_id
                    matched = True
                    break
            if not matched:
                # Unknown class, skip
                continue
        else:
            cls_id = class_name_to_id[cls_name]

        # Bounding box
        bbox_el = obj.find('bndbox')
        if bbox_el is None:
            continue

        try:
            xmin = float(bbox_el.find('xmin').text)
            ymin = float(bbox_el.find('ymin').text)
            xmax = float(bbox_el.find('xmax').text)
            ymax = float(bbox_el.find('ymax').text)
        except (AttributeError, ValueError):
            continue

        # Basic validation
        if xmax <= xmin or ymax <= ymin:
            continue

        cls_list.append(cls_id)
        boxes_list.append([xmin, ymin, xmax, ymax])

    if not cls_list:
        return np.array([], dtype=int), np.empty((0, 4), dtype=np.float32)

    return np.array(cls_list, dtype=int), np.array(boxes_list, dtype=np.float32)


def match_predictions(
    pred_classes: torch.Tensor,
    gt_classes: torch.Tensor,
    iou: torch.Tensor,
    iou_thresholds: torch.Tensor,
) -> np.ndarray:
    """Match predicted boxes to ground-truth boxes and generate the tp matrix.

    Reproduces the logic of ultralytics BaseValidator.match_predictions().
    For each IoU threshold, find (gt, pred) matching pairs and mark them as correct.

    Args:
        pred_classes: predicted classes, shape (N,)
        gt_classes:   ground-truth classes, shape (M,)
        iou:          IoU matrix, shape (M, N) — pairwise IoU between gt_boxes and pred_boxes
        iou_thresholds: IoU threshold vector, shape (K,) — default 10 values [0.5, 0.55, ..., 0.95]

    Returns:
        correct: boolean array of shape (N, K); correct[i, j]=True means the i-th prediction
                 matched some ground-truth annotation under the j-th IoU threshold
    """
    K = len(iou_thresholds)
    N = pred_classes.shape[0]
    correct = np.zeros((N, K), dtype=bool)

    # (M, N) — class match matrix
    correct_class = gt_classes[:, None] == pred_classes  # broadcast
    iou = iou * correct_class  # Zero out IoU for class mismatches
    iou_np = iou.cpu().numpy()

    for j, threshold in enumerate(iou_thresholds.cpu().tolist()):
        # Find all (gt_idx, pred_idx) pairs with IoU >= threshold
        matches = np.nonzero(iou_np >= threshold)
        matches = np.array(matches).T
        if matches.shape[0] == 0:
            continue
        if matches.shape[0] > 1:
            # Sort by IoU in descending order
            matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            # Deduplicate: each predicted box can only match one GT
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # Deduplicate: each GT can only match one predicted box
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), j] = True

    return correct


# ============================================================================
# BatchEvaluator class
# ============================================================================

class BatchEvaluator:
    """YOLO batch evaluator.

    Automatically selects based on whether annotation files exist:
      - Mode A (with annotations): compute full metrics (P, R, F1, mAP, FPS, confusion matrix)
        Supports both YOLO .txt and Pascal VOC .xml annotation formats
      - Mode B (without annotations): measure FPS only + detection statistics
    """

    def __init__(self, config: Dict):
        """Initialize the evaluator.

        Args:
            config: configuration dict, see DEFAULT_CONFIG
        """
        self.config = config
        self.weights = config['weights']
        self.source = os.path.abspath(config['source'])
        self.conf_thres = config['conf_thres']
        self.iou_thres = config['iou_thres']
        self.img_size = config['img_size']
        self.batch_size = config['batch_size']
        self.output_dir = os.path.abspath(config['output_dir'])
        self.save_annotated = config['save_annotated']
        self.save_format = config['save_format']
        self.output_prefix = config.get('output_prefix', '') or ''
        self.warmup_iters = config['warmup_iterations']
        self.labels_format_config = config.get('labels_format', 'auto') or 'auto'

        # Device handling: '' or None → auto-select GPU > CPU
        raw_device = config.get('device', '') or ''
        if raw_device.strip():
            self.device_str = raw_device.strip()
        else:
            self.device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Validate paths
        if not os.path.isfile(self.weights):
            raise FileNotFoundError(f"Model weights file does not exist: {self.weights}")
        if not os.path.isdir(self.source):
            raise FileNotFoundError(f"Image source folder does not exist: {self.source}")

        # Load the model
        print(f"Loading model: {self.weights}")
        self.model = YOLO(self.weights)
        # Move the model to the target device
        if self.device_str != 'cpu' and torch.cuda.is_available():
            self.model.model.to(self.device_str)
        self.device = next(self.model.model.parameters()).device
        print(f"Model loaded, using device: {self.device}")

        # Get class names
        self.names = self.model.names  # {0: 'class0', 1: 'class1', ...}
        self.nc = len(self.names)

        # Build reverse mapping: class name → ID (for XML parsing)
        self.class_name_to_id = {name: idx for idx, name in self.names.items()}

        # IoU thresholds (COCO standard: 0.5 to 0.95, step 0.05)
        self.iouv = torch.linspace(0.5, 0.95, 10)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Statistics variables
        self._reset_stats()

        # Annotation format (determined in detect_mode)
        self._detected_label_format: Optional[str] = None

    def _reset_stats(self):
        """Reset the statistics variables."""
        self.stats = {
            'tp': [],           # List of tp matrices per image
            'conf': [],         # List of confidence scores per image
            'pred_cls': [],     # List of predicted classes per image
            'target_cls': [],   # List of ground-truth classes per image
        }
        self.image_count = 0
        self.total_targets = 0
        self.total_inference_time = 0.0
        self.total_preprocess_time = 0.0
        self.total_postprocess_time = 0.0

    # ------------------------------------------------------------------
    # Mode detection
    # ------------------------------------------------------------------

    def detect_mode(self, image_paths: List[str]) -> str:
        """Detect which evaluation mode to use, and auto-detect the annotation format.

        Returns:
            'A' — has annotation files, full metrics can be computed
            'B' — no annotation files, measure FPS only
        """
        labels_dir = find_labels_folder(self.source, self.config.get('labels'))
        if labels_dir is None:
            self._detected_label_format = None
            return 'B'

        # Detect annotation format
        fmt = detect_label_format(labels_dir, self.labels_format_config)
        if fmt is None:
            self._detected_label_format = None
            return 'B'

        self._detected_label_format = fmt

        # Check that at least one image has an annotation file of the matching format
        ext = f'.{fmt}'
        has_labels = False
        for img_path in image_paths[:10]:  # Spot-check the first 10 images
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, f"{stem}{ext}")
            if os.path.isfile(label_path):
                has_labels = True
                break

        return 'A' if has_labels else 'B'

    # ------------------------------------------------------------------
    # Label loading
    # ------------------------------------------------------------------

    def _resolve_label_path(self, img_path: str, labels_dir: str) -> str:
        """Find the corresponding annotation file path based on the image path.

        Returns the path with the correct extension based on the detected annotation
        format (.txt or .xml).
        """
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ext = f".{self._detected_label_format}" if self._detected_label_format else '.txt'
        return os.path.join(labels_dir, f"{stem}{ext}")

    def _load_labels(self, label_path: str, img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load the annotation file based on the detected annotation format.

        Args:
            label_path: annotation file path
            img_width: image width (only needed for YOLO format)
            img_height: image height (only needed for YOLO format)

        Returns:
            (classes, boxes_xyxy)
        """
        if self._detected_label_format == 'xml':
            return load_xml_labels(label_path, self.class_name_to_id)
        else:
            # Default to YOLO format
            return load_yolo_labels(label_path, img_width, img_height)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self, image_paths: List[str]):
        """Warm up the GPU with the first few images (eliminate the extra overhead of the first inference)."""
        warmup_imgs = image_paths[:min(self.warmup_iters, len(image_paths))]
        if not warmup_imgs:
            return
        print(f"Warming up GPU ({len(warmup_imgs)} iterations)...")
        for img_path in warmup_imgs:
            _ = self.model(img_path, imgsz=self.img_size, conf=self.conf_thres,
                           iou=self.iou_thres, device=self.device_str, verbose=False)

    # ------------------------------------------------------------------
    # Mode A: full evaluation (with annotations)
    # ------------------------------------------------------------------

    def evaluate_with_labels(self, image_paths: List[str], labels_dir: str):
        """Run Mode A evaluation — compute full metrics.

        For each image:
          1. Model inference
          2. Extract predicted boxes (xyxy, conf, cls)
          3. Load ground-truth annotations (xyxy, cls) — auto-adapt YOLO/TXT or VOC/XML format
          4. Compute IoU and match → tp matrix
          5. Accumulate statistics
        """
        label_fmt = self._detected_label_format or 'txt'
        fmt_name = {'txt': 'YOLO (.txt)', 'xml': 'Pascal VOC (.xml)'}.get(label_fmt, label_fmt)

        print(f"\n{'=' * 60}")
        print(f"Mode A — full metric evaluation (with annotations)")
        print(f"Image folder: {self.source}")
        print(f"Label folder: {labels_dir}")
        print(f"Annotation format: {fmt_name}")
        print(f"Number of images: {len(image_paths)}")
        print(f"Confidence threshold: {self.conf_thres}")
        print(f"IoU threshold: {self.iou_thres}")
        print(f"Batch size: {self.batch_size}")
        print(f"{'=' * 60}\n")

        # Warmup
        self._warmup(image_paths)

        # Initialize confusion matrix
        confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf_thres)

        # Timing statistics
        total_inference_ms = 0.0

        # ---- Batch-by-batch processing ----
        pbar = tqdm(range(0, len(image_paths), self.batch_size),
                    desc="Evaluating", unit="batch")
        for batch_start in pbar:
            batch_paths = image_paths[batch_start:batch_start + self.batch_size]

            # 1. Batch inference with timing
            t0 = time.perf_counter()
            results = self.model(
                batch_paths,
                imgsz=self.img_size,
                conf=self.conf_thres,
                iou=self.iou_thres,
                device=self.device_str,
                verbose=False,
                stream=False,
            )
            t1 = time.perf_counter()
            batch_inference_ms = (t1 - t0) * 1000.0
            total_inference_ms += batch_inference_ms

            # 2. Process image by image
            for img_path, result in zip(batch_paths, results):
                self.image_count += 1
                boxes = result.boxes
                img_h, img_w = result.orig_shape  # Original image size

                # Extract predictions — determine the device of the prediction tensors
                if boxes is not None and len(boxes) > 0:
                    pred_xyxy = boxes.xyxy  # (N, 4) — already in original coordinate space
                    pred_conf = boxes.conf   # (N,)
                    pred_cls = boxes.cls     # (N,)
                    pred_device = pred_xyxy.device
                else:
                    pred_device = torch.device('cpu')
                    pred_xyxy = torch.empty((0, 4), device=pred_device)
                    pred_conf = torch.empty((0,), device=pred_device)
                    pred_cls = torch.empty((0,), device=pred_device)

                n_pred = len(pred_cls)

                # Load ground-truth annotations — auto-adapt YOLO/TXT or VOC/XML format
                label_path = self._resolve_label_path(img_path, labels_dir)
                gt_cls_np, gt_xyxy_np = self._load_labels(label_path, img_w, img_h)
                gt_cls = torch.from_numpy(gt_cls_np).to(pred_device)
                gt_xyxy = torch.from_numpy(gt_xyxy_np).to(pred_device)
                n_gt = len(gt_cls)

                self.total_targets += n_gt

                # Initialize the tp matrix for this image
                tp = torch.zeros(n_pred, len(self.iouv), dtype=torch.bool, device=pred_device)

                if n_pred > 0 and n_gt > 0:
                    # Compute pairwise IoU — the two tensors are already on the same device
                    iou = box_iou(gt_xyxy, pred_xyxy)  # (M, N)
                    # Match predictions to ground-truth annotations
                    tp_np = match_predictions(
                        pred_cls, gt_cls, iou, self.iouv.to(pred_device),
                    )
                    tp = torch.from_numpy(tp_np).to(pred_device)

                # Accumulate statistics (save to CPU to save GPU memory, convert back during final aggregation)
                self.stats['tp'].append(tp.cpu())
                self.stats['conf'].append(pred_conf.cpu())
                self.stats['pred_cls'].append(pred_cls.cpu())
                self.stats['target_cls'].append(gt_cls.cpu())

                # Update confusion matrix
                if n_pred > 0 or n_gt > 0:
                    pred_for_cm = None
                    if n_pred > 0:
                        pred_for_cm = torch.cat([pred_xyxy, pred_conf.unsqueeze(1), pred_cls.unsqueeze(1)], dim=1)
                    confusion_matrix.process_batch(
                        pred_for_cm.cpu() if pred_for_cm is not None else None,
                        gt_xyxy.cpu(),
                        gt_cls.cpu(),
                    )

                # Update progress bar info
                pbar.set_postfix({
                    'images': self.image_count,
                    'targets': self.total_targets,
                    'speed': f'{batch_inference_ms / len(batch_paths):.1f}ms/img'
                })

            # Optional: save annotated images
            if self.save_annotated:
                for r in results:
                    save_path = os.path.join(self.output_dir, os.path.basename(r.path))
                    r.save(filename=save_path)

        # ---- Aggregate statistics ----
        pbar.close()
        print(f"\nComputing metrics...")

        # Concatenate all statistics
        tp_all = torch.cat(self.stats['tp'], dim=0).cpu().numpy()
        conf_all = torch.cat(self.stats['conf'], dim=0).cpu().numpy()
        pred_cls_all = torch.cat(self.stats['pred_cls'], dim=0).cpu().numpy()
        target_cls_all = torch.cat(self.stats['target_cls'], dim=0).cpu().numpy()

        # Compute FPS
        avg_inference_ms = total_inference_ms / max(self.image_count, 1)
        fps = 1000.0 / avg_inference_ms if avg_inference_ms > 0 else 0.0

        speed_info = {
            'total_time_s': total_inference_ms / 1000.0,
            'avg_inference_ms': avg_inference_ms,
            'fps': fps,
            'num_images': self.image_count,
            'batch_size': self.batch_size,
            'device': str(self.device),
        }

        # If there are no detection results
        if len(conf_all) == 0:
            print("\n⚠️  The model did not detect any objects in any image, so metrics cannot be computed.")
            print("   Please try lowering the conf_thres parameter.")
            self._print_fps_only(speed_info)
            return {'speed': speed_info, 'metrics': None}

        # Call ap_per_class to compute metrics
        tp_out, fp_out, p, r, f1, ap, ap_class, p_curve, r_curve, f1_curve, x, prec_values = \
            ap_per_class(
                tp_all, conf_all, pred_cls_all, target_cls_all,
                plot=False,
                save_dir=Path(self.output_dir),
                names=self.names,
            )

        # Build structured results
        results_data = {
            'speed': speed_info,
            'per_class': {},
            'mean': {},
        }

        # Per-class metrics
        ap50_per_class = ap[:, 0]  # AP@IoU=0.5
        ap_per_class_mean = ap.mean(1)  # mAP@0.5:0.95

        for i, cls_idx in enumerate(ap_class):
            cls_name = self.names.get(cls_idx, f'class_{cls_idx}')
            n_instances = int(np.bincount(target_cls_all.astype(int), minlength=self.nc)[cls_idx])

            results_data['per_class'][cls_name] = {
                'class_id': int(cls_idx),
                'precision': round(float(p[i]), 4),
                'recall': round(float(r[i]), 4),
                'f1': round(float(f1[i]), 4),
                'ap50': round(float(ap50_per_class[i]), 4),
                'ap50_95': round(float(ap_per_class_mean[i]), 4),
                'instances': n_instances,
                'tp': int(tp_out[i]),
                'fp': int(fp_out[i]),
            }

        # Overall mean
        results_data['mean'] = {
            'precision': round(float(p.mean()), 4),
            'recall': round(float(r.mean()), 4),
            'f1': round(float(f1.mean()), 4),
            'mAP50': round(float(ap50_per_class.mean()), 4),
            'mAP50_95': round(float(ap_per_class_mean.mean()), 4),
            'total_images': self.image_count,
            'total_instances': self.total_targets,
        }

        # Confusion matrix
        results_data['confusion_matrix'] = confusion_matrix.matrix.tolist()

        # Print results
        self._print_mode_a_results(results_data, ap_class)
        self._export_results(results_data)

        return results_data

    # ------------------------------------------------------------------
    # Mode B: FPS benchmark (without annotations)
    # ------------------------------------------------------------------

    def evaluate_without_labels(self, image_paths: List[str]):
        """Run Mode B evaluation — FPS benchmark.

        Perform batch inference on the images, measure speed and count detections per class.
        """
        print(f"\n{'=' * 60}")
        print(f"Mode B — FPS benchmark (without annotations)")
        print(f"Image folder: {self.source}")
        print(f"Number of images: {len(image_paths)}")
        print(f"Confidence threshold: {self.conf_thres}")
        print(f"Batch size: {self.batch_size}")
        print(f"{'=' * 60}\n")

        # Warmup
        self._warmup(image_paths)

        # Statistics
        total_inference_ms = 0.0
        per_class_counts = {self.names.get(i, f'class_{i}'): 0 for i in range(self.nc)}

        # ---- Batch-by-batch processing ----
        pbar = tqdm(range(0, len(image_paths), self.batch_size),
                    desc="Inference", unit="batch")
        for batch_start in pbar:
            batch_paths = image_paths[batch_start:batch_start + self.batch_size]

            # Inference with timing
            t0 = time.perf_counter()
            results = self.model(
                batch_paths,
                imgsz=self.img_size,
                conf=self.conf_thres,
                iou=self.iou_thres,
                device=self.device_str,
                verbose=False,
                stream=False,
            )
            t1 = time.perf_counter()
            total_inference_ms += (t1 - t0) * 1000.0

            # Count detection results
            for result in results:
                self.image_count += 1
                if result.boxes is not None and len(result.boxes) > 0:
                    for cls_id in result.boxes.cls.int().tolist():
                        cls_name = self.names.get(cls_id, f'class_{cls_id}')
                        per_class_counts[cls_name] += 1

            # Optional: save annotated images
            if self.save_annotated:
                for r in results:
                    save_path = os.path.join(self.output_dir, os.path.basename(r.path))
                    r.save(filename=save_path)

        pbar.close()

        # ---- Compute FPS ----
        avg_inference_ms = total_inference_ms / max(self.image_count, 1)
        fps = 1000.0 / avg_inference_ms if avg_inference_ms > 0 else 0.0

        speed_info = {
            'total_time_s': total_inference_ms / 1000.0,
            'avg_inference_ms': avg_inference_ms,
            'fps': fps,
            'num_images': self.image_count,
            'batch_size': self.batch_size,
            'device': str(self.device),
        }

        results_data = {
            'speed': speed_info,
            'per_class_counts': per_class_counts,
        }

        # Print results
        self._print_mode_b_results(results_data)
        self._export_results(results_data)

        return results_data

    # ------------------------------------------------------------------
    # Output: console printing
    # ------------------------------------------------------------------

    def _print_mode_a_results(self, results_data: Dict, ap_class: np.ndarray):
        """Print the full metric table for Mode A."""
        speed = results_data['speed']
        mean = results_data['mean']

        print(f"\n{'=' * 70}")
        print(f"  Batch evaluation results — full metrics (Mode A)")
        print(f"{'=' * 70}")
        print(f"  Model:      {os.path.basename(self.weights)}")
        print(f"  Images:     {mean['total_images']}")
        print(f"  Total instances: {mean['total_instances']}")
        print(f"  Confidence: {self.conf_thres}")
        print(f"  NMS IoU:    {self.iou_thres}")
        print(f"{'-' * 70}")

        # Table header
        header = f"  {'Class':<14s} {'Instances':>7s} {'P':>8s} {'R':>8s} {'F1':>8s} {'mAP50':>8s} {'mAP50-95':>9s}"
        print(header)
        print(f"  {'-' * 66}")

        # Per-class rows
        per_class = results_data['per_class']
        for cls_name in sorted(per_class.keys()):
            d = per_class[cls_name]
            print(f"  {cls_name:<14s} {d['instances']:>7d} {d['precision']:>8.4f} "
                  f"{d['recall']:>8.4f} {d['f1']:>8.4f} {d['ap50']:>8.4f} {d['ap50_95']:>9.4f}")

        # Overall mean row
        print(f"  {'-' * 66}")
        print(f"  {'Overall (mean)':<14s} {mean['total_instances']:>7d} {mean['precision']:>8.4f} "
              f"{mean['recall']:>8.4f} {mean['f1']:>8.4f} {mean['mAP50']:>8.4f} {mean['mAP50_95']:>9.4f}")

        # Speed
        print(f"{'-' * 70}")
        print(f"  Inference speed:")
        print(f"    Total time:     {speed['total_time_s']:.2f} s")
        print(f"    Avg per image:  {speed['avg_inference_ms']:.1f} ms")
        print(f"    FPS:           {speed['fps']:.1f} images/s")
        print(f"    Batch size:    {speed['batch_size']}")
        print(f"    Device:        {speed['device']}")
        print(f"{'-' * 70}")
        print(f"  Results saved to: {self.output_dir}")
        print(f"{'=' * 70}\n")

    def _print_mode_b_results(self, results_data: Dict):
        """Print FPS and detection statistics for Mode B."""
        speed = results_data['speed']
        counts = results_data['per_class_counts']

        print(f"\n{'=' * 70}")
        print(f"  FPS benchmark results (Mode B — without annotations)")
        print(f"{'=' * 70}")
        print(f"  Model:      {os.path.basename(self.weights)}")
        print(f"  Images:     {speed['num_images']}")
        print(f"  Confidence: {self.conf_thres}")
        print(f"  Batch size: {speed['batch_size']}")
        print(f"{'-' * 70}")
        print(f"  Inference speed:")
        print(f"    Total time:     {speed['total_time_s']:.2f} s")
        print(f"    Avg per image:  {speed['avg_inference_ms']:.1f} ms")
        print(f"    FPS:           {speed['fps']:.1f} images/s")
        print(f"    Device:        {speed['device']}")
        print(f"{'-' * 70}")
        print(f"  Detection counts per class:")
        for cls_name, count in sorted(counts.items()):
            if count > 0:
                print(f"    {cls_name:<20s}: {count:>6d}")
        print(f"{'-' * 70}")
        print(f"  Results saved to: {self.output_dir}")
        print(f"{'=' * 70}\n")

    def _print_fps_only(self, speed_info: Dict):
        """Print only speed information when there are no detections."""
        print(f"\n  Inference speed (no detections):")
        print(f"    Total time:     {speed_info['total_time_s']:.2f} s")
        print(f"    Avg per image:  {speed_info['avg_inference_ms']:.1f} ms")
        print(f"    FPS:           {speed_info['fps']:.1f} images/s")

    # ------------------------------------------------------------------
    # Output: file export
    # ------------------------------------------------------------------

    def _export_results(self, results_data: Dict):
        """Export results to CSV and/or JSON files."""
        fmt = self.save_format.lower()

        if fmt in ('csv', 'both'):
            self._export_csv(results_data)

        if fmt in ('json', 'both'):
            self._export_json(results_data)

    def _export_csv(self, results_data: Dict):
        """Export results to a CSV file."""
        pf = self.output_prefix  # prefix

        # Mode A results
        if 'per_class' in results_data and results_data['per_class']:
            csv_path = os.path.join(self.output_dir, f'{pf}evaluation_metrics.csv')
            fieldnames = ['class', 'instances', 'precision', 'recall', 'f1', 'ap50', 'ap50_95']
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for cls_name, d in sorted(results_data['per_class'].items()):
                    writer.writerow({
                        'class': cls_name,
                        'instances': d['instances'],
                        'precision': d['precision'],
                        'recall': d['recall'],
                        'f1': d['f1'],
                        'ap50': d['ap50'],
                        'ap50_95': d['ap50_95'],
                    })
                # Overall mean row
                mean = results_data['mean']
                writer.writerow({
                    'class': 'Overall (mean)',
                    'instances': mean['total_instances'],
                    'precision': mean['precision'],
                    'recall': mean['recall'],
                    'f1': mean['f1'],
                    'ap50': mean['mAP50'],
                    'ap50_95': mean['mAP50_95'],
                })
            print(f"  CSV metrics file: {csv_path}")

        # Speed info (available in both modes)
        csv_speed_path = os.path.join(self.output_dir, f'{pf}evaluation_speed.csv')
        speed = results_data['speed']
        with open(csv_speed_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['num_images', speed['num_images']])
            writer.writerow(['batch_size', speed['batch_size']])
            writer.writerow(['device', speed['device']])
            writer.writerow(['total_time_s', f"{speed['total_time_s']:.4f}"])
            writer.writerow(['avg_inference_ms', f"{speed['avg_inference_ms']:.2f}"])
            writer.writerow(['fps', f"{speed['fps']:.2f}"])
        print(f"  CSV speed file: {csv_speed_path}")

    def _export_json(self, results_data: Dict):
        """Export results to a JSON file."""
        pf = self.output_prefix
        json_path = os.path.join(self.output_dir, f'{pf}evaluation_results.json')

        # Convert numpy types to native Python types
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results_data), f, ensure_ascii=False, indent=2)
        print(f"  JSON results file: {json_path}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Run the evaluation workflow. Auto-detect the mode and annotation format and execute."""
        # 1. Get the image list
        print(f"\nScanning image folder: {self.source}")
        image_paths = get_image_paths(self.source)
        print(f"Found {len(image_paths)} images")

        # 2. Detect mode (also detect annotation format)
        mode = self.detect_mode(image_paths)
        labels_dir = find_labels_folder(self.source, self.config.get('labels'))

        # 3. Run evaluation
        if mode == 'A' and labels_dir:
            fmt_name = {'txt': 'YOLO (.txt)', 'xml': 'Pascal VOC (.xml)'}.get(
                self._detected_label_format, self._detected_label_format)
            print(f"Detected label folder: {labels_dir}")
            print(f"Detected annotation format: {fmt_name}")
            return self.evaluate_with_labels(image_paths, labels_dir)
        else:
            print("\n⚠️  No annotation files detected, will use Mode B (FPS only)")
            if labels_dir:
                print(f"   Label folder exists ({labels_dir}) but has no annotation files matching the images")
            else:
                print(f"   No label folder found. Checked: "
                      f"{self.config.get('labels') or 'N/A'}、"
                      f"{os.path.join(self.source, 'labels')}、"
                      f"{os.path.join(self.source, 'annotations')}、"
                      f"{self.source}")
            print("   To compute P/R/F1/mAP, make sure each image has a corresponding .txt (YOLO) or .xml (Pascal VOC) annotation file.\n")
            return self.evaluate_without_labels(image_paths)


# ============================================================================
# Command-line argument parsing
# ============================================================================

def parse_args() -> Dict:
    """Parse command-line arguments and return a dict that overrides CONFIG."""
    parser = argparse.ArgumentParser(
        description='YOLO batch evaluation script — compute Precision, Recall, F1, FPS and other metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_evaluate.py --weights best.pt --source Data/test/ --conf 0.5
  python batch_evaluate.py --weights best.pt --source Data/test/ --labels Data/labels/ --batch 8
  python batch_evaluate.py --weights best.pt --source Data/test/ --labels-format xml
  python batch_evaluate.py  (uses parameters from DEFAULT_CONFIG)
        """
    )
    parser.add_argument('--weights', type=str, help='model weights path')
    parser.add_argument('--source', type=str, help='test image folder')
    parser.add_argument('--labels', type=str, default=None, help='label folder (optional)')
    parser.add_argument('--labels-format', type=str, dest='labels_format',
                        choices=['txt', 'xml', 'auto'], default=None,
                        help='annotation format: txt (YOLO), xml (Pascal VOC), auto (auto-detect, default)')
    parser.add_argument('--conf', type=float, dest='conf_thres', help='confidence threshold')
    parser.add_argument('--iou', type=float, dest='iou_thres', help='NMS IoU threshold')
    parser.add_argument('--img', type=int, dest='img_size', help='input image size')
    parser.add_argument('--batch', type=int, dest='batch_size', help='batch size')
    parser.add_argument('--device', type=str, help='device (cpu, cuda:0, etc.)')
    parser.add_argument('--output', type=str, dest='output_dir', help='output directory')
    parser.add_argument('--save-img', action='store_true', dest='save_annotated',
                        help='save annotated images')
    parser.add_argument('--format', type=str, dest='save_format',
                        choices=['csv', 'json', 'both'], help='save format')
    parser.add_argument('--prefix', type=str, dest='output_prefix', default=None,
                        help='output filename prefix (leave empty to not add)')
    parser.add_argument('--warmup', type=int, dest='warmup_iterations', help='warmup iteration count')

    args = parser.parse_args()
    # Only return non-None arguments
    return {k: v for k, v in vars(args).items() if v is not None}


# ============================================================================
# Main function
# ============================================================================

def main():
    """Main entry point. Merge DEFAULT_CONFIG and command-line arguments, then run evaluation."""
    # Merge config: command-line arguments take priority
    config = DEFAULT_CONFIG.copy()
    cli_overrides = parse_args()
    config.update(cli_overrides)

    # Create the evaluator and run
    try:
        evaluator = BatchEvaluator(config)
        results = evaluator.run()
        return results
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please check the path settings in CONFIG.")
        return None
    except Exception as e:
        print(f"\n❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    """
    Direct run instructions:

    1. Modify the parameters in the DEFAULT_CONFIG dictionary above
    2. Run directly: python batch_evaluate.py
    3. Or override individual parameters from the command line: python batch_evaluate.py --conf 0.6

    Output files:
      - evaluation_metrics.csv   : per-class + overall metrics
      - evaluation_speed.csv     : FPS speed info
      - evaluation_results.json  : full results (when JSON format is selected)
    """
    main()
