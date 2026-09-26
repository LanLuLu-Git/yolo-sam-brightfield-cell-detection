#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a trained YOLO checkpoint to ONNX format.

Usage:
1. Set MODEL_PATH to the path of your trained checkpoint.
2. Set OUTPUT_PATH to where you want to save the ONNX file.
3. Run: python export_onnx.py

Configurable parameters:
- MODEL_PATH: path to the trained checkpoint, e.g. 'best.pt'
- OUTPUT_PATH: path of the exported ONNX file, e.g. 'best.onnx'
- IMG_SIZE: model input size, default 640
- DYNAMIC: whether to use dynamic input size, default False
- SIMPLIFY: whether to simplify the ONNX model, default True
- HALF: whether to export in half precision, default False
- DEVICE: export device, default 'cpu' (or 'cuda:0', etc.)
"""

import os
from ultralytics import YOLO

# ------------------------------ Configurable parameters ------------------------------
# Path to the trained checkpoint
MODEL_PATH = 'runs/detect/251128-C2C12-293-120-demo-yolov8n/weights/best.pt'

# Path of the exported ONNX file
OUTPUT_PATH = 'runs/detect/251128-C2C12-293-120-demo-yolov8n/weights/best.onnx'

# Model input size
IMG_SIZE = 640

# Whether to use dynamic input size
DYNAMIC = False

# Whether to simplify the ONNX model
SIMPLIFY = True

# Whether to export in half precision
HALF = False

# Export device - auto-detect GPU, fall back to CPU
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
# ------------------------------------------------------------------------

def export_model_to_onnx():
    """Export a YOLO model to ONNX format."""

    # Load the model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Export the model
    print("Exporting ONNX model...")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Input size: {IMG_SIZE}")
    print(f"Dynamic size: {DYNAMIC}")
    print(f"Simplify model: {SIMPLIFY}")
    print(f"Half precision: {HALF}")
    print(f"Device: {DEVICE}")
    print("ONNX opset version: 17")  # version that supports FFT ops

    try:
        # Use the export method to export to ONNX format
        exported_path = model.export(
            format='onnx',              # export format is ONNX
            imgsz=IMG_SIZE,             # input image size
            dynamic=DYNAMIC,            # whether to use dynamic input size
            simplify=SIMPLIFY,          # whether to simplify the ONNX model
            half=HALF,                  # whether to use half precision
            device=DEVICE,              # export device
            save=True,                  # save the exported model
            #opset=17,                   # set a higher opset version to support FFT ops
        )

        print("\n✅ Model exported successfully!")
        print(f"📁 Exported file path: {exported_path}")

    except Exception as e:
        print("\n❌ Model export failed!")
        print(f"Error: {e}")
        return False

    return True

if __name__ == "__main__":
    export_model_to_onnx()
