#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 YOLO 训练好的权重转换为 ONNX 格式

使用方法：
1. 修改脚本中的 MODEL_PATH 为你的训练权重路径
2. 修改脚本中的 OUTPUT_PATH 为你想要保存 ONNX 文件的路径
3. 运行脚本：python export_onnx.py

可修改参数说明：
- MODEL_PATH: 训练好的权重文件路径，如 'best.pt'
- OUTPUT_PATH: 导出的 ONNX 文件路径，如 'best.onnx'
- IMG_SIZE: 模型输入尺寸，默认 640
- DYNAMIC: 是否使用动态输入尺寸，默认 False
- SIMPLIFY: 是否简化 ONNX 模型，默认 True
- HALF: 是否使用半精度导出，默认 False
- DEVICE: 导出设备，默认 'cpu'，也可以使用 'cuda:0' 等
"""

import os
from ultralytics import YOLO

# ------------------------------ 可修改参数 ------------------------------
# 训练好的权重文件路径
MODEL_PATH = 'runs/detect/251128-C2C12-293-120-demo-yolov8n/weights/best.pt'

# 导出的 ONNX 文件路径
OUTPUT_PATH = 'runs/detect/251128-C2C12-293-120-demo-yolov8n/weights/best.onnx'

# 模型输入尺寸
IMG_SIZE = 640

# 是否使用动态输入尺寸
DYNAMIC = False

# 是否简化 ONNX 模型
SIMPLIFY = True

# 是否使用半精度导出
HALF = False

# 导出设备 - 自动检测GPU，如果没有则使用CPU
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
# ------------------------------------------------------------------------

def export_model_to_onnx():
    """将 YOLO 模型导出为 ONNX 格式"""
    
    # 加载模型
    print(f"正在加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # 导出模型
    print(f"正在导出 ONNX 模型...")
    print(f"输出路径: {OUTPUT_PATH}")
    print(f"输入尺寸: {IMG_SIZE}")
    print(f"动态尺寸: {DYNAMIC}")
    print(f"简化模型: {SIMPLIFY}")
    print(f"半精度: {HALF}")
    print(f"设备: {DEVICE}")
    print(f"ONNX opset 版本: 17")  # 支持 FFT 操作的版本
    
    try:
        # 使用 export 方法导出 ONNX 格式
        exported_path = model.export(
            format='onnx',              # 导出格式为 ONNX
            imgsz=IMG_SIZE,             # 输入图像尺寸
            dynamic=DYNAMIC,            # 是否使用动态输入尺寸
            simplify=SIMPLIFY,          # 是否简化 ONNX 模型
            half=HALF,                  # 是否使用半精度
            device=DEVICE,              # 导出设备
            save=True,                  # 保存导出的模型
            #opset=17,                   # 设置较高的 opset 版本，支持 FFT 操作
        )
        
        print(f"\n✅ 模型导出成功!")
        print(f"📁 导出文件路径: {exported_path}")
        
    except Exception as e:
        print(f"\n❌ 模型导出失败!")
        print(f"错误信息: {e}")
        return False
    
    return True

if __name__ == "__main__":
    export_model_to_onnx()