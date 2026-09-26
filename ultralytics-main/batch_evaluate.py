#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 批量评估脚本 — 使用训练后的权重对测试图片文件夹进行批量检测，
统计 Precision、Recall、F1-score、mAP、FPS 等指标。

============================================================
⚠️ 重要说明（请先阅读）
============================================================
 Precision、Recall、F1-score 需要对比预测结果与真实标注 (ground truth)
 才能计算。本脚本支持两种模式:

 模式 A（有标注）: 图片 + 标注文件
   → 支持两种标注格式:
     • YOLO 格式 (.txt):   class_id cx cy w h (归一化坐标)
     • Pascal VOC 格式 (.xml): <annotation> → <object> → <name> + <bndbox>
     通过 labels_format 参数指定: 'txt', 'xml', 或 'auto' (自动检测)
   → 输出完整指标: P, R, F1, mAP50, mAP50-95, 混淆矩阵, FPS

 模式 B（无标注）: 仅图片
   → 输出: FPS 基准测试 + 各类检测数量统计

 脚本会自动检测标签文件夹。检测顺序:
   1. CONFIG 中 'labels' 参数指定的路径
   2. {source}/labels/ 子文件夹 (YOLO 格式)
   3. {source}/annotations/ 子文件夹 (Pascal VOC 格式)
   4. source 文件夹自身（图片和标签混放）

============================================================
使用方式
============================================================

 方式 1 — 直接修改 CONFIG 字典，然后运行:
    python batch_evaluate.py

 方式 2 — 命令行参数（会覆盖 CONFIG 中的对应设置）:
    python batch_evaluate.py --weights best.pt --source Data/test/ --conf 0.5

============================================================
标注格式说明
============================================================

 YOLO 格式 (.txt):
   每行一个目标: class_id cx cy w h
     - class_id: 类别编号 (整数, 从 0 开始)
     - cx, cy:   边界框中心点坐标 (归一化, 0~1)
     - w, h:     边界框宽高 (归一化, 0~1)
   文件名与图片名相同, 扩展名为 .txt
   例如: image_001.jpg 对应 image_001.txt

 Pascal VOC 格式 (.xml):
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
   文件名与图片名相同, 扩展名为 .xml
   例如: image_001.jpg 对应 image_001.xml
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
# 默认配置 — 在这里修改参数即可运行，无需命令行
# ============================================================================

DEFAULT_CONFIG = {
    # --- 模型 ---
    'weights': R'runs\detect\260726-MCF7-MCF10A-FEB\weights\best.pt',   # 训练好的模型权重路径 (.pt)
    # --- 数据 ---
    'source': R'D:\Data\徐庭臣训练数据集\mcf7-mcf10a-260414\val\images',  # 测试图片文件夹路径
    'labels': R'D:\Data\徐庭臣训练数据集\mcf7-mcf10a-260414\val\labels',  # 标签文件夹路径 (None=自动检测)
    'labels_format': 'auto',                               # 标注格式: 'txt', 'xml', 'auto' (自动检测)
    # --- 检测参数 ---
    'conf_thres': 0.3,                                 # 置信度阈值 (0~1)
    'iou_thres': 0.5,                                  # NMS 的 IoU 阈值 (0~1)
    'img_size': 640,                                   # 输入图像尺寸
    'batch_size': 4,                                   # 批处理大小
    'device': '',                                      # 设备: ''=自动, 'cpu', 'cuda:0'
    # --- 输出 ---
    'output_dir': 'Data\测试对比',            # 结果保存目录
    'save_annotated': False,                           # 是否保存带检测框的图片
    'save_format': 'csv',                              # 'csv', 'json', 'both'
    'output_prefix': 'MCF7-MCF10A-FEB-',                      # 输出文件名前缀 (如 'yolov8n_' → 'yolov8n_evaluation_metrics.csv')
    'warmup_iterations': 5,                            # GPU 预热迭代次数
}

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ============================================================================
# 辅助函数
# ============================================================================

def get_image_paths(source_path: str) -> List[str]:
    """获取文件夹中所有图片路径，递归搜索子文件夹。

    Args:
        source_path: 图片文件夹路径

    Returns:
        排序后的图片路径列表
    """
    source_path = os.path.abspath(source_path)
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"图片文件夹不存在: {source_path}")

    image_paths = []
    for root, _, files in os.walk(source_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        raise FileNotFoundError(f"文件夹中未找到支持的图片 ({source_path})")

    return sorted(image_paths)


def find_labels_folder(source_dir: str, labels_config: Optional[str]) -> Optional[str]:
    """自动检测标签文件夹的位置。

    检测顺序:
        1. labels_config 指定的路径
        2. {source_dir}/labels/ 子文件夹
        3. {source_dir}/annotations/ 子文件夹 (Pascal VOC 格式)
        4. source_dir 自身

    Args:
        source_dir: 图片源文件夹
        labels_config: 用户配置的 labels 参数

    Returns:
        找到的标签文件夹路径，或 None
    """
    # 1. 用户显式指定的路径
    if labels_config and os.path.isdir(labels_config):
        return os.path.abspath(labels_config)
    if labels_config:
        print(f"  警告: 指定的标签文件夹不存在: {labels_config}")

    # 2. source/labels/ 子文件夹 (YOLO 格式)
    labels_subdir = os.path.join(source_dir, 'labels')
    if os.path.isdir(labels_subdir):
        return labels_subdir

    # 3. source/annotations/ 子文件夹 (Pascal VOC 格式)
    annotations_subdir = os.path.join(source_dir, 'annotations')
    if os.path.isdir(annotations_subdir):
        return annotations_subdir

    # 4. source 自身（图片和标签混放）
    # 检查 source 下是否有 .txt 或 .xml 标注文件
    txt_files = glob.glob(os.path.join(source_dir, '*.txt'))
    xml_files = glob.glob(os.path.join(source_dir, '*.xml'))
    if txt_files or xml_files:
        return source_dir

    return None


def detect_label_format(labels_dir: str, labels_format_config: Optional[str] = None) -> Optional[str]:
    """检测标注文件夹中的标注格式。

    Args:
        labels_dir: 标签文件夹路径
        labels_format_config: 用户配置的格式 ('txt', 'xml', 'auto', None)

    Returns:
        'txt' — YOLO 格式标注
        'xml' — Pascal VOC 格式标注
        None — 无法检测到任何标注文件
    """
    # 用户显式指定格式
    if labels_format_config and labels_format_config != 'auto':
        fmt = labels_format_config.lower()
        if fmt in ('txt', 'xml'):
            # 验证指定格式的标注文件是否存在
            pattern = os.path.join(labels_dir, f'*.{fmt}')
            if glob.glob(pattern):
                return fmt
            print(f"  警告: 指定格式 '{fmt}' 但在 {labels_dir} 中未找到 .{fmt} 文件")
        else:
            print(f"  警告: 未知的 labels_format '{labels_format_config}'，将自动检测")

    # 自动检测
    txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    xml_files = glob.glob(os.path.join(labels_dir, '*.xml'))

    if txt_files and xml_files:
        # 两者都存在时优先使用数量多的格式
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
    """加载 YOLO 格式标注文件，转换为绝对坐标 (xyxy)。

    YOLO 格式: class_id cx cy w h  (归一化坐标)

    Args:
        label_path: .txt 标注文件路径
        img_width: 原始图片宽度
        img_height: 原始图片高度

    Returns:
        (classes, boxes_xyxy):
            classes:  形状 (M,),  dtype=int — 类别ID
            boxes:    形状 (M, 4), dtype=float32 — 绝对坐标 [x1, y1, x2, y2]
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
    """加载 Pascal VOC XML 格式标注文件，返回绝对坐标 (xyxy)。

    XML 格式:
      <annotation>
        <object>
          <name>class_name</name>
          <bndbox>
            <xmin>100</xmin><ymin>200</ymin>
            <xmax>300</xmax><ymax>400</ymax>
          </bndbox>
        </object>
      </annotation>

    坐标已经是像素绝对坐标, 无需转换。

    Args:
        label_path: .xml 标注文件路径
        class_name_to_id: 类别名称到 ID 的映射, 如 {'cell': 0, 'nucleus': 1}

    Returns:
        (classes, boxes_xyxy):
            classes:  形状 (M,),  dtype=int — 类别ID
            boxes:    形状 (M, 4), dtype=float32 — 绝对坐标 [x1, y1, x2, y2]
    """
    if not os.path.isfile(label_path):
        return np.array([], dtype=int), np.empty((0, 4), dtype=np.float32)

    cls_list, boxes_list = [], []
    try:
        tree = ET.parse(label_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"  警告: XML 解析失败: {label_path}")
        return np.array([], dtype=int), np.empty((0, 4), dtype=np.float32)

    for obj in root.iter('object'):
        # 跳过困难样本 (difficult=1)
        diff = obj.find('difficult')
        if diff is not None and diff.text is not None and diff.text.strip() == '1':
            continue

        # 类别名称
        name_el = obj.find('name')
        if name_el is None or name_el.text is None:
            continue
        cls_name = name_el.text.strip()

        # 将类别名称映射为 ID
        if cls_name not in class_name_to_id:
            # 尝试模糊匹配 (去除空格、大小写不敏感)
            cls_name_lower = cls_name.lower().replace(' ', '')
            matched = False
            for known_name, known_id in class_name_to_id.items():
                if known_name.lower().replace(' ', '') == cls_name_lower:
                    cls_id = known_id
                    matched = True
                    break
            if not matched:
                # 未知类别，跳过
                continue
        else:
            cls_id = class_name_to_id[cls_name]

        # 边界框
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

        # 基本校验
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
    """将预测框匹配到真实标注框，生成 tp 矩阵。

    复现 ultralytics BaseValidator.match_predictions() 的逻辑。
    对每个 IoU 阈值，找到 (gt, pred) 匹配对，标记为正确预测。

    Args:
        pred_classes: 预测类别, 形状 (N,)
        gt_classes:   真实类别, 形状 (M,)
        iou:          IoU 矩阵, 形状 (M, N) — gt_boxes 与 pred_boxes 的成对 IoU
        iou_thresholds: IoU 阈值向量, 形状 (K,)  — 默认 10 个 [0.5, 0.55, ..., 0.95]

    Returns:
        correct: 形状 (N, K) 的布尔数组, correct[i, j]=True 表示第 i 个预测在第 j 个
                 IoU 阈值下与某个真实标注匹配成功
    """
    K = len(iou_thresholds)
    N = pred_classes.shape[0]
    correct = np.zeros((N, K), dtype=bool)

    # (M, N) — 类别匹配矩阵
    correct_class = gt_classes[:, None] == pred_classes  # 广播
    iou = iou * correct_class  # 将类别不匹配的 IoU 清零
    iou_np = iou.cpu().numpy()

    for j, threshold in enumerate(iou_thresholds.cpu().tolist()):
        # 找到所有 IoU >= 阈值的 (gt_idx, pred_idx) 对
        matches = np.nonzero(iou_np >= threshold)
        matches = np.array(matches).T
        if matches.shape[0] == 0:
            continue
        if matches.shape[0] > 1:
            # 按 IoU 降序排列
            matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            # 去重: 每个预测框只能匹配一个 GT
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # 去重: 每个 GT 也只能匹配一个预测框
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), j] = True

    return correct


# ============================================================================
# BatchEvaluator 类
# ============================================================================

class BatchEvaluator:
    """YOLO 批量评估器。

    根据是否有标注文件自动选择:
      - 模式 A (有标注): 计算完整指标 (P, R, F1, mAP, FPS, 混淆矩阵)
        支持 YOLO .txt 和 Pascal VOC .xml 两种标注格式
      - 模式 B (无标注): 仅测 FPS + 检测统计
    """

    def __init__(self, config: Dict):
        """初始化评估器。

        Args:
            config: 配置字典, 参见 DEFAULT_CONFIG
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

        # 设备处理: '' 或 None → 自动选择 GPU > CPU
        raw_device = config.get('device', '') or ''
        if raw_device.strip():
            self.device_str = raw_device.strip()
        else:
            self.device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # 验证路径
        if not os.path.isfile(self.weights):
            raise FileNotFoundError(f"模型权重文件不存在: {self.weights}")
        if not os.path.isdir(self.source):
            raise FileNotFoundError(f"图片源文件夹不存在: {self.source}")

        # 加载模型
        print(f"正在加载模型: {self.weights}")
        self.model = YOLO(self.weights)
        # 将模型移到目标设备
        if self.device_str != 'cpu' and torch.cuda.is_available():
            self.model.model.to(self.device_str)
        self.device = next(self.model.model.parameters()).device
        print(f"模型加载完成, 使用设备: {self.device}")

        # 获取类别名称
        self.names = self.model.names  # {0: 'class0', 1: 'class1', ...}
        self.nc = len(self.names)

        # 构建反向映射: 类别名称 → ID (用于 XML 解析)
        self.class_name_to_id = {name: idx for idx, name in self.names.items()}

        # IoU 阈值 (COCO 标准: 0.5 到 0.95, 步长 0.05)
        self.iouv = torch.linspace(0.5, 0.95, 10)

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 统计变量
        self._reset_stats()

        # 标注格式 (在 detect_mode 时确定)
        self._detected_label_format: Optional[str] = None

    def _reset_stats(self):
        """重置统计变量。"""
        self.stats = {
            'tp': [],           # 各图片的 tp 矩阵列表
            'conf': [],         # 各图片的置信度列表
            'pred_cls': [],     # 各图片的预测类别列表
            'target_cls': [],   # 各图片的真实类别列表
        }
        self.image_count = 0
        self.total_targets = 0
        self.total_inference_time = 0.0
        self.total_preprocess_time = 0.0
        self.total_postprocess_time = 0.0

    # ------------------------------------------------------------------
    # 模式检测
    # ------------------------------------------------------------------

    def detect_mode(self, image_paths: List[str]) -> str:
        """检测应使用哪种评估模式，同时自动检测标注格式。

        Returns:
            'A' — 有标注文件，可计算完整指标
            'B' — 无标注文件，仅测 FPS
        """
        labels_dir = find_labels_folder(self.source, self.config.get('labels'))
        if labels_dir is None:
            self._detected_label_format = None
            return 'B'

        # 检测标注格式
        fmt = detect_label_format(labels_dir, self.labels_format_config)
        if fmt is None:
            self._detected_label_format = None
            return 'B'

        self._detected_label_format = fmt

        # 检查至少有一张图片有对应格式的标注文件
        ext = f'.{fmt}'
        has_labels = False
        for img_path in image_paths[:10]:  # 抽查前 10 张
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, f"{stem}{ext}")
            if os.path.isfile(label_path):
                has_labels = True
                break

        return 'A' if has_labels else 'B'

    # ------------------------------------------------------------------
    # 标签加载
    # ------------------------------------------------------------------

    def _resolve_label_path(self, img_path: str, labels_dir: str) -> str:
        """根据图片路径找到对应的标注文件路径。

        根据检测到的标注格式 (.txt 或 .xml) 返回正确扩展名的路径。
        """
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ext = f".{self._detected_label_format}" if self._detected_label_format else '.txt'
        return os.path.join(labels_dir, f"{stem}{ext}")

    def _load_labels(self, label_path: str, img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """根据检测到的标注格式加载标注文件。

        Args:
            label_path: 标注文件路径
            img_width: 图片宽度 (仅 YOLO 格式需要)
            img_height: 图片高度 (仅 YOLO 格式需要)

        Returns:
            (classes, boxes_xyxy)
        """
        if self._detected_label_format == 'xml':
            return load_xml_labels(label_path, self.class_name_to_id)
        else:
            # 默认使用 YOLO 格式
            return load_yolo_labels(label_path, img_width, img_height)

    # ------------------------------------------------------------------
    # 预热
    # ------------------------------------------------------------------

    def _warmup(self, image_paths: List[str]):
        """用前几张图片预热 GPU（消除首次推理的额外开销）。"""
        warmup_imgs = image_paths[:min(self.warmup_iters, len(image_paths))]
        if not warmup_imgs:
            return
        print(f"正在预热 GPU ({len(warmup_imgs)} 次迭代)...")
        for img_path in warmup_imgs:
            _ = self.model(img_path, imgsz=self.img_size, conf=self.conf_thres,
                           iou=self.iou_thres, device=self.device_str, verbose=False)

    # ------------------------------------------------------------------
    # 模式 A: 完整评估 (有标注)
    # ------------------------------------------------------------------

    def evaluate_with_labels(self, image_paths: List[str], labels_dir: str):
        """执行模式 A 评估 — 计算完整指标。

        对每张图片:
          1. 模型推理
          2. 提取预测框 (xyxy, conf, cls)
          3. 加载真实标注 (xyxy, cls) — 自动适配 YOLO/TXT 或 VOC/XML 格式
          4. 计算 IoU 并匹配 → tp 矩阵
          5. 累积统计数据
        """
        label_fmt = self._detected_label_format or 'txt'
        fmt_name = {'txt': 'YOLO (.txt)', 'xml': 'Pascal VOC (.xml)'}.get(label_fmt, label_fmt)

        print(f"\n{'=' * 60}")
        print(f"模式 A — 完整指标评估 (有标注)")
        print(f"图片文件夹: {self.source}")
        print(f"标签文件夹: {labels_dir}")
        print(f"标注格式:   {fmt_name}")
        print(f"图片数量:   {len(image_paths)}")
        print(f"置信度阈值: {self.conf_thres}")
        print(f"IoU 阈值:   {self.iou_thres}")
        print(f"批处理大小: {self.batch_size}")
        print(f"{'=' * 60}\n")

        # 预热
        self._warmup(image_paths)

        # 初始化混淆矩阵
        confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf_thres)

        # 计时统计
        total_inference_ms = 0.0

        # ---- 逐批处理 ----
        pbar = tqdm(range(0, len(image_paths), self.batch_size),
                    desc="评估进度", unit="batch")
        for batch_start in pbar:
            batch_paths = image_paths[batch_start:batch_start + self.batch_size]

            # 1. 批量推理并计时
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

            # 2. 逐张图片处理
            for img_path, result in zip(batch_paths, results):
                self.image_count += 1
                boxes = result.boxes
                img_h, img_w = result.orig_shape  # 原始图片尺寸

                # 提取预测 — 确定预测张量所在设备
                if boxes is not None and len(boxes) > 0:
                    pred_xyxy = boxes.xyxy  # (N, 4) — 已在原始坐标空间
                    pred_conf = boxes.conf   # (N,)
                    pred_cls = boxes.cls     # (N,)
                    pred_device = pred_xyxy.device
                else:
                    pred_device = torch.device('cpu')
                    pred_xyxy = torch.empty((0, 4), device=pred_device)
                    pred_conf = torch.empty((0,), device=pred_device)
                    pred_cls = torch.empty((0,), device=pred_device)

                n_pred = len(pred_cls)

                # 加载真实标注 — 自动适配 YOLO/TXT 或 VOC/XML 格式
                label_path = self._resolve_label_path(img_path, labels_dir)
                gt_cls_np, gt_xyxy_np = self._load_labels(label_path, img_w, img_h)
                gt_cls = torch.from_numpy(gt_cls_np).to(pred_device)
                gt_xyxy = torch.from_numpy(gt_xyxy_np).to(pred_device)
                n_gt = len(gt_cls)

                self.total_targets += n_gt

                # 初始化该图片的 tp 矩阵
                tp = torch.zeros(n_pred, len(self.iouv), dtype=torch.bool, device=pred_device)

                if n_pred > 0 and n_gt > 0:
                    # 计算成对 IoU — 两个张量已在同一设备
                    iou = box_iou(gt_xyxy, pred_xyxy)  # (M, N)
                    # 匹配预测到真实标注
                    tp_np = match_predictions(
                        pred_cls, gt_cls, iou, self.iouv.to(pred_device),
                    )
                    tp = torch.from_numpy(tp_np).to(pred_device)

                # 累积统计 (保存到 CPU 以节省显存，最终汇总时再转)
                self.stats['tp'].append(tp.cpu())
                self.stats['conf'].append(pred_conf.cpu())
                self.stats['pred_cls'].append(pred_cls.cpu())
                self.stats['target_cls'].append(gt_cls.cpu())

                # 更新混淆矩阵
                if n_pred > 0 or n_gt > 0:
                    pred_for_cm = None
                    if n_pred > 0:
                        pred_for_cm = torch.cat([pred_xyxy, pred_conf.unsqueeze(1), pred_cls.unsqueeze(1)], dim=1)
                    confusion_matrix.process_batch(
                        pred_for_cm.cpu() if pred_for_cm is not None else None,
                        gt_xyxy.cpu(),
                        gt_cls.cpu(),
                    )

                # 更新进度条信息
                pbar.set_postfix({
                    '图片': self.image_count,
                    '目标': self.total_targets,
                    '速度': f'{batch_inference_ms / len(batch_paths):.1f}ms/张'
                })

            # 可选：保存标注图片
            if self.save_annotated:
                for r in results:
                    save_path = os.path.join(self.output_dir, os.path.basename(r.path))
                    r.save(filename=save_path)

        # ---- 汇总统计 ----
        pbar.close()
        print(f"\n正在计算指标...")

        # 拼接所有统计量
        tp_all = torch.cat(self.stats['tp'], dim=0).cpu().numpy()
        conf_all = torch.cat(self.stats['conf'], dim=0).cpu().numpy()
        pred_cls_all = torch.cat(self.stats['pred_cls'], dim=0).cpu().numpy()
        target_cls_all = torch.cat(self.stats['target_cls'], dim=0).cpu().numpy()

        # 计算 FPS
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

        # 如果没有检测结果
        if len(conf_all) == 0:
            print("\n⚠️  模型未在任何图片上检测到目标，无法计算指标。")
            print("   请尝试降低 conf_thres 参数。")
            self._print_fps_only(speed_info)
            return {'speed': speed_info, 'metrics': None}

        # 调用 ap_per_class 计算指标
        tp_out, fp_out, p, r, f1, ap, ap_class, p_curve, r_curve, f1_curve, x, prec_values = \
            ap_per_class(
                tp_all, conf_all, pred_cls_all, target_cls_all,
                plot=False,
                save_dir=Path(self.output_dir),
                names=self.names,
            )

        # 构建结构化结果
        results_data = {
            'speed': speed_info,
            'per_class': {},
            'mean': {},
        }

        # 每类指标
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

        # 总体均值
        results_data['mean'] = {
            'precision': round(float(p.mean()), 4),
            'recall': round(float(r.mean()), 4),
            'f1': round(float(f1.mean()), 4),
            'mAP50': round(float(ap50_per_class.mean()), 4),
            'mAP50_95': round(float(ap_per_class_mean.mean()), 4),
            'total_images': self.image_count,
            'total_instances': self.total_targets,
        }

        # 混淆矩阵
        results_data['confusion_matrix'] = confusion_matrix.matrix.tolist()

        # 打印结果
        self._print_mode_a_results(results_data, ap_class)
        self._export_results(results_data)

        return results_data

    # ------------------------------------------------------------------
    # 模式 B: FPS 基准测试 (无标注)
    # ------------------------------------------------------------------

    def evaluate_without_labels(self, image_paths: List[str]):
        """执行模式 B 评估 — FPS 基准测试。

        对图片进行批量推理，测量速度并统计各类检测数量。
        """
        print(f"\n{'=' * 60}")
        print(f"模式 B — FPS 基准测试 (无标注)")
        print(f"图片文件夹: {self.source}")
        print(f"图片数量:   {len(image_paths)}")
        print(f"置信度阈值: {self.conf_thres}")
        print(f"批处理大小: {self.batch_size}")
        print(f"{'=' * 60}\n")

        # 预热
        self._warmup(image_paths)

        # 统计
        total_inference_ms = 0.0
        per_class_counts = {self.names.get(i, f'class_{i}'): 0 for i in range(self.nc)}

        # ---- 逐批处理 ----
        pbar = tqdm(range(0, len(image_paths), self.batch_size),
                    desc="推理进度", unit="batch")
        for batch_start in pbar:
            batch_paths = image_paths[batch_start:batch_start + self.batch_size]

            # 推理并计时
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

            # 统计检测结果
            for result in results:
                self.image_count += 1
                if result.boxes is not None and len(result.boxes) > 0:
                    for cls_id in result.boxes.cls.int().tolist():
                        cls_name = self.names.get(cls_id, f'class_{cls_id}')
                        per_class_counts[cls_name] += 1

            # 可选：保存标注图片
            if self.save_annotated:
                for r in results:
                    save_path = os.path.join(self.output_dir, os.path.basename(r.path))
                    r.save(filename=save_path)

        pbar.close()

        # ---- 计算 FPS ----
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

        # 打印结果
        self._print_mode_b_results(results_data)
        self._export_results(results_data)

        return results_data

    # ------------------------------------------------------------------
    # 输出: 控制台打印
    # ------------------------------------------------------------------

    def _print_mode_a_results(self, results_data: Dict, ap_class: np.ndarray):
        """打印模式 A 的完整指标表格。"""
        speed = results_data['speed']
        mean = results_data['mean']

        print(f"\n{'=' * 70}")
        print(f"  批量评估结果 — 完整指标 (模式 A)")
        print(f"{'=' * 70}")
        print(f"  模型:       {os.path.basename(self.weights)}")
        print(f"  图片数:     {mean['total_images']}")
        print(f"  总实例数:   {mean['total_instances']}")
        print(f"  置信度:     {self.conf_thres}")
        print(f"  NMS IoU:    {self.iou_thres}")
        print(f"{'-' * 70}")

        # 表头
        header = f"  {'类别':<14s} {'实例数':>7s} {'P':>8s} {'R':>8s} {'F1':>8s} {'mAP50':>8s} {'mAP50-95':>9s}"
        print(header)
        print(f"  {'-' * 66}")

        # 每类行
        per_class = results_data['per_class']
        for cls_name in sorted(per_class.keys()):
            d = per_class[cls_name]
            print(f"  {cls_name:<14s} {d['instances']:>7d} {d['precision']:>8.4f} "
                  f"{d['recall']:>8.4f} {d['f1']:>8.4f} {d['ap50']:>8.4f} {d['ap50_95']:>9.4f}")

        # 总体均值行
        print(f"  {'-' * 66}")
        print(f"  {'总体 (mean)':<14s} {mean['total_instances']:>7d} {mean['precision']:>8.4f} "
              f"{mean['recall']:>8.4f} {mean['f1']:>8.4f} {mean['mAP50']:>8.4f} {mean['mAP50_95']:>9.4f}")

        # 速度
        print(f"{'-' * 70}")
        print(f"  推理速度:")
        print(f"    总时间:       {speed['total_time_s']:.2f} 秒")
        print(f"    平均/张:      {speed['avg_inference_ms']:.1f} ms")
        print(f"    FPS:          {speed['fps']:.1f} 张/秒")
        print(f"    批次大小:     {speed['batch_size']}")
        print(f"    设备:         {speed['device']}")
        print(f"{'-' * 70}")
        print(f"  结果已保存至: {self.output_dir}")
        print(f"{'=' * 70}\n")

    def _print_mode_b_results(self, results_data: Dict):
        """打印模式 B 的 FPS 和检测统计。"""
        speed = results_data['speed']
        counts = results_data['per_class_counts']

        print(f"\n{'=' * 70}")
        print(f"  FPS 基准测试结果 (模式 B — 无标注)")
        print(f"{'=' * 70}")
        print(f"  模型:       {os.path.basename(self.weights)}")
        print(f"  图片数:     {speed['num_images']}")
        print(f"  置信度:     {self.conf_thres}")
        print(f"  批次大小:   {speed['batch_size']}")
        print(f"{'-' * 70}")
        print(f"  推理速度:")
        print(f"    总时间:       {speed['total_time_s']:.2f} 秒")
        print(f"    平均/张:      {speed['avg_inference_ms']:.1f} ms")
        print(f"    FPS:          {speed['fps']:.1f} 张/秒")
        print(f"    设备:         {speed['device']}")
        print(f"{'-' * 70}")
        print(f"  各类检测数量:")
        for cls_name, count in sorted(counts.items()):
            if count > 0:
                print(f"    {cls_name:<20s}: {count:>6d}")
        print(f"{'-' * 70}")
        print(f"  结果已保存至: {self.output_dir}")
        print(f"{'=' * 70}\n")

    def _print_fps_only(self, speed_info: Dict):
        """无检测时仅打印速度信息。"""
        print(f"\n  推理速度 (无检测结果):")
        print(f"    总时间:       {speed_info['total_time_s']:.2f} 秒")
        print(f"    平均/张:      {speed_info['avg_inference_ms']:.1f} ms")
        print(f"    FPS:          {speed_info['fps']:.1f} 张/秒")

    # ------------------------------------------------------------------
    # 输出: 文件导出
    # ------------------------------------------------------------------

    def _export_results(self, results_data: Dict):
        """导出结果到 CSV 和/或 JSON 文件。"""
        fmt = self.save_format.lower()

        if fmt in ('csv', 'both'):
            self._export_csv(results_data)

        if fmt in ('json', 'both'):
            self._export_json(results_data)

    def _export_csv(self, results_data: Dict):
        """导出结果到 CSV 文件。"""
        pf = self.output_prefix  # 前缀

        # 模式 A 的结果
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
                # 总体均值行
                mean = results_data['mean']
                writer.writerow({
                    'class': '总体(mean)',
                    'instances': mean['total_instances'],
                    'precision': mean['precision'],
                    'recall': mean['recall'],
                    'f1': mean['f1'],
                    'ap50': mean['mAP50'],
                    'ap50_95': mean['mAP50_95'],
                })
            print(f"  CSV 指标文件: {csv_path}")

        # 速度信息 (两种模式都有)
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
        print(f"  CSV 速度文件: {csv_speed_path}")

    def _export_json(self, results_data: Dict):
        """导出结果到 JSON 文件。"""
        pf = self.output_prefix
        json_path = os.path.join(self.output_dir, f'{pf}evaluation_results.json')

        # 转换 numpy 类型为 Python 原生类型
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
        print(f"  JSON 结果文件: {json_path}")

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """执行评估流程。自动检测模式和标注格式并运行。"""
        # 1. 获取图片列表
        print(f"\n正在扫描图片文件夹: {self.source}")
        image_paths = get_image_paths(self.source)
        print(f"找到 {len(image_paths)} 张图片")

        # 2. 检测模式 (同时检测标注格式)
        mode = self.detect_mode(image_paths)
        labels_dir = find_labels_folder(self.source, self.config.get('labels'))

        # 3. 执行评估
        if mode == 'A' and labels_dir:
            fmt_name = {'txt': 'YOLO (.txt)', 'xml': 'Pascal VOC (.xml)'}.get(
                self._detected_label_format, self._detected_label_format)
            print(f"检测到标签文件夹: {labels_dir}")
            print(f"检测到标注格式:   {fmt_name}")
            return self.evaluate_with_labels(image_paths, labels_dir)
        else:
            print("\n⚠️  未检测到标注文件，将使用模式 B (仅测 FPS)")
            if labels_dir:
                print(f"   标签文件夹存在 ({labels_dir}) 但没有与图片匹配的标注文件")
            else:
                print(f"   未找到标签文件夹。检查了: "
                      f"{self.config.get('labels') or 'N/A'}、"
                      f"{os.path.join(self.source, 'labels')}、"
                      f"{os.path.join(self.source, 'annotations')}、"
                      f"{self.source}")
            print("   如需计算 P/R/F1/mAP，请确保每张图片有对应的 .txt (YOLO) 或 .xml (Pascal VOC) 标注文件。\n")
            return self.evaluate_without_labels(image_paths)


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_args() -> Dict:
    """解析命令行参数，返回覆盖 CONFIG 的字典。"""
    parser = argparse.ArgumentParser(
        description='YOLO 批量评估脚本 — 统计 Precision, Recall, F1, FPS 等指标',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python batch_evaluate.py --weights best.pt --source Data/test/ --conf 0.5
  python batch_evaluate.py --weights best.pt --source Data/test/ --labels Data/labels/ --batch 8
  python batch_evaluate.py --weights best.pt --source Data/test/ --labels-format xml
  python batch_evaluate.py  (使用 DEFAULT_CONFIG 中的参数)
        """
    )
    parser.add_argument('--weights', type=str, help='模型权重路径')
    parser.add_argument('--source', type=str, help='测试图片文件夹')
    parser.add_argument('--labels', type=str, default=None, help='标签文件夹 (可选)')
    parser.add_argument('--labels-format', type=str, dest='labels_format',
                        choices=['txt', 'xml', 'auto'], default=None,
                        help='标注格式: txt (YOLO), xml (Pascal VOC), auto (自动检测, 默认)')
    parser.add_argument('--conf', type=float, dest='conf_thres', help='置信度阈值')
    parser.add_argument('--iou', type=float, dest='iou_thres', help='NMS IoU 阈值')
    parser.add_argument('--img', type=int, dest='img_size', help='输入图像尺寸')
    parser.add_argument('--batch', type=int, dest='batch_size', help='批处理大小')
    parser.add_argument('--device', type=str, help='设备 (cpu, cuda:0, 等)')
    parser.add_argument('--output', type=str, dest='output_dir', help='输出目录')
    parser.add_argument('--save-img', action='store_true', dest='save_annotated',
                        help='保存标注图片')
    parser.add_argument('--format', type=str, dest='save_format',
                        choices=['csv', 'json', 'both'], help='保存格式')
    parser.add_argument('--prefix', type=str, dest='output_prefix', default=None,
                        help='输出文件名前缀 (留空则不添加)')
    parser.add_argument('--warmup', type=int, dest='warmup_iterations', help='预热迭代次数')

    args = parser.parse_args()
    # 只返回非 None 的参数
    return {k: v for k, v in vars(args).items() if v is not None}


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主入口。合并 DEFAULT_CONFIG 和命令行参数后执行评估。"""
    # 合并配置: 命令行参数优先
    config = DEFAULT_CONFIG.copy()
    cli_overrides = parse_args()
    config.update(cli_overrides)

    # 创建评估器并运行
    try:
        evaluator = BatchEvaluator(config)
        results = evaluator.run()
        return results
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("请检查 CONFIG 中的路径设置。")
        return None
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    """
    直接运行说明:

    1. 在上方的 DEFAULT_CONFIG 字典中修改参数
    2. 直接运行: python batch_evaluate.py
    3. 或用命令行覆盖个别参数: python batch_evaluate.py --conf 0.6

    输出文件:
      - evaluation_metrics.csv   : 每类 + 总体指标
      - evaluation_speed.csv     : FPS 速度信息
      - evaluation_results.json  : 完整结果 (选择 JSON 格式时)
    """
    main()
