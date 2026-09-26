# -*- coding: utf-8 -*-
"""
YOLO 训练脚本 —— 兼容 XML (Pascal VOC) 格式数据集。

功能：
  1. 输入数据集根目录（包含 images/ 和 annotations/ 两个文件夹）。
  2. 自动检测与 images/、annotations/ 并列的 train/val/test 目录：
     - 存在（如 train/images/ + train/annotations/）→ 直接使用
     - 不存在 → 自动按指定比例划分（默认 8:2:0）
  3. 将 Pascal VOC XML 标注转换为 YOLO 格式（归一化坐标），复制图像到临时目录。
  4. 生成数据集 YAML 配置文件。
  5. 启动 YOLO 训练，结果保存到 runs/detect/<YYMMDD>_<suffix>/。

使用方式：
  - 直接点击运行：修改下方 Config 配置区参数后，直接 Run 即可
  - 命令行运行：python train_xml.py --data "D:/Data/my_dataset" --epochs 200
  - 干跑验证：python train_xml.py --dry-run
"""

import argparse
import logging
import os
import random
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_xml")

# ── 常量 ──────────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "val", "test")


# ============================================================================
# ██  配置区 —— 修改以下参数后直接点击运行即可  ██
# ============================================================================
class Config:
    # ── 数据集 ──────────────────────────────────────────────────────────────
    # 数据集根目录（包含 images/ 和 annotations/ 两个文件夹）
    DATA = r"D:\Data\细胞图像及标签\260731-three-hq"

    # 若 images/annotations 下无 train/val/test 子目录，自动按比例划分
    AUTO_SPLIT = True                # 是否自动划分
    TRAIN_RATIO = 0.8                # 训练集比例
    VAL_RATIO = 0.2                  # 验证集比例（test = 1 - train - val）
    SEED = 42                        # 随机种子（保证划分可复现）

    # ── 模型 ────────────────────────────────────────────────────────────────
    MODEL = "ultralytics-main/ultralytics/cfg/models/v8/yolov8FEB.yaml"
    #MODEL = "ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml"
    PRETRAINED = None                # 预训练权重 .pt 路径，如 "yolov8n.pt"
    FREEZE = None                    # 冻结前 N 层，如 10

    # ── 训练超参 ────────────────────────────────────────────────────────────
    EPOCHS = 600
    BATCH = 4
    IMGSZ = 640
    DEVICE = "0"                     # "0"=GPU0, "cpu"=CPU
    WORKERS = 0                      # Windows 建议 0
    LR = 0.01                        # 初始学习率
    OPTIMIZER = "auto"               # auto / SGD / Adam / AdamW

    # ── 输出 ────────────────────────────────────────────────────────────────
    SUFFIX = "three-hq-FEB"     # 输出文件夹后缀，日期自动前缀
    PROJECT = "runs/detect"          # 训练结果保存的根目录

    # ── 运行模式 ────────────────────────────────────────────────────────────
    RESUME = False                   # 从上次 checkpoint 恢复
    DRY_RUN = False                  # True=仅验证不训练
    KEEP_TEMP = False                # True=保留临时转换文件


# ============================================================================
# 1. 参数解析（命令行会覆盖 Config 中的值）
# ============================================================================
def parse_args():
    """命令行参数优先级高于 Config。直接运行时使用 Config 中的值。"""
    parser = argparse.ArgumentParser(
        description="从 XML 数据集训练 YOLO 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="直接运行: 修改 Config 类后点击 Run\n"
               "命令行示例: python train_xml.py --data D:/Data/my_dataset --epochs 200",
    )

    # 数据集
    parser.add_argument("--data", default=Config.DATA, help="数据集根目录")
    parser.add_argument("--no-auto-split", action="store_true",
                        default=not Config.AUTO_SPLIT,
                        help="禁止自动划分，要求已存在 train/val/test")
    parser.add_argument("--train-ratio", type=float, default=Config.TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=Config.VAL_RATIO)
    parser.add_argument("--seed", type=int, default=Config.SEED)

    # 模型
    parser.add_argument("--model", default=Config.MODEL)
    parser.add_argument("--pretrained", default=Config.PRETRAINED)
    parser.add_argument("--freeze", type=int, default=Config.FREEZE)

    # 训练
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch", type=int, default=Config.BATCH)
    parser.add_argument("--imgsz", type=int, default=Config.IMGSZ)
    parser.add_argument("--device", default=Config.DEVICE)
    parser.add_argument("--workers", type=int, default=Config.WORKERS)
    parser.add_argument("--lr", type=float, default=Config.LR)
    parser.add_argument("--optimizer", default=Config.OPTIMIZER)

    # 输出 & 运行模式
    parser.add_argument("--suffix", default=Config.SUFFIX)
    parser.add_argument("--project", default=Config.PROJECT)
    parser.add_argument("--resume", action="store_true", default=Config.RESUME)
    parser.add_argument("--dry-run", action="store_true", default=Config.DRY_RUN)
    parser.add_argument("--keep-temp", action="store_true", default=Config.KEEP_TEMP)

    args = parser.parse_args()

    # 比例校验
    if args.train_ratio + args.val_ratio > 1.0:
        parser.error("train-ratio + val-ratio 不能大于 1.0")
    if args.train_ratio <= 0 or args.val_ratio < 0:
        parser.error("比例必须为非负，且 train-ratio > 0")

    return args


# ============================================================================
# 2. 数据集结构发现
# ============================================================================
def discover_dataset_structure(root_dir):
    """
    检测与 images/、annotations/ 并列的 train/val/test 目录，
    每个 split 下须有 images/ 和 annotations/ 子目录（如 train/images/ + train/annotations/）。
    存在 → 返回 {split: {images, annotations}}，否则 → None。
    """
    root = Path(root_dir)

    splits = {}
    has_pre_split = True

    for split_name in SPLIT_NAMES:
        img_dir = root / split_name / "images"
        ann_dir = root / split_name / "annotations"
        if img_dir.is_dir() and ann_dir.is_dir():
            splits[split_name] = {"images": str(img_dir), "annotations": str(ann_dir)}
        elif split_name == "test":
            splits[split_name] = None
        else:
            has_pre_split = False
            break

    if has_pre_split and "train" in splits and "val" in splits:
        logger.info("检测到预设的 train/val/test 目录结构（并列于 images/annotations），将直接使用")
        return splits

    logger.info("未检测到并列的 train/val/test 目录，将自动划分数据集")
    return None


# ============================================================================
# 3. 图像与标注配对
# ============================================================================
def pair_images_annotations(images_dir, annotations_dir):
    """扫描目录，按 stem 配对图片和 XML，返回 [(img_path, xml_path), ...]"""
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)

    image_files = {}
    for f in images_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            image_files[f.stem] = str(f)

    xml_files = {}
    for f in annotations_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".xml":
            xml_files[f.stem] = str(f)

    pairs = []
    for stem, img_path in image_files.items():
        if stem in xml_files:
            pairs.append((img_path, xml_files[stem]))
        else:
            logger.warning("图片缺少对应 XML: %s", img_path)

    for stem, xml_path in xml_files.items():
        if stem not in image_files:
            logger.warning("XML 缺少对应图片: %s", xml_path)

    logger.info("配对完成: %d 图片, %d XML, %d 有效配对",
                len(image_files), len(xml_files), len(pairs))
    return pairs


# ============================================================================
# 4. 数据集划分
# ============================================================================
def split_pairs(pairs, train_ratio, val_ratio, seed):
    """随机打乱后按比例划分。返回 {train: [...], val: [...], test: [...]}"""
    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    result = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
    logger.info("划分完成 (seed=%d): train=%d, val=%d, test=%d",
                seed, len(result["train"]), len(result["val"]), len(result["test"]))
    return result


# ============================================================================
# 5. XML → YOLO 转换
# ============================================================================
def convert_box(size, box):
    """VOC 绝对坐标 → YOLO 归一化坐标 (cx, cy, bw, bh)"""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    cx = (box[0] + box[1]) / 2.0
    cy = (box[2] + box[3]) / 2.0
    bw = box[1] - box[0]
    bh = box[3] - box[2]
    return (cx * dw, cy * dh, bw * dw, bh * dh)


def collect_classes(pairs):
    """扫描所有 XML，自动收集类别名（排序后返回列表）"""
    classes = set()
    for _, xml_path in pairs:
        try:
            tree = ET.parse(xml_path)
            for obj in tree.getroot().iter("object"):
                name_el = obj.find("name")
                if name_el is not None and name_el.text:
                    classes.add(name_el.text.strip())
        except ET.ParseError:
            logger.warning("XML 解析失败 (收集类别时跳过): %s", xml_path)
    return sorted(classes)


def convert_split(pairs, output_img_dir, output_lbl_dir, class_to_id):
    """转换一批配对：复制图片 + 写 YOLO .txt。返回成功数。"""
    output_img_dir = Path(output_img_dir)
    output_lbl_dir = Path(output_lbl_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for img_path, xml_path in pairs:
        img_path = Path(img_path)

        # 复制图片
        dst_img = output_img_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(str(img_path), str(dst_img))

        # 解析 XML
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
        except ET.ParseError:
            logger.warning("XML 解析失败: %s", xml_path)
            continue

        size_el = root.find("size")
        if size_el is None:
            logger.warning("XML 缺少 <size>: %s", xml_path)
            continue
        w_el, h_el = size_el.find("width"), size_el.find("height")
        if w_el is None or h_el is None:
            logger.warning("XML <size> 缺宽/高: %s", xml_path)
            continue
        w, h = int(w_el.text), int(h_el.text)

        # 写标签
        lines = []
        for obj in root.iter("object"):
            diff = obj.find("difficult")
            if diff is not None and diff.text is not None and diff.text.strip() == "1":
                continue

            name_el = obj.find("name")
            if name_el is None or name_el.text is None:
                continue
            cls_name = name_el.text.strip()
            if cls_name not in class_to_id:
                logger.warning("未知类别 '%s' → %s，已跳过", cls_name, xml_path)
                continue

            bbox_el = obj.find("bndbox")
            if bbox_el is None:
                continue
            try:
                xmin = float(bbox_el.find("xmin").text)
                xmax = float(bbox_el.find("xmax").text)
                ymin = float(bbox_el.find("ymin").text)
                ymax = float(bbox_el.find("ymax").text)
            except (AttributeError, ValueError):
                logger.warning("bndbox 数值异常: %s", xml_path)
                continue

            xmin = max(0.0, min(float(w), xmin))
            xmax = max(0.0, min(float(w), xmax))
            ymin = max(0.0, min(float(h), ymin))
            ymax = max(0.0, min(float(h), ymax))
            if xmax <= xmin or ymax <= ymin:
                logger.warning("无效 bbox: %s", xml_path)
                continue

            cls_id = class_to_id[cls_name]
            cx, cy, bw, bh = convert_box((w, h), (xmin, xmax, ymin, ymax))
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        with open(output_lbl_dir / f"{img_path.stem}.txt", "w", encoding="utf-8") as f:
            f.writelines(lines)
        converted += 1

    return converted


def build_yolo_dataset(pairs_by_split, temp_dir, class_to_id):
    """在 temp_dir 下构建 YOLO 格式的完整数据集。返回总样本数。"""
    temp_dir = Path(temp_dir)
    total = 0
    for split_name in ("train", "val", "test"):
        pairs = pairs_by_split.get(split_name)
        if not pairs:
            continue
        cnt = convert_split(
            pairs,
            str(temp_dir / split_name / "images"),
            str(temp_dir / split_name / "labels"),
            class_to_id,
        )
        logger.info("  %s: %d/%d 转换完成", split_name, cnt, len(pairs))
        total += cnt
    return total


# ============================================================================
# 6. 生成数据集 YAML
# ============================================================================
def generate_yaml(temp_dir, class_names):
    """生成 ultralytics 可读取的 dataset.yaml"""
    temp_dir = Path(temp_dir)
    yaml_path = temp_dir / "dataset.yaml"

    lines = [
        f"path: {temp_dir.as_posix()}\n",
        "train: train\n",
        "val: val\n",
    ]
    test_img = temp_dir / "test" / "images"
    if test_img.is_dir() and any(test_img.iterdir()):
        lines.append("test: test\n")
    lines.append(f"nc: {len(class_names)}\n")
    lines.append(f"names: {class_names}\n")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info("数据集 YAML 已生成: %s", yaml_path)
    return str(yaml_path)


# ============================================================================
# 7. 训练执行
# ============================================================================
def run_training(args, yaml_path):
    """执行训练，遇到 OOM 自动降 batch 重试（最多 3 次）"""
    output_name = f"{datetime.now():%y%m%d}_{args.suffix}"
    output_dir = f"{args.project}/{output_name}"

    logger.info("=" * 60)
    logger.info("输出: %s", output_dir)
    logger.info("模型: %s", args.model)
    logger.info("Epochs=%d  Batch=%d  ImgSz=%d  Device=%s",
                args.epochs, args.batch, args.imgsz, args.device)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY-RUN] 跳过实际训练")
        return None

    logger.info("加载模型: %s", args.model)
    model = YOLO(args.model)
    if args.pretrained:
        logger.info("加载预训练权重: %s", args.pretrained)
        model = model.load(args.pretrained)

    train_kwargs = dict(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=output_name,
        verbose=True,
        plots=True,
        save=True,
    )
    if args.lr is not None:
        train_kwargs["lr0"] = args.lr
    if args.optimizer != "auto":
        train_kwargs["optimizer"] = args.optimizer
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    if args.resume:
        train_kwargs["resume"] = True

    # OOM 重试
    current_batch = args.batch
    for attempt in range(4):
        try:
            logger.info("开始训练 (batch=%d, 第 %d 次尝试)", current_batch, attempt + 1)
            results = model.train(**train_kwargs)
            logger.info("训练完成: %s", output_dir)
            return results
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and current_batch > 1:
                current_batch = max(1, current_batch // 2)
                train_kwargs["batch"] = current_batch
                logger.warning("OOM! batch 降为 %d，重试中…", current_batch)
            else:
                logger.error("训练失败: %s", e)
                raise

    logger.error("batch=1 仍 OOM，训练终止")
    sys.exit(10)


# ============================================================================
# 8. 主流程
# ============================================================================
def main():
    args = parse_args()

    # ── 输入校验 ──
    root_dir = Path(args.data).resolve()
    if not root_dir.is_dir():
        logger.error("数据集根目录不存在: %s", root_dir)
        sys.exit(1)

    images_root = root_dir / "images"
    annotations_root = root_dir / "annotations"
    if not images_root.is_dir():
        logger.error("images 目录不存在: %s", images_root)
        sys.exit(2)
    if not annotations_root.is_dir():
        logger.error("annotations 目录不存在: %s", annotations_root)
        sys.exit(3)

    # ── 发现结构 → 配对 → 收集类别 ──
    pre_split = discover_dataset_structure(str(root_dir))

    if args.no_auto_split and pre_split is None:
        logger.error("--no-auto-split 已设置但未找到 train/val 子目录")
        sys.exit(4)

    if pre_split:
        pairs_by_split = {}
        for sn in ("train", "val", "test"):
            info = pre_split.get(sn)
            pairs_by_split[sn] = pair_images_annotations(info["images"], info["annotations"]) if info else []
        all_pairs = [p for v in pairs_by_split.values() for p in v]
    else:
        all_pairs = pair_images_annotations(str(images_root), str(annotations_root))
        if not all_pairs:
            logger.error("无有效图片-标注配对")
            sys.exit(5)
        pairs_by_split = split_pairs(all_pairs, args.train_ratio, args.val_ratio, args.seed)

    if not all_pairs:
        logger.error("无任何配对，退出")
        sys.exit(5)

    # ── 类别 ──
    class_names = collect_classes(all_pairs)
    if not class_names:
        logger.error("未从 XML 中发现任何类别")
        sys.exit(6)
    class_to_id = {n: i for i, n in enumerate(class_names)}
    logger.info("发现 %d 个类别: %s", len(class_names), class_names)

    # ── DRY-RUN 摘要 ──
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("[DRY-RUN] 数据集摘要")
        logger.info("  根目录: %s", root_dir)
        for sn in ("train", "val", "test"):
            p = pairs_by_split.get(sn, [])
            if p:
                logger.info("  %s: %d 对", sn, len(p))
        logger.info("  类别 (%d): %s", len(class_names), class_names)
        logger.info("  输出: %s/%s_%s/", args.project, f"{datetime.now():%y%m%d}", args.suffix)
        logger.info("  模型: %s", args.model)
        logger.info("  Epochs=%d  Batch=%d  ImgSz=%d", args.epochs, args.batch, args.imgsz)

    # ── 创建临时目录 → 转换 → YAML → 训练 ──
    script_dir = Path(__file__).resolve().parent
    temp_dir = tempfile.mkdtemp(prefix="xml_train_", dir=str(script_dir))
    logger.info("临时目录: %s", temp_dir)

    try:
        total = build_yolo_dataset(pairs_by_split, temp_dir, class_to_id)
        logger.info("共转换 %d 个样本", total)

        yaml_path = generate_yaml(temp_dir, class_names)

        if args.dry_run:
            logger.info("[DRY-RUN] 生成 YAML 内容:")
            for line in Path(yaml_path).read_text(encoding="utf-8").splitlines():
                logger.info("  %s", line)
            logger.info("[DRY-RUN] 完成（未执行训练）")
            return

        run_training(args, yaml_path)

    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        if args.keep_temp:
            logger.info("临时目录已保留: %s", temp_dir)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("临时目录已清理: %s", temp_dir)


if __name__ == "__main__":
    main()
