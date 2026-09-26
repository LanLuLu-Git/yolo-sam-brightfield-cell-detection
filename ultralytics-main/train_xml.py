# -*- coding: utf-8 -*-
"""
YOLO training script — compatible with XML (Pascal VOC) format datasets.

Features:
  1. Input a dataset root directory (containing images/ and annotations/ folders).
  2. Auto-detect train/val/test directories alongside images/ and annotations/:
     - Present (e.g. train/images/ + train/annotations/) → use directly
     - Absent → auto-split by the specified ratio (default 8:2:0)
  3. Convert Pascal VOC XML annotations to YOLO format (normalized coordinates), and copy images to a temp directory.
  4. Generate the dataset YAML config file.
  5. Start YOLO training, results saved to runs/detect/<YYMMDD>_<suffix>/.

Usage:
  - Run directly: modify the parameters in the Config section below, then just click Run
  - Command line: python train_xml.py --data "D:/Data/my_dataset" --epochs 200
  - Dry-run validation: python train_xml.py --dry-run
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

# ── Logging config ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_xml")

# ── Constants ──────────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "val", "test")


# ============================================================================
# ██  Config section —— modify the parameters below and click Run directly  ██
# ============================================================================
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────────
    # Dataset root directory (containing the images/ and annotations/ folders)
    DATA = r"D:\Data\细胞图像及标签\260731-three-hq"

    # If there is no train/val/test subdirectory under images/annotations, auto-split by ratio
    AUTO_SPLIT = True                # whether to auto-split
    TRAIN_RATIO = 0.8                # train set ratio
    VAL_RATIO = 0.2                  # validation set ratio (test = 1 - train - val)
    SEED = 42                        # random seed (makes the split reproducible)

    # ── Model ────────────────────────────────────────────────────────────────
    MODEL = "ultralytics-main/ultralytics/cfg/models/v8/yolov8FEB.yaml"
    #MODEL = "ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml"
    PRETRAINED = None                # pretrained weights .pt path, e.g. "yolov8n.pt"
    FREEZE = None                    # freeze the first N layers, e.g. 10

    # ── Training hyperparameters ────────────────────────────────────────────
    EPOCHS = 600
    BATCH = 4
    IMGSZ = 640
    DEVICE = "0"                     # "0"=GPU0, "cpu"=CPU
    WORKERS = 0                      # 0 recommended on Windows
    LR = 0.01                        # initial learning rate
    OPTIMIZER = "auto"               # auto / SGD / Adam / AdamW

    # ── Output ────────────────────────────────────────────────────────────────
    SUFFIX = "three-hq-FEB"     # output folder suffix, date is auto-prefixed
    PROJECT = "runs/detect"          # root directory where training results are saved

    # ── Run mode ────────────────────────────────────────────────────────────
    RESUME = False                   # resume from the last checkpoint
    DRY_RUN = False                  # True=validate only, no training
    KEEP_TEMP = False                # True=keep the temporary converted files


# ============================================================================
# 1. Argument parsing (command line overrides values in Config)
# ============================================================================
def parse_args():
    """Command-line arguments take priority over Config. Direct runs use the values in Config."""
    parser = argparse.ArgumentParser(
        description="Train a YOLO model from an XML dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Direct run: modify the Config class and click Run\n"
               "Command-line example: python train_xml.py --data D:/Data/my_dataset --epochs 200",
    )

    # Dataset
    parser.add_argument("--data", default=Config.DATA, help="dataset root directory")
    parser.add_argument("--no-auto-split", action="store_true",
                        default=not Config.AUTO_SPLIT,
                        help="disable auto-split, require train/val/test to already exist")
    parser.add_argument("--train-ratio", type=float, default=Config.TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=Config.VAL_RATIO)
    parser.add_argument("--seed", type=int, default=Config.SEED)

    # Model
    parser.add_argument("--model", default=Config.MODEL)
    parser.add_argument("--pretrained", default=Config.PRETRAINED)
    parser.add_argument("--freeze", type=int, default=Config.FREEZE)

    # Training
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch", type=int, default=Config.BATCH)
    parser.add_argument("--imgsz", type=int, default=Config.IMGSZ)
    parser.add_argument("--device", default=Config.DEVICE)
    parser.add_argument("--workers", type=int, default=Config.WORKERS)
    parser.add_argument("--lr", type=float, default=Config.LR)
    parser.add_argument("--optimizer", default=Config.OPTIMIZER)

    # Output & run mode
    parser.add_argument("--suffix", default=Config.SUFFIX)
    parser.add_argument("--project", default=Config.PROJECT)
    parser.add_argument("--resume", action="store_true", default=Config.RESUME)
    parser.add_argument("--dry-run", action="store_true", default=Config.DRY_RUN)
    parser.add_argument("--keep-temp", action="store_true", default=Config.KEEP_TEMP)

    args = parser.parse_args()

    # Ratio validation
    if args.train_ratio + args.val_ratio > 1.0:
        parser.error("train-ratio + val-ratio cannot be greater than 1.0")
    if args.train_ratio <= 0 or args.val_ratio < 0:
        parser.error("ratios must be non-negative, and train-ratio > 0")

    return args


# ============================================================================
# 2. Dataset structure discovery
# ============================================================================
def discover_dataset_structure(root_dir):
    """
    Detect train/val/test directories alongside images/ and annotations/.
    Each split must have images/ and annotations/ subdirectories (e.g. train/images/ + train/annotations/).
    Present → return {split: {images, annotations}}, otherwise → None.
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
        logger.info("Detected a pre-split train/val/test directory structure (alongside images/annotations), will use directly")
        return splits

    logger.info("No train/val/test directories detected alongside images/annotations, will auto-split the dataset")
    return None


# ============================================================================
# 3. Pairing images and annotations
# ============================================================================
def pair_images_annotations(images_dir, annotations_dir):
    """Scan directories, pair images and XML by stem, return [(img_path, xml_path), ...]"""
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
            logger.warning("Image missing corresponding XML: %s", img_path)

    for stem, xml_path in xml_files.items():
        if stem not in image_files:
            logger.warning("XML missing corresponding image: %s", xml_path)

    logger.info("Pairing done: %d images, %d XML, %d valid pairs",
                len(image_files), len(xml_files), len(pairs))
    return pairs


# ============================================================================
# 4. Dataset split
# ============================================================================
def split_pairs(pairs, train_ratio, val_ratio, seed):
    """Shuffle randomly then split by ratio. Return {train: [...], val: [...], test: [...]}"""
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
    logger.info("Split done (seed=%d): train=%d, val=%d, test=%d",
                seed, len(result["train"]), len(result["val"]), len(result["test"]))
    return result


# ============================================================================
# 5. XML → YOLO conversion
# ============================================================================
def convert_box(size, box):
    """VOC absolute coordinates → YOLO normalized coordinates (cx, cy, bw, bh)"""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    cx = (box[0] + box[1]) / 2.0
    cy = (box[2] + box[3]) / 2.0
    bw = box[1] - box[0]
    bh = box[3] - box[2]
    return (cx * dw, cy * dh, bw * dw, bh * dh)


def collect_classes(pairs):
    """Scan all XML, auto-collect class names (return a sorted list)"""
    classes = set()
    for _, xml_path in pairs:
        try:
            tree = ET.parse(xml_path)
            for obj in tree.getroot().iter("object"):
                name_el = obj.find("name")
                if name_el is not None and name_el.text:
                    classes.add(name_el.text.strip())
        except ET.ParseError:
            logger.warning("XML parsing failed (skipped while collecting classes): %s", xml_path)
    return sorted(classes)


def convert_split(pairs, output_img_dir, output_lbl_dir, class_to_id):
    """Convert a batch of pairs: copy images + write YOLO .txt. Return the success count."""
    output_img_dir = Path(output_img_dir)
    output_lbl_dir = Path(output_lbl_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for img_path, xml_path in pairs:
        img_path = Path(img_path)

        # Copy image
        dst_img = output_img_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(str(img_path), str(dst_img))

        # Parse XML
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
        except ET.ParseError:
            logger.warning("XML parsing failed: %s", xml_path)
            continue

        size_el = root.find("size")
        if size_el is None:
            logger.warning("XML missing <size>: %s", xml_path)
            continue
        w_el, h_el = size_el.find("width"), size_el.find("height")
        if w_el is None or h_el is None:
            logger.warning("XML <size> missing width/height: %s", xml_path)
            continue
        w, h = int(w_el.text), int(h_el.text)

        # Write labels
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
                logger.warning("Unknown class '%s' → %s, skipped", cls_name, xml_path)
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
                logger.warning("bndbox has invalid values: %s", xml_path)
                continue

            xmin = max(0.0, min(float(w), xmin))
            xmax = max(0.0, min(float(w), xmax))
            ymin = max(0.0, min(float(h), ymin))
            ymax = max(0.0, min(float(h), ymax))
            if xmax <= xmin or ymax <= ymin:
                logger.warning("Invalid bbox: %s", xml_path)
                continue

            cls_id = class_to_id[cls_name]
            cx, cy, bw, bh = convert_box((w, h), (xmin, xmax, ymin, ymax))
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        with open(output_lbl_dir / f"{img_path.stem}.txt", "w", encoding="utf-8") as f:
            f.writelines(lines)
        converted += 1

    return converted


def build_yolo_dataset(pairs_by_split, temp_dir, class_to_id):
    """Build a complete YOLO-format dataset under temp_dir. Return the total sample count."""
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
        logger.info("  %s: %d/%d converted", split_name, cnt, len(pairs))
        total += cnt
    return total


# ============================================================================
# 6. Generate dataset YAML
# ============================================================================
def generate_yaml(temp_dir, class_names):
    """Generate a dataset.yaml readable by ultralytics"""
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

    logger.info("Dataset YAML generated: %s", yaml_path)
    return str(yaml_path)


# ============================================================================
# 7. Training execution
# ============================================================================
def run_training(args, yaml_path):
    """Run training, auto-reduce batch and retry on OOM (up to 3 times)"""
    output_name = f"{datetime.now():%y%m%d}_{args.suffix}"
    output_dir = f"{args.project}/{output_name}"

    logger.info("=" * 60)
    logger.info("Output: %s", output_dir)
    logger.info("Model: %s", args.model)
    logger.info("Epochs=%d  Batch=%d  ImgSz=%d  Device=%s",
                args.epochs, args.batch, args.imgsz, args.device)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY-RUN] skipping actual training")
        return None

    logger.info("Loading model: %s", args.model)
    model = YOLO(args.model)
    if args.pretrained:
        logger.info("Loading pretrained weights: %s", args.pretrained)
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

    # OOM retry
    current_batch = args.batch
    for attempt in range(4):
        try:
            logger.info("Starting training (batch=%d, attempt %d)", current_batch, attempt + 1)
            results = model.train(**train_kwargs)
            logger.info("Training finished: %s", output_dir)
            return results
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and current_batch > 1:
                current_batch = max(1, current_batch // 2)
                train_kwargs["batch"] = current_batch
                logger.warning("OOM! batch reduced to %d, retrying...", current_batch)
            else:
                logger.error("Training failed: %s", e)
                raise

    logger.error("still OOM at batch=1, terminating training")
    sys.exit(10)


# ============================================================================
# 8. Main flow
# ============================================================================
def main():
    args = parse_args()

    # ── Input validation ──
    root_dir = Path(args.data).resolve()
    if not root_dir.is_dir():
        logger.error("Dataset root directory does not exist: %s", root_dir)
        sys.exit(1)

    images_root = root_dir / "images"
    annotations_root = root_dir / "annotations"
    if not images_root.is_dir():
        logger.error("images directory does not exist: %s", images_root)
        sys.exit(2)
    if not annotations_root.is_dir():
        logger.error("annotations directory does not exist: %s", annotations_root)
        sys.exit(3)

    # ── Discover structure → pair → collect classes ──
    pre_split = discover_dataset_structure(str(root_dir))

    if args.no_auto_split and pre_split is None:
        logger.error("--no-auto-split is set but no train/val subdirectories were found")
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
            logger.error("no valid image-annotation pairs")
            sys.exit(5)
        pairs_by_split = split_pairs(all_pairs, args.train_ratio, args.val_ratio, args.seed)

    if not all_pairs:
        logger.error("no pairs at all, exiting")
        sys.exit(5)

    # ── Classes ──
    class_names = collect_classes(all_pairs)
    if not class_names:
        logger.error("no classes found in the XML")
        sys.exit(6)
    class_to_id = {n: i for i, n in enumerate(class_names)}
    logger.info("Found %d classes: %s", len(class_names), class_names)

    # ── DRY-RUN summary ──
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("[DRY-RUN] dataset summary")
        logger.info("  Root directory: %s", root_dir)
        for sn in ("train", "val", "test"):
            p = pairs_by_split.get(sn, [])
            if p:
                logger.info("  %s: %d pairs", sn, len(p))
        logger.info("  Classes (%d): %s", len(class_names), class_names)
        logger.info("  Output: %s/%s_%s/", args.project, f"{datetime.now():%y%m%d}", args.suffix)
        logger.info("  Model: %s", args.model)
        logger.info("  Epochs=%d  Batch=%d  ImgSz=%d", args.epochs, args.batch, args.imgsz)

    # ── Create temp dir → convert → YAML → train ──
    script_dir = Path(__file__).resolve().parent
    temp_dir = tempfile.mkdtemp(prefix="xml_train_", dir=str(script_dir))
    logger.info("Temp directory: %s", temp_dir)

    try:
        total = build_yolo_dataset(pairs_by_split, temp_dir, class_to_id)
        logger.info("Converted %d samples in total", total)

        yaml_path = generate_yaml(temp_dir, class_names)

        if args.dry_run:
            logger.info("[DRY-RUN] generated YAML content:")
            for line in Path(yaml_path).read_text(encoding="utf-8").splitlines():
                logger.info("  %s", line)
            logger.info("[DRY-RUN] finished (training not executed)")
            return

        run_training(args, yaml_path)

    except KeyboardInterrupt:
        logger.info("interrupted by user")
    finally:
        if args.keep_temp:
            logger.info("Temp directory kept: %s", temp_dir)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Temp directory cleaned up: %s", temp_dir)


if __name__ == "__main__":
    main()
