import os
import glob
from ultralytics import YOLO
import time
from tqdm import tqdm

# -----------------------------# Centralized area of editable parameters# -----------------------------# Default config parameters# Modify parameters directly here to run, no command-line input needed
DEFAULT_CONFIG = {
    'weights': 'runs/260721前/260107-C2C12-293-YOLOv8n/weights/best.pt',  # model weights file path
    'source': r"D:\Data\实验结果汇总\识别对比图\C2C12-HEK293\带探针\251223-C2C12-HEK293-86.tif",             # input image path or folder, modify directly here
    'output_dir': r"D:\Data\实验结果汇总\识别对比图\C2C12-HEK293\带探针",      # root directory for saving results
    'conf_thres': 0.2,             # confidence threshold
    'iou_thres': 0.5,             # IoU threshold
    'img_size': 640,               # input image size
    'device': 'cuda:0',                  # device selection, empty string auto-selects, 'cpu' or 'cuda:0'
    'save_crop': False,            # whether to save cropped objects
    'show_labels': True,           # whether to show labels
    'show_conf': True,             # whether to show confidence
    'agnostic_nms': False,         # whether to enable class-agnostic NMS
    'classes': None,               # specify detection classes, None means all classes, e.g. [0, 1, 2]
    'batch_size': 4,               # batch size
    'result_suffix': '-yolov8nFEB'   # custom suffix for detection result images, None means no suffix
}

class Args:
    """Create a class that simulates command-line arguments"""
    def __init__(self, config):
        self.weights = config['weights']
        self.source = config['source']
        self.output = config['output_dir']
        self.conf = config['conf_thres']
        self.iou = config['iou_thres']
        self.img_size = config['img_size']
        self.device = config['device']
        self.classes = config['classes']
        self.agnostic_nms = config['agnostic_nms']
        self.save_crop = config['save_crop']
        self.show_labels = config['show_labels']
        self.show_conf = config['show_conf']
        self.batch_size = config['batch_size']
        self.result_suffix = config['result_suffix']

def get_image_paths(source_path):
    """Get all image paths that need detection
    Args:
        source_path: input image path or folder
    Returns:
        list: list of image paths
    """
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    # If it's a file
    if os.path.isfile(source_path):
        # Check whether the file extension is a supported image format
        ext = os.path.splitext(source_path)[1].lower()
        if ext in image_extensions:
            return [source_path]
        else:
            print(f"Warning: file {source_path} is not a supported image format")
            return []
    # If it's a folder
    elif os.path.isdir(source_path):
        image_paths = []
        # Iterate over all supported-format images in the folder
        for ext in image_extensions:
            # Use glob to find all files matching the extension
            pattern = os.path.join(source_path, f'*{ext}')
            image_paths.extend(glob.glob(pattern))
            # Find images in subfolders
            pattern = os.path.join(source_path, f'**/*{ext}')
            image_paths.extend(glob.glob(pattern, recursive=True))
        # Deduplicate and sort
        image_paths = sorted(list(set(image_paths)))
        print(f"Found {len(image_paths)} images in folder {source_path}")
        return image_paths
    else:
        print(f"Error: path {source_path} does not exist")
        return []

def batch_process_images(model, image_paths, args):
    """Process images in batches
    Args:
        model: YOLO model object
        image_paths: list of image paths
        args: argument object
    """
    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print(f"Start processing {len(image_paths)} images...")
    start_time = time.time()

    # Process images in batches
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Processing"):
        # Get the current batch of images
        batch_paths = image_paths[i:i+args.batch_size]

        # Run detection
        results = model(batch_paths,
                        conf=args.conf,
                        iou=args.iou,
                        classes=args.classes,
                        agnostic_nms=args.agnostic_nms,
                        save=True,
                        save_crop=args.save_crop,
                        show_labels=args.show_labels,
                        show_conf=args.show_conf,
                        project=args.output,
                        name='',  # do not create a subdirectory
                        exist_ok=True,
                        imgsz=args.img_size,
                        device=args.device)

        # Record the detection result for each image and handle the custom suffix
        for img_path, result in zip(batch_paths, results):
            img_name = os.path.basename(img_path)
            boxes = result.boxes  # detection box info

            # Apply the custom suffix
            if args.result_suffix:
                # Get the original filename and extension
                base_name, ext = os.path.splitext(img_name)
                # Build the new filename
                new_name = f"{base_name}{args.result_suffix}{ext}"
                # Build the full source and destination paths
                src_path = os.path.join(args.output, img_name)
                dst_path = os.path.join(args.output, new_name)

                # Rename the file
                if os.path.exists(src_path):
                    try:
                        os.rename(src_path, dst_path)
                        tqdm.write(f"Image {img_name} renamed to {new_name}")
                    except Exception as e:
                        tqdm.write(f"Error renaming image {img_name}: {e}")

            if len(boxes) > 0:
                tqdm.write(f"Image {img_name}: detected {len(boxes)} objects")

    total_time = time.time() - start_time
    print(f"\nDetection complete!")
    print(f"Total processing time: {total_time:.2f} s")
    print(f"Average processing speed: {total_time/len(image_paths):.2f} s/image")
    print(f"Results saved in: {os.path.abspath(args.output)}")

def main(config=None):
    """Main function
    Args:
        config: configuration dict, use the default config if None
    """
    # Use the passed config or the default config
    if config is None:
        config = DEFAULT_CONFIG

    # Create the argument object
    args = Args(config)

    # Verify that the weights file exists
    if not os.path.isfile(args.weights):
        print(f"Error: weights file {args.weights} does not exist")
        return

    # Verify that the image source exists
    if not (os.path.isfile(args.source) or os.path.isdir(args.source)):
        print(f"Error: image source {args.source} does not exist")
        return

    # Load the model
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)
    print(f"Model loaded, using device: {model.device}")

    # Get image paths
    image_paths = get_image_paths(args.source)
    if not image_paths:
        print("No images to detect were found")
        return

    # Process images in batches
    batch_process_images(model, image_paths, args)

if __name__ == '__main__':
    """
    Direct run instructions:

    1. Modify the parameters in the DEFAULT_CONFIG dictionary above
    2. Run this script directly to start detection

    Common parameter descriptions:
    - weights: model weights file path
    - source: input image path or folder
    - output_dir: result save directory
    - conf_thres: confidence threshold, range 0-1
    - batch_size: batch size, adjust according to computer performance
    - device: device selection, 'cpu' or 'cuda:0'
    - result_suffix: custom suffix for detection result images, None means no suffix, default is '_detected'
    """
    main()