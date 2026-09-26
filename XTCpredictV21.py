import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import logging
import time
from datetime import datetime
import torch
from segment_anything import sam_model_registry, SamPredictor

import sys
# Important: put the ultralytics-main path first so the locally modified package is used
sys.path.insert(0, r'D:/Project/YOLOwithSAM/ultralytics-main')
site_packages_path = r'D:/APP/Miniconda3/envs/yolosam/lib/site-packages'
if site_packages_path in sys.path:
    sys.path.remove(site_packages_path)
# Add it back at the end
sys.path.append(site_packages_path)

from ultralytics import YOLO
from ultralytics.change_model import FEB

"""2025-12-12 v2.0 improvements:
Structure and encapsulation
    Use the XTCPredictConfig class to centralize all configuration parameters
    Use the XTCPredictor class to encapsulate all business logic, improving cohesion
    Split large functions into several small methods, each with a single responsibility
Logging system
    Replace print with the logging module, supporting output to both console and file
    Log at multiple levels (info, warning, error, critical)
    Record detailed error stack traces for easier debugging
Code readability
    Improve docstrings describing the function, parameters, and return values of classes and methods
    Use clearer variable and method naming, following PEP 8
    Use constants instead of hardcoded values to improve maintainability
Error handling
    Add more comprehensive exception catching and handling
    Add detailed logs for operations that may fail
    Separate error handling from business logic
Extensibility
    Centralized configuration for easy parameter modification
    Modular design for easy addition of new features
    Clear method boundaries for easier unit testing
Modifications and improvements
    Add image exception handling
    Fix the color mapping mismatch
251219: fix exception handling, add deletion of exceptional tif images
       change the distance threshold from 5e-5 to 1e-4
"""
# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xtc_predict.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set the font for Chinese text
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class XTCPredictConfig:
    """Configuration class that centralizes all parameters."""
    def __init__(self):
        # Path configuration
        self.input_image_dir = "Data/测试对比/C2C12-293/原图"
        self.result_image_dir = 'Data/测试对比/C2C12-293/识别结果/yolov8n-FEB'
        self.result_data_dir = "Data/测试对比/C2C12-293/识别结果/yolov8n-FEB"
        self.yolo_weights_path = 'runs/260721前/260314-YOLOv8FEB-C2C12-293-400/weights/best.pt'
        self.sam_checkpoint = os.path.join(os.path.dirname(__file__),
                                          'segment-anything-main/models/sam_vit_h_4b8939.pth')
        self.template_path = os.path.join(os.path.dirname(__file__), "tp15.png")

        # Model configuration
        self.sam_model_type = 'vit_h'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Processing configuration
        self.check_interval = 2  # interval (seconds) to check for new files
        self.coordinate_scale = 1e-7 * 2*0.955  # coordinate conversion factor
        self.distance_threshold = 5e-5  # distance threshold
        self.yolo_confidence = 0.38  # YOLO detection confidence threshold (0-1)
        self.supported_extensions = ['.tif', '.jpg', '.jpeg', '.png']  # supported image extensions
        self.enable_template_matching = False  # whether to enable probe detection via template matching
        self.enable_csv_output = False  # whether to generate CSV results
        self.save_original_image = False  # whether to save a backup of the original image


class XTCPredictor:
    """Main XTC predictor class, encapsulating all processing logic."""
    def __init__(self, config: XTCPredictConfig):
        self.config = config
        self.sam_predictor = self._init_sam_predictor()
        self.yolo_model = self._init_yolo_model()
     # Predefined color mapping
        self.fixed_colors = np.array([
            [1.0, 0.0, 0.0, 0.6],  # red
            [0.0, 1.0, 0.0, 0.6],  # green
            [0.0, 0.0, 1.0, 0.6],  # blue
            [1.0, 1.0, 0.0, 0.6],  # yellow
            [1.0, 0.0, 1.0, 0.6],  # purple
            [0.0, 1.0, 1.0, 0.6]   # cyan
            ])

    def _init_sam_predictor(self) -> SamPredictor: # initialize the SAM predictor
        """Initialize the SAM predictor."""
        try:
            sam = sam_model_registry[self.config.sam_model_type](
                checkpoint=self.config.sam_checkpoint
            )
            sam.to(device=self.config.device)
            logger.info(f"SAM model initialized, using device: {self.config.device}")
            return SamPredictor(sam)
        except Exception as e:
            logger.error(f"SAM model initialization failed: {str(e)}", exc_info=True)
            raise

    def _init_yolo_model(self) -> YOLO: # initialize the YOLO model
        """Initialize the YOLO model."""
        try:
            model = YOLO(self.config.yolo_weights_path)
            logger.info("YOLO model initialized")
            return model
        except Exception as e:
            logger.error(f"YOLO model initialization failed: {str(e)}", exc_info=True)
            raise

    def perform_yolo_detection(self, image_path: str): # run YOLO object detection
        """Run YOLO object detection."""
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return None

        try:
            results = self.yolo_model(image_path, device=self.config.device, conf=self.config.yolo_confidence)
            logger.info(f"YOLO detection done, image: {image_path}, confidence threshold: {self.config.yolo_confidence}")
            return results
        except Exception as e:
            logger.error(f"YOLO detection failed: {str(e)}", exc_info=True)
            return None

    def perform_sam_segmentation(self, image_path: str, yolo_results) -> tuple: # run SAM image segmentation
        """Run SAM image segmentation."""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot read image: {image_path}")
            return None, None

        try:
            self.sam_predictor.set_image(image)


            # Create the color mapping dictionary
            class_colors = {
                class_id: self.fixed_colors[class_id % len(self.fixed_colors)]
                for class_id in yolo_results[0].names
            }

            # Initialize results
            combined_mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
            centroids = []

            # Process each detection result
            for r in yolo_results:
                boxes = r.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy()

                for i, box in enumerate(xyxy):
                    # Prepare the input box
                    input_box = np.array([box])

                    # Predict using the bounding box
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box,
                        multimask_output=True,
                    )

                    # Find the mask with the highest confidence
                    best_mask = masks[np.argmax(scores)]

                    # Compute the object centroid coordinates
                    mask_indices = np.argwhere(best_mask)
                    centroid = np.mean(mask_indices, axis=0).astype(int)[::-1]  # convert to (x, y) format
                    centroids.append(centroid)

                    # Get the class ID and color of the current object
                    class_id = int(cls[i])
                    color = class_colors[class_id]

                    # Build the RGBA representation of the mask
                    mask_rgba = np.zeros((best_mask.shape[0], best_mask.shape[1], 4), dtype=np.float32)
                    mask_rgba[best_mask] = color

                    # Merge into the combined mask
                    combined_mask = np.where(mask_rgba[..., 3:4] > 0, mask_rgba, combined_mask)

            logger.info("SAM segmentation complete")
            return combined_mask, centroids

        except Exception as e:
            logger.error(f"SAM segmentation failed: {str(e)}", exc_info=True)
            return None, None

    def _find_sphere_probe(self, image) -> tuple: # find Sphere_Probe coordinates via template matching
        """Find Sphere_Probe coordinates using template matching."""
        if not os.path.exists(self.config.template_path):
            logger.warning(f"Template file does not exist: {self.config.template_path}")
            return None

        try:
            template = cv2.imread(self.config.template_path)
            if template is None:
                logger.warning(f"Cannot read template file: {self.config.template_path}")
                return None

            # Perform template matching
            res = cv2.matchTemplate(image[:,:,0], template[:,:,0], cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)

            # Compute the center point
            center_x = max_loc[0] + template.shape[1] // 2
            center_y = max_loc[1] + template.shape[0] // 2

            logger.info("Sphere_Probe detection complete")
            return (center_x, center_y)

        except Exception as e:
            logger.error(f"Sphere_Probe detection failed: {str(e)}", exc_info=True)
            return None

    def post_process_image(self, image_path: str, yolo_results, combined_mask, centroids) -> bool: # post-process the image and save results
        """Post-process the image and save results."""
        # Check whether the results are valid
        if yolo_results is None or combined_mask is None or centroids is None or len(centroids) == 0:
            logger.error("Post-processing failed: incomplete input parameters")
            return False

        try:
            # Constant definitions
            CIRCLE_COLOR = (0, 0, 255)
            TEXT_COLOR = (0, 0, 0)
            FONT = cv2.FONT_HERSHEY_SIMPLEX
            FONT_SCALE = 1.5
            THICKNESS = 3

            # Create a date-named folder
            today_date = datetime.now().strftime("%Y-%m-%d")
            today_folder = os.path.join(self.config.result_image_dir, today_date)
            os.makedirs(today_folder, exist_ok=True)

            # Read the original image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Cannot read image: {image_path}")
                return False

            # Image overlay processing
            image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
            result = (image_rgba * 0.7 + combined_mask * 0.3) * 255
            result = result.astype(np.uint8)
            # Convert RGBA back to BGR so OpenCV saves it correctly
            result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)

            # Get the YOLO detection boxes
            xyxy = []
            for r in yolo_results:
                xyxy.extend(r.boxes.xyxy.cpu().numpy())

            # Find the Sphere_Probe coordinates
            sphere_probe_coords = None
            if self.config.enable_template_matching:
                sphere_probe_coords = self._find_sphere_probe(image)

            # Save CSV data
            if self.config.enable_csv_output:
                self._save_centroid_data(sphere_probe_coords, yolo_results, centroids, xyxy)

            # Annotate centroids and relative positions
            self._draw_annotations(result, sphere_probe_coords, centroids, yolo_results)

            # Save the result image
            self._save_processed_images(result, image, image_path, today_folder)

            logger.info("Image post-processing complete")
            return True

        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}", exc_info=True)
            return False

    def _save_centroid_data(self, sphere_coords, yolo_results, centroids, xyxy, write_header=False): # save centroid coordinates to CSV
        """Save centroid coordinate data to CSV."""
        try:
            os.makedirs(self.config.result_data_dir, exist_ok=True)
            csv_path = os.path.join(self.config.result_data_dir, "resultdata.csv")

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['Cell_Name', 'X_Distance', 'Y_Distance'])

                if sphere_coords is not None:
                    for i, (box, centroid) in enumerate(zip(xyxy, centroids)):
                        class_id = int(yolo_results[0].boxes.cls.cpu().numpy()[i])
                        class_name = yolo_results[0].names[class_id]

                        if class_name not in ['Sphere_Probe', 'Dead']:
                            x_dist = -(centroid[0] - sphere_coords[0]) * self.config.coordinate_scale
                            y_dist = -(centroid[1] - sphere_coords[1]) * self.config.coordinate_scale

                            if abs(x_dist) < self.config.distance_threshold and abs(y_dist) < self.config.distance_threshold:
                                writer.writerow([class_name, y_dist, x_dist])
                else:
                    logger.warning("Sphere_Probe not detected, skipping CSV write")

            logger.info(f"Coordinate data saved to: {csv_path}")

        except PermissionError as e:
            logger.error(f"Cannot write CSV file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to save CSV data: {str(e)}", exc_info=True)
            raise

    def _draw_annotations(self, result, sphere_coords, centroids, yolo_results): # draw annotations on the image
        """Draw annotations on the image."""
        # Mark the Sphere_Probe
        if sphere_coords is not None:
            cv2.circle(result, sphere_coords, 5, (0, 0, 255), -1)
            cv2.putText(
                result,
                "Sphere_Probe",
                (sphere_coords[0]+10, sphere_coords[1]+10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

        # Mark the cell centroids and information
        for i, centroid in enumerate(centroids):
            class_id = int(yolo_results[0].boxes.cls.cpu().numpy()[i])
            class_name = yolo_results[0].names[class_id]

            # Draw the centroid
            cv2.circle(result, tuple(centroid), 5, (0, 0, 255), -1)

            """
            # Draw the label
            if sphere_coords is not None:
                x_dist = (centroid[0] - sphere_coords[0])
                y_dist = (centroid[1] - sphere_coords[1])
                label = f"{class_name}: ({x_dist:.2f}, {y_dist:.2f})"
                cv2.putText(
                    result,
                    label,
                    (centroid[0]+10, centroid[1]+10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )
            """

        # Add a timestamp
        #current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #cv2.putText(
        #    result,
        #    current_time,
        #    (20, 50),
        #    cv2.FONT_HERSHEY_SIMPLEX,
        #    1.5,
        #    (0, 0, 0),
        #    3
        #)

    def _save_processed_images(self, result, original_image, image_path, save_folder): # save the processed image and original backup
        """Save the processed image and a backup of the original."""
        # Generate the timestamp and filename
        original_filename = os.path.basename(image_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name, file_ext = os.path.splitext(original_filename)

        # Save the original backup
        if self.config.save_original_image:
            original_save_path = os.path.join(save_folder, f"original_{timestamp}_{file_name}{file_ext}")
            cv2.imwrite(original_save_path, original_image)
            logger.info(f"Original backup: {original_save_path}")

        # Save the processed image
        processed_save_path = os.path.join(save_folder, f"processed_{timestamp}_{file_name}{file_ext}")
        cv2.imwrite(processed_save_path, result)

        # Save to the default result path
        default_result_path = os.path.join(self.config.result_image_dir, "resultimage.jpg")
        cv2.imwrite(default_result_path, result)

        logger.info(f"Processed image: {processed_save_path}")

    def safe_delete_file(self, file_path: str) -> bool: # safely delete a file
        """Safely delete a file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.info(f"File does not exist, nothing to delete: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete file: {str(e)}", exc_info=True)
            return False

    def convert_tif_to_jpg(self, tif_path: str) -> str: # convert a tif image to jpg and save it
        """Convert a tif image to jpg format and save it.

        Args:
            tif_path: path of the tif image

        Returns:
            str: path of the converted jpg image, or None if conversion fails
        """
        try:
            if not tif_path.lower().endswith('.tif'):
                logger.warning(f"Not a tif image: {tif_path}")
                return tif_path  # return the original path if it is not a tif image

            # Generate the jpg save path
            jpg_path = os.path.splitext(tif_path)[0] + '.jpg'

            # Read the tif image
            img = cv2.imread(tif_path)
            if img is None:
                logger.error(f"Cannot read tif image: {tif_path}")
                return None

            # Save as jpg format
            success = cv2.imwrite(jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not success:
                logger.error(f"Cannot save jpg image: {jpg_path}")
                return None

            logger.info(f"tif image converted successfully: {tif_path} -> {jpg_path}")
            return jpg_path

        except Exception as e:
            logger.error(f"tif to jpg conversion failed: {str(e)}", exc_info=True)
            return None

    def generate_empty_csv(self) -> bool: # generate an empty CSV file
        """Generate an empty CSV file.

        Returns:
            bool: whether the empty CSV file was generated successfully
        """
        try:
            os.makedirs(self.config.result_data_dir, exist_ok=True)
            csv_path = os.path.join(self.config.result_data_dir, "resultdata.csv")

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Cell_Name', 'X_Distance', 'Y_Distance'])

            logger.info(f"Empty CSV file generated: {csv_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate empty CSV file: {str(e)}", exc_info=True)
            return False

    def check_file_exists(self, path: str) -> bool: # check whether a file exists
        """Check whether a file exists."""
        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")
            return False

        return os.path.isfile(path)

    def run(self): # main run loop
        """Batch-process all image files in the directory."""
        logger.info("Program started, begin batch processing image files")
        try:
            input_dir = self.config.input_image_dir

            if not os.path.isdir(input_dir):
                logger.error(f"Input directory does not exist: {input_dir}")
                return

            image_files = []
            for filename in os.listdir(input_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.config.supported_extensions:
                    image_files.append(os.path.join(input_dir, filename))

            if not image_files:
                logger.info(f"No supported image files found in the input directory: {input_dir}")
                return

            logger.info(f"Found {len(image_files)} image files, starting batch processing")

            if self.config.enable_csv_output:
                csv_path = os.path.join(self.config.result_data_dir, "resultdata.csv")
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                self.generate_empty_csv()

            for idx, image_path in enumerate(image_files, 1):
                logger.info(f"Processing image {idx}/{len(image_files)}: {image_path}")

                yolo_results = self.perform_yolo_detection(image_path)
                if not yolo_results:
                    logger.warning(f"YOLO detection returned no result: {image_path}")
                    continue

                combined_mask, centroids = self.perform_sam_segmentation(
                    image_path,
                    yolo_results
                )

                if combined_mask is None or centroids is None or len(centroids) == 0:
                    logger.warning(f"SAM segmentation returned no result: {image_path}")
                    continue

                self.post_process_image(
                    image_path,
                    yolo_results,
                    combined_mask,
                    centroids
                )

                logger.info(f"Image {idx}/{len(image_files)} processed")

            logger.info("Batch processing complete")

        except KeyboardInterrupt:
            logger.info("User interrupted, exiting...")
        except Exception as e:
            logger.critical(f"Program error: {str(e)}", exc_info=True)


if __name__ == '__main__': # main entry point
    # Initialize config and predictor
    config = XTCPredictConfig()
    predictor = XTCPredictor(config)

    # Start the main program
    predictor.run()
