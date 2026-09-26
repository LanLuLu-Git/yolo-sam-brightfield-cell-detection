import os
import cv2
import numpy as np
import csv
import logging
import time
from datetime import datetime
import torch
from segment_anything import sam_model_registry, SamPredictor

import sys
# Important: put the ultralytics-main path first so the locally modified package is used
sys.path.insert(0, r'D:/Project/YOLOwithSAM/ultralytics-main')
# If ultralytics exists in site-packages, remove it so the local version is used
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



class XTCPredictConfig:
    """Configuration class that centralizes all parameters."""
    def __init__(self):
        # Path configuration
        self.input_image_path = 'C:/Users/14366/Desktop/xtc/1.tif'
        self.result_image_dir = 'D:/Project/YOLOwithSAM/result/jpkrun'
        self.result_data_dir = 'C:/Users/14366/Desktop/xtc'
        self.yolo_weights_path = 'runs/260721前/260605-yolov8FEB-three-70/weights/best.pt'
        self.sam_checkpoint = os.path.join(os.path.dirname(__file__),
                                          'segment-anything-main/models/sam_vit_h_4b8939.pth')
        self.template_path = os.path.join(os.path.dirname(__file__), "tp260719.jpg")

        # Model configuration
        self.sam_model_type = 'vit_h'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Processing configuration
        self.check_interval = 5  # interval (seconds) to check for new files
        self.coordinate_scale = 1e-7 * 2*0.955  # coordinate conversion factor
        self.distance_threshold = 5e-5  # distance threshold
        self.yolo_confidence = 0.1  # YOLO detection confidence threshold (0-1)

        # --- Microsphere exclusion configuration ---
        self.MICROSPHERE_EXCLUSION_RADIUS = 10  # microsphere center exclusion radius (pixels); cell detections within this radius are discarded
        self.DUPLICATE_DISTANCE_THRESHOLD = 10  # duplicate detection distance threshold (pixels); detections within this distance are considered duplicates, keep the one with the larger contour area

        # --- Class configuration - customize class names and colors ---
        # format: {class_id: {'name': 'class name', 'color': [R, G, B, A]}}
        self.class_config = {
            0: {'name': 'MCF10A', 'color': [0, 255, 0, 255]},    # green
        }

        # --- Display configuration - control how results are rendered ---
        self.show_centroid = True            # whether to show the cell centroid cross marker (following the class color)
        self.show_cell_name = True         # whether to show the cell class name
        self.show_dashed_contour = True      # whether to show the dashed cell contour (following the class color)

        # --- Cell centroid cross marker style ---
        self.centroid_cross_size = 10            # half-length of the cross arm (pixels)
        self.centroid_cross_thickness = 3        # cross line width (pixels)

        # --- Microsphere marker configuration ---
        self.show_sphere_center = True       # whether to show the orange cross marker (+) at the microsphere center
        self.show_sphere_label = True       # whether to show the microsphere text label "Sphere_Probe"
        self.show_sphere_box = True          # whether to show the square box at the microsphere center

        # --- Microsphere cross marker style ---
        self.sphere_center_color = (0, 165, 255)    # cross marker color (OpenCV BGR: orange)
        self.sphere_center_size = 10                # half-length of the cross arm (pixels)
        self.sphere_center_thickness = 3            # cross line width (pixels)

        # --- Dashed contour style ---
        self.dash_length = 20                # pixel length of a dash segment
        self.gap_length = 10                 # pixel length of the gap between dashes
        self.outline_thickness = 6           # contour line width (pixels)

        # --- Microsphere square box style ---
        self.sphere_box_size = 524           # side length of the square box (pixels); the box center is the sphere center
        self.sphere_box_color = (0, 165, 255)       # orange (OpenCV BGR)
        self.sphere_box_thickness = 4        # box line width (pixels)

        # --- Scale bar configuration ---
        self.show_scale_bar = True               # whether to show the scale bar
        self.scale_bar_length_um = 50.0          # real length represented by the scale bar (micrometers)
        self.scale_bar_position = 'bottom-right'  # position: 'bottom-right', 'bottom-left', 'top-right', 'top-left'
        self.scale_bar_color = [255, 255, 255]   # scale bar color (BGR), default white
        self.scale_bar_thickness = 4             # main line thickness of the scale bar (pixels)
        self.scale_bar_margin = 50               # distance of the scale bar from the image edge (pixels)
        self.scale_bar_font_scale = 0.7          # scale bar text size


class XTCPredictor:
    """Main XTC predictor class, encapsulating all processing logic."""
    def __init__(self, config: XTCPredictConfig):
        self.config = config
        self.sam_predictor = self._init_sam_predictor()
        self.yolo_model = self._init_yolo_model()
        self._populate_class_config()  # automatically populate class_config from the YOLO model

    def _populate_class_config(self): # automatically populate class_config from the YOLO model
        """Automatically populate class_config from the YOLO model class names and assign colors."""
        yolo_names = self.yolo_model.names
        # Color palette (RGBA format, reused cyclically by class_id)
        color_palette = [
            [0, 255, 0, 255],    # green
            [255, 0, 0, 255],    # red
            [0, 0, 255, 255],    # blue
            [255, 255, 0, 255],  # yellow
            [255, 0, 255, 255],  # purple
            [0, 255, 255, 255],  # cyan
        ]
        for class_id, name in yolo_names.items():
            # Only add classes not yet configured manually (preserve the user's manual definitions in XTCPredictConfig)
            if class_id not in self.config.class_config:
                color = color_palette[class_id % len(color_palette)]
                self.config.class_config[class_id] = {'name': name, 'color': color}
        logger.info(f"Class configuration populated: { {k: v['name'] for k, v in self.config.class_config.items()} }")

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
        """Run SAM image segmentation.

        Returns:
            tuple: (combined_mask, centroids, class_ids, mask_contours)
        """
        # Read the image (supports non-ASCII paths)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.error(f"Cannot read image: {image_path}")
            return None, None, None, None

        try:
            self.sam_predictor.set_image(image)

            # Initialize results
            combined_mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
            centroids = []
            class_ids = []
            mask_contours = []

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

                    # Compute the object centroid coordinates (add empty-mask check)
                    mask_indices = np.argwhere(best_mask)
                    if len(mask_indices) > 0:
                        centroid = np.mean(mask_indices, axis=0).astype(int)[::-1]  # convert to (x, y) format
                        centroids.append(centroid)
                        class_id = int(cls[i])
                        class_ids.append(class_id)

                        # Extract the outer contour from the best SAM mask
                        contours, _ = cv2.findContours(
                            best_mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        if contours and len(contours) > 0:
                            mask_contour = max(contours, key=cv2.contourArea)
                        else:
                            mask_contour = None
                        mask_contours.append(mask_contour)

                        # Get the color of the current object (from class_config)
                        class_info = self.config.class_config.get(
                            class_id, {'color': [128, 128, 128, 255]}
                        )
                        color = class_info['color']

                        # Build the RGBA representation of the mask
                        mask_rgba = np.zeros((best_mask.shape[0], best_mask.shape[1], 4), dtype=np.float32)
                        mask_rgba[best_mask] = color

                        # Merge into the combined mask
                        combined_mask = np.where(mask_rgba[..., 3:4] > 0, mask_rgba, combined_mask)

            logger.info("SAM segmentation complete")
            return combined_mask, centroids, class_ids, mask_contours

        except Exception as e:
            logger.error(f"SAM segmentation failed: {str(e)}", exc_info=True)
            return None, None, None, None

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

    def _draw_dashed_contour(self, image, contour, bgr_color):
        """Draw a single dashed line along a closed contour.

        OpenCV has no native dashed-polyline support; here we sample along the
        contour at a 1px step and alternately draw/skip cv2.line segments
        according to dash_length/gap_length.
        """
        dash_len = self.config.dash_length
        gap_len = self.config.gap_length
        thickness = self.config.outline_thickness

        # Step 1: simplify the contour to reduce the number of vertices
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.002 * perimeter  # simplification tolerance is 0.2% of the perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = approx.reshape(-1, 2).astype(np.float64)
        n = len(pts)

        if n < 2:
            return

        # Step 2: vectorized pre-computation of segment lengths and direction vectors
        next_idx = (np.arange(n) + 1) % n
        seg_vecs = pts[next_idx] - pts
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        # Avoid degenerate zero-length segments
        seg_dirs = np.zeros_like(seg_vecs)
        valid = seg_lens > 1e-10
        seg_dirs[valid] = seg_vecs[valid] / seg_lens[valid, np.newaxis]
        cum_lens = np.cumsum(seg_lens)
        total_len = cum_lens[-1]

        if total_len < dash_len:
            # Contour too short, draw a solid line directly
            cv2.polylines(image, [approx], True, bgr_color, thickness)
            return

        # Step 3: sample along the contour at a 1px step, alternating dash/gap phases
        step = 1.0
        cur_dist = 0.0
        in_dash = True          # currently in the dash phase
        phase_dist = 0.0        # distance already traveled in the current phase
        prev_pt = None          # previous sample point (set to None at phase switches to break the line)

        while cur_dist <= total_len:
            # Locate the point on the contour corresponding to cur_dist
            seg_idx = np.searchsorted(cum_lens, cur_dist, side='right')
            if seg_idx >= n:
                seg_idx = n - 1
            prev_cum = 0.0 if seg_idx == 0 else cum_lens[seg_idx - 1]
            local_dist = cur_dist - prev_cum
            curr_pt = tuple(
                (pts[seg_idx] + seg_dirs[seg_idx] * local_dist).astype(np.int32)
            )

            if in_dash and prev_pt is not None:
                cv2.line(image, prev_pt, curr_pt, bgr_color, thickness)

            prev_pt = curr_pt
            cur_dist += step
            phase_dist += step

            # Check whether the current dash or gap phase is complete
            threshold = dash_len if in_dash else gap_len
            if phase_dist >= threshold:
                in_dash = not in_dash
                phase_dist = 0.0
                prev_pt = None  # break the line at phase switches

    def _draw_dashed_contours(self, image, mask_contours, class_ids):
        """Iterate over all contours and draw dashed contours using the class color."""
        if mask_contours is None:
            return

        for contour, class_id in zip(mask_contours, class_ids):
            if contour is None or len(contour) < 4:
                continue

            class_info = self.config.class_config.get(
                class_id, {'name': f'Class_{class_id}', 'color': [128, 128, 128, 255]}
            )
            rgba = class_info['color']
            # class_config stores [R, G, B, A]; OpenCV uses BGR
            bgr_color = (int(rgba[2]), int(rgba[1]), int(rgba[0]))

            self._draw_dashed_contour(image, contour, bgr_color)

    def _draw_scale_bar(self, image):
        """Draw a scale bar on the image.

        Convert the real micrometer length to a pixel length via coordinate_scale,
        and draw the horizontal scale bar line, endpoint ticks, and text label at
        the specified position.

        Conversion formula: pixel_length = scale_bar_length_um / (coordinate_scale * 1e6),
        where coordinate_scale * 1e6 ≈ 0.191 µm/pixel.
        """
        if not self.config.show_scale_bar:
            return

        # Micrometers per pixel
        um_per_pixel = self.config.coordinate_scale * 1e6  # ≈ 0.191 µm/pixel

        if um_per_pixel <= 0:
            logger.warning("Invalid coordinate conversion factor, skipping scale bar drawing")
            return

        # Scale bar pixel length
        pixel_length = int(self.config.scale_bar_length_um / um_per_pixel)

        if pixel_length < 10:
            logger.warning(f"Scale bar pixel length too short ({pixel_length} px), skipping drawing")
            return

        h, w = image.shape[:2]
        margin = self.config.scale_bar_margin
        thickness = self.config.scale_bar_thickness
        color = tuple(self.config.scale_bar_color)

        # Compute the position
        pos = self.config.scale_bar_position
        bar_height = max(thickness, 6)  # half-height of the endpoint ticks

        if pos == 'bottom-right':
            x1 = w - margin - pixel_length
            x2 = w - margin
            y_base = h - margin - bar_height
        elif pos == 'bottom-left':
            x1 = margin
            x2 = margin + pixel_length
            y_base = h - margin - bar_height
        elif pos == 'top-right':
            x1 = w - margin - pixel_length
            x2 = w - margin
            y_base = margin + bar_height
        elif pos == 'top-left':
            x1 = margin
            x2 = margin + pixel_length
            y_base = margin + bar_height
        else:
            logger.warning(f"Unknown scale bar position: {pos}, using default position 'bottom-right'")
            x1 = w - margin - pixel_length
            x2 = w - margin
            y_base = h - margin - bar_height

        # Ensure the scale bar is within the image bounds
        x1 = max(0, x1)
        x2 = min(w - 1, x2)
        pixel_length = x2 - x1

        if pixel_length < 10:
            logger.warning("Scale bar outside the image bounds, skipping drawing")
            return

        # Draw the main horizontal line
        cv2.line(image, (x1, y_base), (x2, y_base), color, thickness)

        # Draw the vertical ticks at both ends
        tick_thickness = max(thickness // 2 + 1, 2)
        cv2.line(image, (x1, y_base - bar_height), (x1, y_base + bar_height), color, tick_thickness)
        cv2.line(image, (x2, y_base - bar_height), (x2, y_base + bar_height), color, tick_thickness)

        # Draw the text label (centered above the scale bar)
        if self.config.scale_bar_length_um >= 1.0:
            label = f"{self.config.scale_bar_length_um:.0f} um"
        else:
            label = f"{self.config.scale_bar_length_um:.1f} um"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.scale_bar_font_scale
        text_thickness = max(thickness // 2, 1)

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        text_x = x1 + (pixel_length - text_w) // 2
        text_y = y_base - bar_height - baseline  # distance of the text bottom above the scale bar

        # Ensure the text does not exceed the top of the image
        if text_y - text_h < 0:
            # If there is not enough space above, place the text below the scale bar
            text_y = y_base + bar_height + text_h + baseline

        cv2.putText(image, label, (text_x, text_y), font, font_scale, color, text_thickness)

    def post_process_image(self, image_path: str, yolo_results, combined_mask, centroids, class_ids, mask_contours) -> bool: # post-process the image and save results
        """Post-process the image and save results."""
        # Check whether the results are valid
        if yolo_results is None or combined_mask is None or centroids is None or class_ids is None or mask_contours is None or len(centroids) == 0:
            logger.error("Post-processing failed: incomplete input parameters")
            return False

        try:
            # Create a date-named folder
            today_date = datetime.now().strftime("%Y-%m-%d")
            today_folder = os.path.join(self.config.result_image_dir, today_date)
            os.makedirs(today_folder, exist_ok=True)

            # Read the original image (supports non-ASCII paths)
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                logger.error(f"Cannot read image: {image_path}")
                return False

            # Draw directly on the original image (no semi-transparent mask overlay)
            result = image.copy()

            # Get the YOLO detection boxes
            xyxy = []
            for r in yolo_results:
                xyxy.extend(r.boxes.xyxy.cpu().numpy())

            # Find the Sphere_Probe coordinates
            sphere_probe_coords = self._find_sphere_probe(image)

            # --- Microsphere center exclusion: filter out cell detections near the microsphere center ---
            if sphere_probe_coords is not None:
                keep_indices = []
                excluded_count = 0
                for i, centroid in enumerate(centroids):
                    dist = np.sqrt(
                        (centroid[0] - sphere_probe_coords[0])**2 +
                        (centroid[1] - sphere_probe_coords[1])**2
                    )
                    if dist > self.config.MICROSPHERE_EXCLUSION_RADIUS:
                        keep_indices.append(i)
                    else:
                        excluded_count += 1

                if excluded_count > 0:
                    centroids = [centroids[i] for i in keep_indices]
                    class_ids = [class_ids[i] for i in keep_indices]
                    mask_contours = [mask_contours[i] for i in keep_indices]
                    xyxy = [xyxy[i] for i in keep_indices]
                    logger.info(f"Microsphere center exclusion: removed {excluded_count} nearby cells, kept {len(centroids)}")

                if len(centroids) == 0:
                    logger.warning("No cells remain after microsphere exclusion, skipping drawing and saving")
                    return True

            # --- Duplicate detection filtering: keep the one with the largest contour area among nearby detections ---
            if len(centroids) > 1:
                # Compute the area of each contour (used for priority when deduplicating)
                areas = []
                for contour in mask_contours:
                    if contour is not None and len(contour) >= 3:
                        areas.append(cv2.contourArea(contour))
                    else:
                        areas.append(0)

                # Sort indices by area descending (larger area kept first)
                sorted_indices = sorted(range(len(centroids)), key=lambda i: areas[i], reverse=True)
                kept = []
                removed = set()
                threshold = self.config.DUPLICATE_DISTANCE_THRESHOLD

                for i in sorted_indices:
                    if i in removed:
                        continue
                    kept.append(i)
                    # Mark other cells too close to the current kept cell as duplicates
                    cx, cy = centroids[i]
                    for j in sorted_indices:
                        if j == i or j in removed or j in kept:
                            continue
                        dx = centroids[j][0] - cx
                        dy = centroids[j][1] - cy
                        if np.sqrt(dx*dx + dy*dy) < threshold:
                            removed.add(j)

                if removed:
                    kept.sort()  # restore the original order
                    dup_count = len(removed)
                    centroids = [centroids[i] for i in kept]
                    class_ids = [class_ids[i] for i in kept]
                    mask_contours = [mask_contours[i] for i in kept]
                    xyxy = [xyxy[i] for i in kept]
                    logger.info(f"Duplicate filtering: removed {dup_count} duplicates, kept {len(centroids)}")

            # Draw the dashed contours of the SAM masks
            if self.config.show_dashed_contour:
                self._draw_dashed_contours(result, mask_contours, class_ids)

            # Save CSV data
            self._save_centroid_data(sphere_probe_coords, class_ids, centroids, xyxy)

            # Annotate centroids and relative positions
            self._draw_annotations(result, sphere_probe_coords, centroids, class_ids)

            # Draw the scale bar
            self._draw_scale_bar(result)

            # Save the result image
            self._save_processed_images(result, image, image_path, today_folder)

            logger.info("Image post-processing complete")
            return True

        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}", exc_info=True)
            return False

    def _save_centroid_data(self, sphere_coords, class_ids, centroids, xyxy): # save centroid coordinate data to CSV
        """Save centroid coordinate data to CSV."""
        try:
            os.makedirs(self.config.result_data_dir, exist_ok=True)
            csv_path = os.path.join(self.config.result_data_dir, "resultdata.csv")

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Cell_Name', 'X_Distance', 'Y_Distance'])

                if sphere_coords is not None:
                    for centroid, class_id in zip(centroids, class_ids):
                        class_info = self.config.class_config.get(class_id, {'name': f'Class_{class_id}'})
                        class_name = class_info['name']

                        if class_name not in ['Sphere_Probe', 'Dead']:
                            x_dist = -(centroid[0] - sphere_coords[0]) * self.config.coordinate_scale
                            y_dist = -(centroid[1] - sphere_coords[1]) * self.config.coordinate_scale

                            if abs(x_dist) < self.config.distance_threshold and abs(y_dist) < self.config.distance_threshold:
                                writer.writerow([class_name, y_dist, x_dist])
                else:
                    logger.warning("Sphere_Probe not detected, generating an empty CSV file")

            logger.info(f"Coordinate data saved to: {csv_path}")

        except PermissionError as e:
            logger.error(f"Cannot write CSV file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to save CSV data: {str(e)}", exc_info=True)
            raise

    def _draw_annotations(self, result, sphere_coords, centroids, class_ids): # draw annotations on the image
        """Draw annotations on the image (all elements can be toggled via XTCPredictConfig)."""

        # --- Orange cross marker (+) at the microsphere center ---
        if self.config.show_sphere_center and sphere_coords is not None:
            cx, cy = sphere_coords
            s = self.config.sphere_center_size
            t = self.config.sphere_center_thickness
            mc = self.config.sphere_center_color
            # The cross consists of a horizontal and a vertical line segment intersecting at the microsphere center
            cv2.line(result, (cx - s, cy), (cx + s, cy), mc, t)  # horizontal line
            cv2.line(result, (cx, cy - s), (cx, cy + s), mc, t)  # vertical line

        # --- Microsphere text label ---
        if self.config.show_sphere_label and sphere_coords is not None:
            cv2.putText(
                result,
                "Sphere_Probe",
                (sphere_coords[0] + 10, sphere_coords[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

        # --- Square box at the microsphere center ---
        if self.config.show_sphere_box and sphere_coords is not None:
            half = self.config.sphere_box_size // 2
            top_left = (sphere_coords[0] - half, sphere_coords[1] - half)
            bottom_right = (sphere_coords[0] + half, sphere_coords[1] + half)
            cv2.rectangle(
                result, top_left, bottom_right,
                self.config.sphere_box_color, self.config.sphere_box_thickness
            )

        # --- Cell centroid and name label ---
        for centroid, class_id in zip(centroids, class_ids):
            class_info = self.config.class_config.get(class_id, {'name': f'Class_{class_id}'})
            class_name = class_info['name']

            if self.config.show_centroid:
                # Draw the cross marker using the class color (consistent with the dashed contour color)
                rgba = class_info.get('color', [128, 128, 128, 255])
                bgr_color = (int(rgba[2]), int(rgba[1]), int(rgba[0]))
                cx, cy = centroid
                s = self.config.centroid_cross_size
                t = self.config.centroid_cross_thickness
                cv2.line(result, (cx - s, cy), (cx + s, cy), bgr_color, t)  # horizontal line
                cv2.line(result, (cx, cy - s), (cx, cy + s), bgr_color, t)  # vertical line

            if self.config.show_cell_name:
                cv2.putText(
                    result,
                    class_name,
                    (centroid[0] + 10, centroid[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )

    def _save_processed_images(self, result, original_image, image_path, save_folder): # save the processed image
        """Save the processed image (supports non-ASCII paths)."""
        # Generate the timestamp and filename
        original_filename = os.path.basename(image_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name, file_ext = os.path.splitext(original_filename)

        # Save the processed image (use cv2.imencode to support non-ASCII paths)
        processed_save_path = os.path.join(save_folder, f"processed_{timestamp}_{file_name}{file_ext}")
        success, buffer = cv2.imencode(file_ext, result)
        if success:
            buffer.tofile(processed_save_path)

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
        """Main run loop."""
        logger.info("Program started, begin monitoring image files")
        try:
            while True:
                if self.check_file_exists(self.config.input_image_path):
                    logger.info(f"New image detected: {self.config.input_image_path}")

                    # Convert tif to jpg
                    processed_image_path = self.convert_tif_to_jpg(self.config.input_image_path)

                    if processed_image_path is None:
                        logger.error("Image conversion failed, generate an empty CSV file and continue to the next loop")
                        # Delete the original image (enable as needed)
                        self.safe_delete_file(self.config.input_image_path)
                        # Generate an empty CSV file
                        self.generate_empty_csv()
                        time.sleep(self.config.check_interval)
                        continue

                    # Run YOLO detection
                    yolo_results = self.perform_yolo_detection(processed_image_path)
                    if not yolo_results:
                        time.sleep(self.config.check_interval)
                        continue

                    # Run SAM segmentation
                    combined_mask, centroids, class_ids, mask_contours = self.perform_sam_segmentation(
                        processed_image_path,
                        yolo_results
                    )
                    # Check whether the results are valid
                    if combined_mask is None or centroids is None or class_ids is None or mask_contours is None or len(centroids) == 0:
                        time.sleep(self.config.check_interval)
                        continue

                    # Post-process
                    self.post_process_image(
                        processed_image_path,
                        yolo_results,
                        combined_mask,
                        centroids,
                        class_ids,
                        mask_contours
                    )
                    # Delete the original and converted images (enable as needed)
                    self.safe_delete_file(self.config.input_image_path)
                    if processed_image_path != self.config.input_image_path:
                        self.safe_delete_file(processed_image_path)

                    logger.info("Current image processed")
                    time.sleep(3)  # short wait to avoid repeated processing
                else:
                    logger.info(f"No image detected, retrying in {self.config.check_interval} seconds")
                    time.sleep(self.config.check_interval)

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
