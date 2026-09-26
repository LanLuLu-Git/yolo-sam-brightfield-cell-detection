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
# 重要：将 ultralytics-main 路径放在最前面，确保优先使用本地修改的包
sys.path.insert(0, r'D:/Project/YOLOwithSAM/ultralytics-main')
# 如果 site-packages 中有 ultralytics，需要移除它以使用本地版本
site_packages_path = r'D:/APP/Miniconda3/envs/yolosam/lib/site-packages'
if site_packages_path in sys.path:
    sys.path.remove(site_packages_path)
# 重新添加回去，放在最后
sys.path.append(site_packages_path)

from ultralytics import YOLO
from ultralytics.change_model import FEB

"""2025年12月12日 2.0改进：
结构化与封装
    使用XTCPredictConfig类集中管理所有配置参数
    使用XTCPredictor类封装所有业务逻辑,提高内聚性
    拆分大型函数为多个小型方法，每个方法只负责单一职责
日志系统
    替换 print 为 logging 模块，支持同时输出到控制台和文件
    分级记录日志(info, warning, error, critical)
    记录详细的错误堆栈信息，便于调试
代码可读性
    完善文档字符串，说明类和方法的功能、参数和返回值
    变量和方法命名更规范，遵循 PEP8 风格
    使用常量代替硬编码数值，提高可维护性
错误处理
    增加更全面的异常捕获和处理
    为可能出错的操作添加详细日志
    分离错误处理与业务逻辑
可扩展性
    配置集中管理，便于修改参数
    模块化设计，便于添加新功能
    清晰的方法边界，便于单元测试
修改与改进
    增加图片异常处理
    修改颜色映射不匹配问题
251219:修改异常处理,增加异常tif图片被删除
       修改距离阈值5e-5为1e-4
"""
# 配置日志系统
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
    """配置类，集中管理所有参数用户"""
    def __init__(self):
        # 路径配置
        self.input_image_path = 'C:/Users/14366/Desktop/xtc/1.tif'
        self.result_image_dir = 'D:/Project/YOLOwithSAM/result/jpkrun'
        self.result_data_dir = 'C:/Users/14366/Desktop/xtc'
        self.yolo_weights_path = 'runs/260721前/260605-yolov8FEB-three-70/weights/best.pt'
        self.sam_checkpoint = os.path.join(os.path.dirname(__file__), 
                                          'segment-anything-main/models/sam_vit_h_4b8939.pth')
        self.template_path = os.path.join(os.path.dirname(__file__), "tp260719.jpg")
        
        # 模型配置
        self.sam_model_type = 'vit_h'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 处理配置
        self.check_interval = 5  # 检查新文件的间隔(秒)
        self.coordinate_scale = 1e-7 * 2*0.955  # 坐标转换系数
        self.distance_threshold = 5e-5  # 距离阈值
        self.yolo_confidence = 0.1  # YOLO检测置信度阈值(0-1之间)

        # --- 微球排除配置 ---
        self.MICROSPHERE_EXCLUSION_RADIUS = 10  # 微球中心排除半径（像素），范围内的细胞识别结果将被舍弃
        self.DUPLICATE_DISTANCE_THRESHOLD = 10  # 重复识别距离阈值（像素），在此距离内的细胞视为重复，保留轮廓面积大的

        # --- 类别配置 - 可自定义类别名称和颜色 ---
        # 格式: {class_id: {'name': '类别名称', 'color': [R, G, B, A]}}
        self.class_config = {
            0: {'name': 'MCF10A', 'color': [0, 255, 0, 255]},    # 绿色
        }

        # --- 显示配置 - 控制识别结果的展示效果 ---
        self.show_centroid = True            # 是否显示细胞中心十字标记（跟随类别颜色）
        self.show_cell_name = True         # 是否显示细胞类别名称
        self.show_dashed_contour = True      # 是否显示细胞虚线轮廓（跟随类别颜色）

        # --- 细胞中心十字标记样式 ---
        self.centroid_cross_size = 10            # 十字臂半长（像素）
        self.centroid_cross_thickness = 3        # 十字线宽（像素）

        # --- 微球标记配置 ---
        self.show_sphere_center = True       # 是否显示微球中心橙色十字标记（+）
        self.show_sphere_label = True       # 是否显示微球文字标签 "Sphere_Probe"
        self.show_sphere_box = True          # 是否显示微球中心正方形框

        # --- 微球十字标记样式 ---
        self.sphere_center_color = (0, 165, 255)    # 十字标记颜色 (OpenCV BGR: 橙色)
        self.sphere_center_size = 10                # 十字臂半长（像素）
        self.sphere_center_thickness = 3            # 十字线宽（像素）

        # --- 虚线轮廓样式 ---
        self.dash_length = 20                # 虚线段的像素长度
        self.gap_length = 10                 # 虚线间隔的像素长度
        self.outline_thickness = 6           # 轮廓线宽度（像素）

        # --- 微球正方形框样式 ---
        self.sphere_box_size = 524           # 正方形框边长（像素），框中心为球心
        self.sphere_box_color = (0, 165, 255)       # 橙色 (OpenCV BGR)
        self.sphere_box_thickness = 4        # 框线宽度（像素）

        # --- 比例尺配置 ---
        self.show_scale_bar = True               # 是否显示比例尺
        self.scale_bar_length_um = 50.0          # 比例尺代表的实际长度（微米）
        self.scale_bar_position = 'bottom-right'  # 位置: 'bottom-right', 'bottom-left', 'top-right', 'top-left'
        self.scale_bar_color = [255, 255, 255]   # 比例尺颜色 (BGR)，默认白色
        self.scale_bar_thickness = 4             # 比例尺主线条粗细（像素）
        self.scale_bar_margin = 50               # 比例尺距图像边缘的距离（像素）
        self.scale_bar_font_scale = 0.7          # 比例尺文字大小


class XTCPredictor:
    """XTC预测器主类,封装所有处理逻辑"""
    def __init__(self, config: XTCPredictConfig):
        self.config = config
        self.sam_predictor = self._init_sam_predictor()
        self.yolo_model = self._init_yolo_model()
        self._populate_class_config()  # 从YOLO模型自动填充类别配置

    def _populate_class_config(self): # 从YOLO模型自动填充class_config
        """从YOLO模型的类别名称自动填充class_config，分配颜色"""
        yolo_names = self.yolo_model.names
        # 颜色调色板（RGBA格式，按class_id循环使用）
        color_palette = [
            [0, 255, 0, 255],    # 绿色
            [255, 0, 0, 255],    # 红色
            [0, 0, 255, 255],    # 蓝色
            [255, 255, 0, 255],  # 黄色
            [255, 0, 255, 255],  # 紫色
            [0, 255, 255, 255],  # 青色
        ]
        for class_id, name in yolo_names.items():
            # 只添加尚未手动配置的类别（保留用户在XTCPredictConfig中的手动定义）
            if class_id not in self.config.class_config:
                color = color_palette[class_id % len(color_palette)]
                self.config.class_config[class_id] = {'name': name, 'color': color}
        logger.info(f"类别配置已填充: { {k: v['name'] for k, v in self.config.class_config.items()} }")

    def _init_sam_predictor(self) -> SamPredictor: # 初始化SAM预测器
        """初始化SAM预测器"""
        try:
            sam = sam_model_registry[self.config.sam_model_type](
                checkpoint=self.config.sam_checkpoint
            )
            sam.to(device=self.config.device)
            logger.info(f"SAM模型初始化完成，使用设备: {self.config.device}")
            return SamPredictor(sam)
        except Exception as e:
            logger.error(f"SAM模型初始化失败: {str(e)}", exc_info=True)
            raise

    def _init_yolo_model(self) -> YOLO: # 初始化YOLO模型
        """初始化YOLO模型"""
        try:
            model = YOLO(self.config.yolo_weights_path)
            logger.info("YOLO模型初始化完成")
            return model
        except Exception as e:
            logger.error(f"YOLO模型初始化失败: {str(e)}", exc_info=True)
            raise

    def perform_yolo_detection(self, image_path: str): # 执行YOLO目标检测
        """执行YOLO目标检测"""
        if not os.path.exists(image_path):
            logger.error(f"图像文件不存在: {image_path}")
            return None

        try:
            results = self.yolo_model(image_path, device=self.config.device, conf=self.config.yolo_confidence)
            logger.info(f"YOLO检测完成，处理图像: {image_path}, 置信度阈值: {self.config.yolo_confidence}")
            return results
        except Exception as e:
            logger.error(f"YOLO检测失败: {str(e)}", exc_info=True)
            return None

    def perform_sam_segmentation(self, image_path: str, yolo_results) -> tuple: # 执行SAM图像分割
        """执行SAM图像分割

        Returns:
            tuple: (combined_mask, centroids, class_ids, mask_contours)
        """
        # 读取图像（支持中文路径）
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return None, None, None, None

        try:
            self.sam_predictor.set_image(image)

            # 初始化结果
            combined_mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
            centroids = []
            class_ids = []
            mask_contours = []

            # 处理每个检测结果
            for r in yolo_results:
                boxes = r.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy()

                for i, box in enumerate(xyxy):
                    # 准备输入框
                    input_box = np.array([box])

                    # 使用边界框进行预测
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box,
                        multimask_output=True,
                    )

                    # 找出最高置信度的掩码
                    best_mask = masks[np.argmax(scores)]

                    # 计算物体中心点坐标（增加空mask检查）
                    mask_indices = np.argwhere(best_mask)
                    if len(mask_indices) > 0:
                        centroid = np.mean(mask_indices, axis=0).astype(int)[::-1]  # 转换为(x,y)格式
                        centroids.append(centroid)
                        class_id = int(cls[i])
                        class_ids.append(class_id)

                        # 从SAM最佳掩码中提取外轮廓
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

                        # 获取当前目标的颜色（从class_config）
                        class_info = self.config.class_config.get(
                            class_id, {'color': [128, 128, 128, 255]}
                        )
                        color = class_info['color']

                        # 构建掩码的RGBA表示
                        mask_rgba = np.zeros((best_mask.shape[0], best_mask.shape[1], 4), dtype=np.float32)
                        mask_rgba[best_mask] = color

                        # 合并到总mask
                        combined_mask = np.where(mask_rgba[..., 3:4] > 0, mask_rgba, combined_mask)

            logger.info("SAM分割完成")
            return combined_mask, centroids, class_ids, mask_contours

        except Exception as e:
            logger.error(f"SAM分割失败: {str(e)}", exc_info=True)
            return None, None, None, None

    def _find_sphere_probe(self, image) -> tuple: # 使用模板匹配查找Sphere_Probe坐标
        """使用模板匹配查找Sphere_Probe坐标"""
        if not os.path.exists(self.config.template_path):
            logger.warning(f"模板文件不存在: {self.config.template_path}")
            return None

        try:
            template = cv2.imread(self.config.template_path)
            if template is None:
                logger.warning(f"无法读取模板文件: {self.config.template_path}")
                return None
                
            # 执行模板匹配
            res = cv2.matchTemplate(image[:,:,0], template[:,:,0], cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            
            # 计算中心点
            center_x = max_loc[0] + template.shape[1] // 2
            center_y = max_loc[1] + template.shape[0] // 2
            
            logger.info("Sphere_Probe检测完成")
            return (center_x, center_y)
            
        except Exception as e:
            logger.error(f"Sphere_Probe检测失败: {str(e)}", exc_info=True)
            return None

    def _draw_dashed_contour(self, image, contour, bgr_color):
        """沿闭合轮廓绘制单条虚线

        OpenCV 无原生虚线 polyline 支持，这里沿轮廓以 1px 步长采样，
        按 dash_length/gap_length 交替绘制和跳过 cv2.line 线段。
        """
        dash_len = self.config.dash_length
        gap_len = self.config.gap_length
        thickness = self.config.outline_thickness

        # Step 1: 简化轮廓以减少顶点数
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.002 * perimeter  # 简化容差为周长的 0.2%
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = approx.reshape(-1, 2).astype(np.float64)
        n = len(pts)

        if n < 2:
            return

        # Step 2: 向量化预计算各段长度和方向向量
        next_idx = (np.arange(n) + 1) % n
        seg_vecs = pts[next_idx] - pts
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        # 避免退化为零的段
        seg_dirs = np.zeros_like(seg_vecs)
        valid = seg_lens > 1e-10
        seg_dirs[valid] = seg_vecs[valid] / seg_lens[valid, np.newaxis]
        cum_lens = np.cumsum(seg_lens)
        total_len = cum_lens[-1]

        if total_len < dash_len:
            # 轮廓太短，直接绘制实线
            cv2.polylines(image, [approx], True, bgr_color, thickness)
            return

        # Step 3: 沿轮廓以 1px 步长采样，交替 dash/gap 相位
        step = 1.0
        cur_dist = 0.0
        in_dash = True          # 当前处于虚线阶段
        phase_dist = 0.0        # 当前相位已行进距离
        prev_pt = None          # 上一个采样点（相位切换时置 None 以断开线段）

        while cur_dist <= total_len:
            # 在轮廓上定位 cur_dist 对应的点
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

            # 检查是否完成当前 dash 或 gap 相位
            threshold = dash_len if in_dash else gap_len
            if phase_dist >= threshold:
                in_dash = not in_dash
                phase_dist = 0.0
                prev_pt = None  # 相位切换时断开线段

    def _draw_dashed_contours(self, image, mask_contours, class_ids):
        """遍历所有轮廓，使用类别对应颜色绘制虚线轮廓"""
        if mask_contours is None:
            return

        for contour, class_id in zip(mask_contours, class_ids):
            if contour is None or len(contour) < 4:
                continue

            class_info = self.config.class_config.get(
                class_id, {'name': f'Class_{class_id}', 'color': [128, 128, 128, 255]}
            )
            rgba = class_info['color']
            # class_config 存储的是 [R, G, B, A]；OpenCV 使用 BGR
            bgr_color = (int(rgba[2]), int(rgba[1]), int(rgba[0]))

            self._draw_dashed_contour(image, contour, bgr_color)

    def _draw_scale_bar(self, image):
        """在图像上绘制比例尺

        根据 coordinate_scale 将实际微米长度换算为像素长度，
        在图像指定位置绘制水平比例尺线条、端点标记和文字标签。

        换算公式: pixel_length = scale_bar_length_um / (coordinate_scale * 1e6)
                  其中 coordinate_scale * 1e6 ≈ 0.191 µm/pixel
        """
        if not self.config.show_scale_bar:
            return

        # 每像素对应微米数
        um_per_pixel = self.config.coordinate_scale * 1e6  # ≈ 0.191 µm/pixel

        if um_per_pixel <= 0:
            logger.warning("坐标转换系数无效，跳过比例尺绘制")
            return

        # 比例尺像素长度
        pixel_length = int(self.config.scale_bar_length_um / um_per_pixel)

        if pixel_length < 10:
            logger.warning(f"比例尺像素长度过短 ({pixel_length} px)，跳过绘制")
            return

        h, w = image.shape[:2]
        margin = self.config.scale_bar_margin
        thickness = self.config.scale_bar_thickness
        color = tuple(self.config.scale_bar_color)

        # 计算位置
        pos = self.config.scale_bar_position
        bar_height = max(thickness, 6)  # 端点竖线半高

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
            logger.warning(f"未知的比例尺位置: {pos}，使用默认位置 'bottom-right'")
            x1 = w - margin - pixel_length
            x2 = w - margin
            y_base = h - margin - bar_height

        # 确保比例尺在图像范围内
        x1 = max(0, x1)
        x2 = min(w - 1, x2)
        pixel_length = x2 - x1

        if pixel_length < 10:
            logger.warning("比例尺超出图像范围，跳过绘制")
            return

        # 绘制主水平线
        cv2.line(image, (x1, y_base), (x2, y_base), color, thickness)

        # 绘制两端竖线（刻度标记）
        tick_thickness = max(thickness // 2 + 1, 2)
        cv2.line(image, (x1, y_base - bar_height), (x1, y_base + bar_height), color, tick_thickness)
        cv2.line(image, (x2, y_base - bar_height), (x2, y_base + bar_height), color, tick_thickness)

        # 绘制文字标签（位于比例尺上方中央）
        if self.config.scale_bar_length_um >= 1.0:
            label = f"{self.config.scale_bar_length_um:.0f} um"
        else:
            label = f"{self.config.scale_bar_length_um:.1f} um"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.scale_bar_font_scale
        text_thickness = max(thickness // 2, 1)

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        text_x = x1 + (pixel_length - text_w) // 2
        text_y = y_base - bar_height - baseline  # 文字底部距比例尺上方

        # 确保文字不超出图像顶部
        if text_y - text_h < 0:
            # 如果上方空间不足，将文字放在比例尺下方
            text_y = y_base + bar_height + text_h + baseline

        cv2.putText(image, label, (text_x, text_y), font, font_scale, color, text_thickness)

    def post_process_image(self, image_path: str, yolo_results, combined_mask, centroids, class_ids, mask_contours) -> bool: # 后处理图像并保存结果
        """后处理图像并保存结果"""
        # 检查结果是否有效
        if yolo_results is None or combined_mask is None or centroids is None or class_ids is None or mask_contours is None or len(centroids) == 0:
            logger.error("后处理失败：输入参数不完整")
            return False

        try:
            # 创建日期文件夹
            today_date = datetime.now().strftime("%Y-%m-%d")
            today_folder = os.path.join(self.config.result_image_dir, today_date)
            os.makedirs(today_folder, exist_ok=True)

            # 读取原始图像（支持中文路径）
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return False

            # 直接基于原图绘制（不再使用半透明遮罩叠加）
            result = image.copy()

            # 获取YOLO检测框
            xyxy = []
            for r in yolo_results:
                xyxy.extend(r.boxes.xyxy.cpu().numpy())

            # 查找Sphere_Probe坐标
            sphere_probe_coords = self._find_sphere_probe(image)

            # --- 微球中心排除：过滤掉微球中心附近的细胞识别结果 ---
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
                    logger.info(f"微球中心排除: 过滤掉 {excluded_count} 个近距离细胞, 保留 {len(centroids)} 个")

                if len(centroids) == 0:
                    logger.warning("微球排除后无剩余细胞，跳过绘制和保存")
                    return True

            # --- 重复识别过滤：相近的识别结果保留轮廓面积最大的 ---
            if len(centroids) > 1:
                # 计算每个轮廓的面积（用于去重时的优先级判断）
                areas = []
                for contour in mask_contours:
                    if contour is not None and len(contour) >= 3:
                        areas.append(cv2.contourArea(contour))
                    else:
                        areas.append(0)

                # 按面积降序排列索引（面积大的优先保留）
                sorted_indices = sorted(range(len(centroids)), key=lambda i: areas[i], reverse=True)
                kept = []
                removed = set()
                threshold = self.config.DUPLICATE_DISTANCE_THRESHOLD

                for i in sorted_indices:
                    if i in removed:
                        continue
                    kept.append(i)
                    # 将与当前保留细胞距离过近的其他细胞标记为重复
                    cx, cy = centroids[i]
                    for j in sorted_indices:
                        if j == i or j in removed or j in kept:
                            continue
                        dx = centroids[j][0] - cx
                        dy = centroids[j][1] - cy
                        if np.sqrt(dx*dx + dy*dy) < threshold:
                            removed.add(j)

                if removed:
                    kept.sort()  # 恢复原始顺序
                    dup_count = len(removed)
                    centroids = [centroids[i] for i in kept]
                    class_ids = [class_ids[i] for i in kept]
                    mask_contours = [mask_contours[i] for i in kept]
                    xyxy = [xyxy[i] for i in kept]
                    logger.info(f"重复识别过滤: 移除 {dup_count} 个重复结果, 保留 {len(centroids)} 个")

            # 绘制 SAM 掩码虚线轮廓
            if self.config.show_dashed_contour:
                self._draw_dashed_contours(result, mask_contours, class_ids)

            # 保存CSV数据
            self._save_centroid_data(sphere_probe_coords, class_ids, centroids, xyxy)

            # 标记中心点和相对位置
            self._draw_annotations(result, sphere_probe_coords, centroids, class_ids)

            # 绘制比例尺
            self._draw_scale_bar(result)

            # 保存结果图像
            self._save_processed_images(result, image, image_path, today_folder)

            logger.info("图像后处理完成")
            return True

        except Exception as e:
            logger.error(f"后处理失败: {str(e)}", exc_info=True)
            return False

    def _save_centroid_data(self, sphere_coords, class_ids, centroids, xyxy): # 保存中心点坐标数据到CSV
        """保存中心点坐标数据到CSV"""
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
                    logger.warning("未检测到Sphere_Probe，生成空CSV文件")

            logger.info(f"坐标数据已保存到: {csv_path}")

        except PermissionError as e:
            logger.error(f"无法写入CSV文件: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"保存CSV数据失败: {str(e)}", exc_info=True)
            raise

    def _draw_annotations(self, result, sphere_coords, centroids, class_ids): # 在图像上绘制标注
        """在图像上绘制标注（所有元素可通过 XTCPredictConfig 开关控制）"""

        # --- 微球中心橙色十字标记（+）---
        if self.config.show_sphere_center and sphere_coords is not None:
            cx, cy = sphere_coords
            s = self.config.sphere_center_size
            t = self.config.sphere_center_thickness
            mc = self.config.sphere_center_color
            # 十字由一横一竖两条线段组成, 交汇于微球中心
            cv2.line(result, (cx - s, cy), (cx + s, cy), mc, t)  # 水平线
            cv2.line(result, (cx, cy - s), (cx, cy + s), mc, t)  # 垂直线

        # --- 微球文字标签 ---
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

        # --- 微球中心正方形框 ---
        if self.config.show_sphere_box and sphere_coords is not None:
            half = self.config.sphere_box_size // 2
            top_left = (sphere_coords[0] - half, sphere_coords[1] - half)
            bottom_right = (sphere_coords[0] + half, sphere_coords[1] + half)
            cv2.rectangle(
                result, top_left, bottom_right,
                self.config.sphere_box_color, self.config.sphere_box_thickness
            )

        # --- 细胞中心点和名称标签 ---
        for centroid, class_id in zip(centroids, class_ids):
            class_info = self.config.class_config.get(class_id, {'name': f'Class_{class_id}'})
            class_name = class_info['name']

            if self.config.show_centroid:
                # 使用类别颜色绘制十字标记（与虚线轮廓颜色一致）
                rgba = class_info.get('color', [128, 128, 128, 255])
                bgr_color = (int(rgba[2]), int(rgba[1]), int(rgba[0]))
                cx, cy = centroid
                s = self.config.centroid_cross_size
                t = self.config.centroid_cross_thickness
                cv2.line(result, (cx - s, cy), (cx + s, cy), bgr_color, t)  # 水平线
                cv2.line(result, (cx, cy - s), (cx, cy + s), bgr_color, t)  # 垂直线

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

    def _save_processed_images(self, result, original_image, image_path, save_folder): # 保存处理后的图像
        """保存处理后的图像（支持中文路径）"""
        # 生成时间戳和文件名
        original_filename = os.path.basename(image_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name, file_ext = os.path.splitext(original_filename)

        # 保存处理后的图像（使用 cv2.imencode 支持中文路径）
        processed_save_path = os.path.join(save_folder, f"processed_{timestamp}_{file_name}{file_ext}")
        success, buffer = cv2.imencode(file_ext, result)
        if success:
            buffer.tofile(processed_save_path)

        logger.info(f"处理后图像: {processed_save_path}")

    def safe_delete_file(self, file_path: str) -> bool: # 安全删除文件
        """安全删除文件"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"文件已删除: {file_path}")
                return True
            else:
                logger.info(f"文件不存在，无需删除: {file_path}")
                return False
        except Exception as e:
            logger.error(f"删除文件失败: {str(e)}", exc_info=True)
            return False

    def convert_tif_to_jpg(self, tif_path: str) -> str: # 将tif图片转换为jpg格式并保存
        """将tif图片转换为jpg格式并保存
        
        Args:
            tif_path: tif图片的路径
            
        Returns:
            str: 转换后的jpg图片路径，如果转换失败返回None
        """
        try:
            if not tif_path.lower().endswith('.tif'):
                logger.warning(f"不是tif格式的图片: {tif_path}")
                return tif_path  # 如果不是tif格式，直接返回原路径
                
            # 生成jpg保存路径
            jpg_path = os.path.splitext(tif_path)[0] + '.jpg'
            
            # 读取tif图片
            img = cv2.imread(tif_path)
            if img is None:
                logger.error(f"无法读取tif图片: {tif_path}")
                return None
                
            # 保存为jpg格式
            success = cv2.imwrite(jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not success:
                logger.error(f"无法保存jpg图片: {jpg_path}")
                return None
                
            logger.info(f"tif图片转换成功: {tif_path} -> {jpg_path}")
            return jpg_path
            
        except Exception as e:
            logger.error(f"tif转jpg失败: {str(e)}", exc_info=True)
            return None
    
    def generate_empty_csv(self) -> bool: # 生成空的CSV文件
        """生成空的CSV文件
        
        Returns:
            bool: 是否成功生成空CSV文件
        """
        try:
            os.makedirs(self.config.result_data_dir, exist_ok=True)
            csv_path = os.path.join(self.config.result_data_dir, "resultdata.csv")
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Cell_Name', 'X_Distance', 'Y_Distance'])
            
            logger.info(f"空CSV文件已生成: {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成空CSV文件失败: {str(e)}", exc_info=True)
            return False
    
    def check_file_exists(self, path: str) -> bool: # 检查文件是否存在
        """检查文件是否存在"""
        if not os.path.exists(path):
            logger.warning(f"路径不存在: {path}")
            return False
        
        return os.path.isfile(path)

    def run(self): # 主运行循环
        """主运行循环"""
        logger.info("程序启动，开始监控图像文件")
        try:
            while True:
                if self.check_file_exists(self.config.input_image_path):
                    logger.info(f"检测到新图像: {self.config.input_image_path}")
                    
                    # 执行tif转jpg格式转换
                    processed_image_path = self.convert_tif_to_jpg(self.config.input_image_path)
                    
                    if processed_image_path is None:
                        logger.error("图片转换失败，生成空CSV文件并进入下一个循环")
                        #删除原图（根据需要启用）
                        self.safe_delete_file(self.config.input_image_path)
                        # 生成空CSV文件
                        self.generate_empty_csv()
                        time.sleep(self.config.check_interval)
                        continue
                    
                    # 执行YOLO检测
                    yolo_results = self.perform_yolo_detection(processed_image_path)
                    if not yolo_results:
                        time.sleep(self.config.check_interval)
                        continue
                    
                    # 执行SAM分割
                    combined_mask, centroids, class_ids, mask_contours = self.perform_sam_segmentation(
                        processed_image_path,
                        yolo_results
                    )
                    # 检查结果是否有效
                    if combined_mask is None or centroids is None or class_ids is None or mask_contours is None or len(centroids) == 0:
                        time.sleep(self.config.check_interval)
                        continue

                    # 后处理
                    self.post_process_image(
                        processed_image_path,
                        yolo_results,
                        combined_mask,
                        centroids,
                        class_ids,
                        mask_contours
                    )
                    #删除原图和转换后的图片（根据需要启用）
                    self.safe_delete_file(self.config.input_image_path)
                    if processed_image_path != self.config.input_image_path:
                        self.safe_delete_file(processed_image_path)
                    
                    logger.info("当前图像处理完成")
                    time.sleep(3)  # 短暂等待，避免重复处理
                else:
                    logger.info(f"未检测到图像，{self.config.check_interval}秒后重试")
                    time.sleep(self.config.check_interval)
                    
        except KeyboardInterrupt:
            logger.info("用户中断程序，正在退出...")
        except Exception as e:
            logger.critical(f"程序运行出错: {str(e)}", exc_info=True)


if __name__ == '__main__': # 主程序入口
    # 初始化配置和预测器
    config = XTCPredictConfig()
    predictor = XTCPredictor(config)
    
    # 启动主程序
    predictor.run() 