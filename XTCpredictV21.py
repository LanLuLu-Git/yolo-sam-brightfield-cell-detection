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
# 重要：将 ultralytics-main 路径放在最前面，确保优先使用本地修改的包
sys.path.insert(0, r'D:/Project/YOLOwithSAM/ultralytics-main')
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class XTCPredictConfig:
    """配置类，集中管理所有参数用户"""
    def __init__(self):
        # 路径配置
        self.input_image_dir = "Data/测试对比/C2C12-293/原图"
        self.result_image_dir = 'Data/测试对比/C2C12-293/识别结果/yolov8n-FEB'
        self.result_data_dir = "Data/测试对比/C2C12-293/识别结果/yolov8n-FEB"
        self.yolo_weights_path = 'runs/260721前/260314-YOLOv8FEB-C2C12-293-400/weights/best.pt'
        self.sam_checkpoint = os.path.join(os.path.dirname(__file__), 
                                          'segment-anything-main/models/sam_vit_h_4b8939.pth')
        self.template_path = os.path.join(os.path.dirname(__file__), "tp15.png")
        
        # 模型配置
        self.sam_model_type = 'vit_h'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 处理配置
        self.check_interval = 2  # 检查新文件的间隔(秒)
        self.coordinate_scale = 1e-7 * 2*0.955  # 坐标转换系数
        self.distance_threshold = 5e-5  # 距离阈值
        self.yolo_confidence = 0.38  # YOLO检测置信度阈值(0-1之间)
        self.supported_extensions = ['.tif', '.jpg', '.jpeg', '.png']  # 支持的图片扩展名
        self.enable_template_matching = False  # 是否启用模板匹配检测探针
        self.enable_csv_output = False  # 是否生成CSV结果
        self.save_original_image = False  # 是否保存原图备份


class XTCPredictor:
    """XTC预测器主类,封装所有处理逻辑"""
    def __init__(self, config: XTCPredictConfig):
        self.config = config
        self.sam_predictor = self._init_sam_predictor()
        self.yolo_model = self._init_yolo_model()
     # 预定义颜色映射
        self.fixed_colors = np.array([
            [1.0, 0.0, 0.0, 0.6],  # 红色
            [0.0, 1.0, 0.0, 0.6],  # 绿色
            [0.0, 0.0, 1.0, 0.6],  # 蓝色
            [1.0, 1.0, 0.0, 0.6],  # 黄色
            [1.0, 0.0, 1.0, 0.6],  # 紫色
            [0.0, 1.0, 1.0, 0.6]   # 青色
            ])       

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
        """执行SAM图像分割"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return None, None

        try:
            self.sam_predictor.set_image(image)
            
            
            # 创建颜色映射字典
            class_colors = {
                class_id: self.fixed_colors[class_id % len(self.fixed_colors)] 
                for class_id in yolo_results[0].names
            }

            # 初始化结果
            combined_mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
            centroids = []
            
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
                    
                    # 计算物体中心点坐标
                    mask_indices = np.argwhere(best_mask)
                    centroid = np.mean(mask_indices, axis=0).astype(int)[::-1]  # 转换为(x,y)格式
                    centroids.append(centroid)
                    
                    # 获取当前目标的类别ID和颜色
                    class_id = int(cls[i])
                    color = class_colors[class_id]
                    
                    # 构建掩码的RGBA表示
                    mask_rgba = np.zeros((best_mask.shape[0], best_mask.shape[1], 4), dtype=np.float32)
                    mask_rgba[best_mask] = color
                    
                    # 合并到总mask
                    combined_mask = np.where(mask_rgba[..., 3:4] > 0, mask_rgba, combined_mask)
            
            logger.info("SAM分割完成")
            return combined_mask, centroids
            
        except Exception as e:
            logger.error(f"SAM分割失败: {str(e)}", exc_info=True)
            return None, None

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

    def post_process_image(self, image_path: str, yolo_results, combined_mask, centroids) -> bool: # 后处理图像并保存结果
        """后处理图像并保存结果"""
        # 检查结果是否有效
        if yolo_results is None or combined_mask is None or centroids is None or len(centroids) == 0:
            logger.error("后处理失败：输入参数不完整")
            return False

        try:
            # 常量定义
            CIRCLE_COLOR = (0, 0, 255)
            TEXT_COLOR = (0, 0, 0)
            FONT = cv2.FONT_HERSHEY_SIMPLEX
            FONT_SCALE = 1.5
            THICKNESS = 3
            
            # 创建日期文件夹
            today_date = datetime.now().strftime("%Y-%m-%d")
            today_folder = os.path.join(self.config.result_image_dir, today_date)
            os.makedirs(today_folder, exist_ok=True)
            
            # 读取原始图像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return False
                
            # 图像叠加处理
            image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
            result = (image_rgba * 0.7 + combined_mask * 0.3) * 255
            result = result.astype(np.uint8)
            # 将RGBA转换回BGR格式以便OpenCV正确保存
            result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
            
            # 获取YOLO检测框
            xyxy = []
            for r in yolo_results:
                xyxy.extend(r.boxes.xyxy.cpu().numpy())
            
            # 查找Sphere_Probe坐标
            sphere_probe_coords = None
            if self.config.enable_template_matching:
                sphere_probe_coords = self._find_sphere_probe(image)
            
            # 保存CSV数据
            if self.config.enable_csv_output:
                self._save_centroid_data(sphere_probe_coords, yolo_results, centroids, xyxy)
            
            # 标记中心点和相对位置
            self._draw_annotations(result, sphere_probe_coords, centroids, yolo_results)
            
            # 保存结果图像
            self._save_processed_images(result, image, image_path, today_folder)
            
            logger.info("图像后处理完成")
            return True
            
        except Exception as e:
            logger.error(f"后处理失败: {str(e)}", exc_info=True)
            return False

    def _save_centroid_data(self, sphere_coords, yolo_results, centroids, xyxy, write_header=False): # 保存中心点坐标数据到CSV
        """保存中心点坐标数据到CSV"""
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
                    logger.warning("未检测到Sphere_Probe，跳过CSV写入")
                    
            logger.info(f"坐标数据已保存到: {csv_path}")
            
        except PermissionError as e:
            logger.error(f"无法写入CSV文件: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"保存CSV数据失败: {str(e)}", exc_info=True)
            raise

    def _draw_annotations(self, result, sphere_coords, centroids, yolo_results): # 在图像上绘制标注
        """在图像上绘制标注"""
        # 标记Sphere_Probe
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
        
        # 标记细胞中心点和信息
        for i, centroid in enumerate(centroids): 
            class_id = int(yolo_results[0].boxes.cls.cpu().numpy()[i])
            class_name = yolo_results[0].names[class_id]
            
            # 绘制中心点
            cv2.circle(result, tuple(centroid), 5, (0, 0, 255), -1)
            
            """
            # 绘制标签 
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
        
        # 添加时间戳
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

    def _save_processed_images(self, result, original_image, image_path, save_folder): # 保存处理后的图像和原图备份
        """保存处理后的图像和原图备份"""
        # 生成时间戳和文件名
        original_filename = os.path.basename(image_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name, file_ext = os.path.splitext(original_filename)
        
        # 保存原图备份
        if self.config.save_original_image:
            original_save_path = os.path.join(save_folder, f"original_{timestamp}_{file_name}{file_ext}")
            cv2.imwrite(original_save_path, original_image)
            logger.info(f"原图备份: {original_save_path}")
        
        # 保存处理后的图像
        processed_save_path = os.path.join(save_folder, f"processed_{timestamp}_{file_name}{file_ext}")
        cv2.imwrite(processed_save_path, result)
        
        # 保存到默认结果路径
        default_result_path = os.path.join(self.config.result_image_dir, "resultimage.jpg")
        cv2.imwrite(default_result_path, result)
        
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
        """批量处理目录中的所有图像文件"""
        logger.info("程序启动，开始批量处理图像文件")
        try:
            input_dir = self.config.input_image_dir
            
            if not os.path.isdir(input_dir):
                logger.error(f"输入目录不存在: {input_dir}")
                return
            
            image_files = []
            for filename in os.listdir(input_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.config.supported_extensions:
                    image_files.append(os.path.join(input_dir, filename))
            
            if not image_files:
                logger.info(f"输入目录中没有找到支持的图像文件: {input_dir}")
                return
            
            logger.info(f"共找到 {len(image_files)} 个图像文件，开始批量处理")
            
            if self.config.enable_csv_output:
                csv_path = os.path.join(self.config.result_data_dir, "resultdata.csv")
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                self.generate_empty_csv()
            
            for idx, image_path in enumerate(image_files, 1):
                logger.info(f"处理第 {idx}/{len(image_files)} 张图像: {image_path}")
                
                yolo_results = self.perform_yolo_detection(image_path)
                if not yolo_results:
                    logger.warning(f"YOLO检测无结果: {image_path}")
                    continue
                
                combined_mask, centroids = self.perform_sam_segmentation(
                    image_path, 
                    yolo_results
                )
                
                if combined_mask is None or centroids is None or len(centroids) == 0:
                    logger.warning(f"SAM分割无结果: {image_path}")
                    continue
                
                self.post_process_image(
                    image_path, 
                    yolo_results, 
                    combined_mask, 
                    centroids
                )
                
                logger.info(f"第 {idx}/{len(image_files)} 张图像处理完成")
            
            logger.info("批量处理完成")
            
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