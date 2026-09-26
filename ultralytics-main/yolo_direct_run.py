import os
import glob
from ultralytics import YOLO
import time
from tqdm import tqdm

# -----------------------------# 可修改参数集中区域# -----------------------------# 默认配置参数# 在这里直接修改参数即可运行，无需命令行输入
DEFAULT_CONFIG = {
    'weights': 'runs/260721前/260107-C2C12-293-YOLOv8n/weights/best.pt',  # 模型权重文件路径
    'source': r"D:\Data\实验结果汇总\识别对比图\C2C12-HEK293\带探针\251223-C2C12-HEK293-86.tif",             # 输入图片路径或文件夹，直接在这里修改
    'output_dir': r"D:\Data\实验结果汇总\识别对比图\C2C12-HEK293\带探针",      # 结果保存根目录
    'conf_thres': 0.2,             # 置信度阈值
    'iou_thres': 0.5,             # IOU阈值
    'img_size': 640,               # 输入图像尺寸
    'device': 'cuda:0',                  # 设备选择，空字符串自动选择，'cpu'或'cuda:0'
    'save_crop': False,            # 是否保存裁剪目标
    'show_labels': True,           # 是否显示标签
    'show_conf': True,             # 是否显示置信度
    'agnostic_nms': False,         # 是否启用类别无关的NMS
    'classes': None,               # 指定检测类别，None表示全部类别，如[0, 1, 2]
    'batch_size': 4,               # 批处理大小
    'result_suffix': '-yolov8nFEB'   # 检测结果图片的自定义后缀，None表示不添加后缀
}

class Args:
    """创建一个模拟命令行参数的类"""
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
    """获取所有需要检测的图片路径
    Args:
        source_path: 输入图片路径或文件夹
    Returns:
        list: 图片路径列表
    """
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 如果是文件
    if os.path.isfile(source_path):
        # 检查文件扩展名是否为支持的图片格式
        ext = os.path.splitext(source_path)[1].lower()
        if ext in image_extensions:
            return [source_path]
        else:
            print(f"警告: 文件 {source_path} 不是支持的图片格式")
            return []
    # 如果是文件夹
    elif os.path.isdir(source_path):
        image_paths = []
        # 遍历文件夹中的所有支持格式的图片
        for ext in image_extensions:
            # 使用glob查找所有匹配扩展名的文件
            pattern = os.path.join(source_path, f'*{ext}')
            image_paths.extend(glob.glob(pattern))
            # 查找子文件夹中的图片
            pattern = os.path.join(source_path, f'**/*{ext}')
            image_paths.extend(glob.glob(pattern, recursive=True))
        # 去重并排序
        image_paths = sorted(list(set(image_paths)))
        print(f"在文件夹 {source_path} 中找到 {len(image_paths)} 张图片")
        return image_paths
    else:
        print(f"错误: 路径 {source_path} 不存在")
        return []

def batch_process_images(model, image_paths, args):
    """批量处理图片
    Args:
        model: YOLO模型对象
        image_paths: 图片路径列表
        args: 参数对象
    """
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print(f"开始处理 {len(image_paths)} 张图片...")
    start_time = time.time()
    
    # 批量处理图片
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="处理进度"):
        # 获取当前批次的图片
        batch_paths = image_paths[i:i+args.batch_size]
        
        # 执行检测
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
                        name='',  # 不创建子目录
                        exist_ok=True,
                        imgsz=args.img_size,
                        device=args.device)
        
        # 记录每张图片的检测结果并处理自定义后缀
        for img_path, result in zip(batch_paths, results):
            img_name = os.path.basename(img_path)
            boxes = result.boxes  # 检测框信息
            
            # 应用自定义后缀
            if args.result_suffix:
                # 获取原始文件名和扩展名
                base_name, ext = os.path.splitext(img_name)
                # 构建新文件名
                new_name = f"{base_name}{args.result_suffix}{ext}"
                # 构建完整的源路径和目标路径
                src_path = os.path.join(args.output, img_name)
                dst_path = os.path.join(args.output, new_name)
                
                # 重命名文件
                if os.path.exists(src_path):
                    try:
                        os.rename(src_path, dst_path)
                        tqdm.write(f"图片 {img_name} 已重命名为 {new_name}")
                    except Exception as e:
                        tqdm.write(f"重命名图片 {img_name} 时出错: {e}")
            
            if len(boxes) > 0:
                tqdm.write(f"图片 {img_name}: 检测到 {len(boxes)} 个目标")
    
    total_time = time.time() - start_time
    print(f"\n检测完成！")
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"平均处理速度: {total_time/len(image_paths):.2f} 秒/张")
    print(f"结果保存在: {os.path.abspath(args.output)}")

def main(config=None):
    """主函数
    Args:
        config: 配置字典，如果为None则使用默认配置
    """
    # 使用传入的配置或默认配置
    if config is None:
        config = DEFAULT_CONFIG
    
    # 创建参数对象
    args = Args(config)
    
    # 验证权重文件是否存在
    if not os.path.isfile(args.weights):
        print(f"错误: 权重文件 {args.weights} 不存在")
        return
    
    # 验证图片源是否存在
    if not (os.path.isfile(args.source) or os.path.isdir(args.source)):
        print(f"错误: 图片源 {args.source} 不存在")
        return
    
    # 加载模型
    print(f"正在加载模型: {args.weights}")
    model = YOLO(args.weights)
    print(f"模型加载完成，使用设备: {model.device}")
    
    # 获取图片路径
    image_paths = get_image_paths(args.source)
    if not image_paths:
        print("没有找到需要检测的图片")
        return
    
    # 批量处理图片
    batch_process_images(model, image_paths, args)

if __name__ == '__main__':
    """
    直接运行说明：
    
    1. 在上方的DEFAULT_CONFIG字典中修改参数
    2. 直接运行此脚本即可开始检测
    
    常用参数说明：
    - weights: 模型权重文件路径
    - source: 输入图片路径或文件夹
    - output_dir: 结果保存目录
    - conf_thres: 置信度阈值，范围0-1
    - batch_size: 批处理大小，可根据电脑性能调整
    - device: 设备选择，'cpu'或'cuda:0'
    - result_suffix: 检测结果图片的自定义后缀，设置为None表示不添加后缀，默认值为'_detected'
    """
    main()