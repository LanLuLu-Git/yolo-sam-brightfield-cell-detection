import torch  # 导入 PyTorch 库
import sys  # 导入 sys 模块获取 Python 版本
import platform  # 导入 platform 模块获取操作系统信息
import numpy as np  # 导入 numpy 库
import cv2  # 导入 OpenCV 库
import ultralytics  # 导入 Ultralytics 库（YOLO）

def print_version_info():
    """
    打印系统和库的版本信息
    """
    print("="*50)
    print("系统和库版本信息")
    print("="*50)
    
    # 打印 Python 版本
    print(f"Python 版本: {sys.version}")
    print(f"Python 解释器路径: {sys.executable}")
    
    # 打印操作系统信息
    print(f"操作系统: {platform.system()} {platform.release()} {platform.version()}")
    print(f"机器架构: {platform.machine()}")
    
    # 打印 PyTorch 相关信息
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"PyTorch 安装路径: {torch.__file__}")
    
    # 检查 CUDA 是否可用并打印版本
    if torch.cuda.is_available():
        print(f"CUDA 可用: Yes")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"GPU 设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"CUDA 可用: No")
    
    # 打印其他常用库版本
    print(f"NumPy 版本: {np.__version__}")
    print(f"OpenCV 版本: {cv2.__version__}")
    print(f"Ultralytics 版本: {ultralytics.__version__}")
    
    print("="*50)

if __name__ == "__main__":
    print_version_info()