"""
YOLO模型测试脚本 - 验证模型泛化能力

使用说明:
python ultralytics-main/test.py --conf 0.76 --weights runs/detect/train5/weights/best.pt --data ultralytics-main/hgc27-mgc803.yaml --img 640

参数说明:
--weights: 训练好的模型权重路径
--conf: 置信度阈值
--data: 数据集配置文件路径
--img: 输入图像尺寸
"""

import argparse
from ultralytics import YOLO
import os

def main():
    # 参数配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径(必须提供)')
    parser.add_argument('--conf', type=float, default=0.76, help='置信度阈值')
    parser.add_argument('--data', type=str, default='ultralytics-main/hgc27-mgc803.yaml', help='数据集配置文件路径')
    parser.add_argument('--img', type=int, default=640, help='输入图像尺寸')
    
    args = parser.parse_args()
    
    # 加载模型
    model = YOLO(args.weights)
    
    # 测试模型
    results = model.val(
        data=args.data,
        conf=args.conf,
        imgsz=args.img,
        split='test',  # 使用测试集验证泛化能力
        verbose=True   # 输出详细评估指标
    )
    
    # 打印关键评估指标
    print(f"\n模型泛化能力评估结果:")
    print(f"mAP@0.5: {float(results.box.map):.4f}")
    print(f"mAP@0.5:0.95: {float(results.box.map):.4f}")
    #print(f"Precision: {float(results.box.p):.4f}")
    #print(f"Recall: {float(results.box.r):.4f}")

if __name__ == '__main__':
    main()
