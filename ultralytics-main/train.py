from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    #model = YOLO("ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml")
    model = YOLO("ultralytics-main/ultralytics/cfg/models/v8/yolov8FEB.yaml")

    # 开始训练n
    results = model.train(
        data="ultralytics-main/260414-mcf7-mcf10a.yaml",  # 你的数据集配置
        epochs=200,              # 训练轮数
        imgsz=640,              # 图像大小
        batch=4,               # 批次大小
        device=0,              # GPU设备
        workers=4                # 数据加载线程数（Windows上建议使用较小的值）
    )