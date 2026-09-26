from ultralytics import YOLO

if __name__ == '__main__':
    # Load the model
    #model = YOLO("ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml")
    model = YOLO("ultralytics-main/ultralytics/cfg/models/v8/yolov8FEB.yaml")

    # Start training
    results = model.train(
        data="ultralytics-main/260414-mcf7-mcf10a.yaml",  # your dataset config
        epochs=200,              # number of training epochs
        imgsz=640,              # image size
        batch=4,               # batch size
        device=0,              # GPU device
        workers=4                # data loading worker count (a smaller value is recommended on Windows)
    )