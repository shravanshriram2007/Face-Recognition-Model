from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    model.train(
    data='E:/Coding/Eye/YOLOtest/Dataset/SplitData/data.yaml',
    epochs=30,
    imgsz=640,
    batch=16
)

if __name__ == '__main__':
    main()