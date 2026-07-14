from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Train on your dataset
    model.train(
        data="data.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        workers=2,
        device=0,
        name="ppe_detector"
    )

if __name__ == "__main__":
    main()