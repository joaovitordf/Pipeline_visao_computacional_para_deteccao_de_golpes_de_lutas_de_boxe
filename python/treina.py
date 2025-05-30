from ultralytics import YOLO

def main():
    MODEL_PRETRAINED = "yolov8l-pose.pt"
    DATA_CONFIG = "data.yaml"

    model = YOLO(MODEL_PRETRAINED, task="pose")

    results = model.train(
        data=DATA_CONFIG,
        imgsz=640,
        epochs=110,
        batch=16,
        device=0
    )

    print("Treino finalizado.", results)

if __name__ == "__main__":
    main()



