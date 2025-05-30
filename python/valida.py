from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\xjoao\PycharmProjects\TCC\python\runs\pose\train13\weights\best.pt")
    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    main()