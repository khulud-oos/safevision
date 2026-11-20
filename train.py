from ultralytics import YOLO

def main():
    # نبدأ من أفضل نموذج مدرب سابقاً (Transfer Learning)
    model = YOLO("runs/detect/runs/safevision_train_v2/weights/best.pt")
    
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=800,
        batch=8,
        device="cpu",
        project="runs",
        name="safevision_train_v4",
        patience=20,
        save=True,
        lr0=0.001,
        cos_lr=True
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()