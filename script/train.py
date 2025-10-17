import os
from pathlib import Path
from ultralytics import YOLO
import torch

def main():
    # Environment setup
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["YOLO_LOGGING_LEVEL"] = "WARNING"
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Paths
    TEST_IMG = Path(r"datasets\images\test")
    yaml_path = Path(r"datasets\dataset.yaml")

    # Start training (Single phase)
    print("Start training ...")

    model = YOLO("model/yolov9s.pt")
    results = model.train(
        data=yaml_path.as_posix(),
        epochs=120,
        batch=8,
        imgsz=640,
        device="0" if torch.cuda.is_available() else "cpu",
        optimizer="AdamW",
        lr0=0.0006,
        lrf=0.02,
        weight_decay=0.0005,
        warmup_epochs=5,
        patience=25,
        cls=1.5,
        box=0.5,
        dfl=1.5,
        mosaic=0.25,
        mixup=0.1,
        copy_paste=0.08,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.6,
        fliplr=0.4,
        flipud=0.15,
        val=True,
        save_period=5,
        exist_ok=True,
        seed=42,
        pretrained=True,
        resume=False,
        verbose=True,
        save=True,
    )
    print("Training completed!")

    # Final validation
    val_results = model.val(
        data=yaml_path.as_posix(),
        imgsz=640,
        device="0" if torch.cuda.is_available() else "cpu"
    )

    print("Final Validation Results:")
    print(f"mAP50:     {val_results.box.map50:.4f}")
    print(f"mAP50-95:  {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall:    {val_results.box.mr:.4f}")

    #Test set prediction
    print("Running prediction on test set ...")
    test_results = model.predict(
        source=TEST_IMG.as_posix(),
        imgsz=640,
        device="0" if torch.cuda.is_available() else "cpu",
        conf=0.25,
        save=True,
        save_txt=True,
        exist_ok=True
    )

    print("Test prediction completed!")

    metrics = model.val(
        data=yaml_path.as_posix(),
        imgsz=640,
        device="0" if torch.cuda.is_available() else "cpu")

    print("Performance Summary:")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP@0.5:   {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

    model.save("model/model.pt")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()