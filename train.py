import os
import sys
from pathlib import Path

# Enable expandable segments to avoid CUDA memory fragmentation on Windows RTX GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_yolo():
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] PyTorch or Ultralytics is not installed.")
        print("Please run setup_env.bat first!")
        sys.exit(1)

    print("===================================================")
    print("        YOLOv8n License Plate Trainer            ")
    print("===================================================")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available : {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Target GPU     : {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("[WARNING] CUDA is NOT available. Training on CPU will be extremely slow!")
        confirm = input("Do you still want to continue on CPU? (y/N): ")
        if confirm.lower() != 'y':
            sys.exit(0)

    # Locate dataset yaml
    base_dir = Path(__file__).parent.resolve()
    data_yaml = base_dir / "License Plate Detection" / "data.yaml"
    
    if not data_yaml.exists():
        print(f"[ERROR] Could not find dataset config at: {data_yaml}")
        sys.exit(1)
        
    print(f"Dataset Config : {data_yaml}")
    print("---------------------------------------------------")

    # Load YOLOv8n pre-trained model
    model = YOLO("yolov8n.pt")

    # Train parameters tuned for 4GB VRAM (RTX 3050)
    print("Starting training with RTX 3050 optimizations (Batch=8, ImgSz=640, AMP=True)...")
    results = model.train(
        data=str(data_yaml),
        epochs=100,            # Max epochs (with early stopping)
        patience=20,           # Early stopping if no improvement for 20 epochs
        imgsz=640,             # Standard image resolution
        batch=8,               # Optimal batch size for 4GB VRAM
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,             # Avoid RAM overhead on Windows
        amp=True,              # Automatic Mixed Precision (FP16)
        cache=False,           # Disable caching to conserve RAM
        optimizer="AdamW",
        lr0=0.001,
        save=True,
        project=str(base_dir / "runs"),
        name="detect_plate",
        exist_ok=True
    )

    print("===================================================")
    print(" Training Complete!")
    print(f" Best model weights saved to: {base_dir / 'runs' / 'detect_plate' / 'weights' / 'best.pt'}")
    print("===================================================")

if __name__ == "__main__":
    train_yolo()
