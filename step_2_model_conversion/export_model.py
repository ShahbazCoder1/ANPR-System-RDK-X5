import sys
from pathlib import Path

def export_to_onnx():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Please run setup_env.bat first!")
        sys.exit(1)

    base_dir = Path(__file__).resolve().parent
    
    # Check for weights across possible locations
    candidate_paths = [
        base_dir.parent / "step_1_model_training" / "runs" / "detect_plate" / "weights" / "best.pt",
        base_dir.parent / "runs" / "detect_plate" / "weights" / "best.pt",
        base_dir / "runs" / "detect_plate" / "weights" / "best.pt",
        base_dir / "best.pt",
    ]
    
    weights_path = None
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        weights_path = Path(sys.argv[1])
    else:
        for p in candidate_paths:
            if p.exists():
                weights_path = p
                break

    if not weights_path:
        print(f"[ERROR] Trained model file (best.pt) not found.")
        print("Please train the model first by running: python train.py in step_1_model_training/")
        print("Or pass custom path: python export_model.py <path_to_best.pt>")
        sys.exit(1)

    print(f"Loading trained PyTorch model: {weights_path}")
    model = YOLO(str(weights_path))

    print("Exporting model to ONNX format (opset=11, simplified, 640x640)...")
    exported_onnx = model.export(
        format="onnx",
        imgsz=640,
        opset=11,          # Strictly compatible with Horizon Open Explorer toolchain
        simplify=True,
        dynamic=False      # Fixed batch size for optimal BPU quantization
    )

    out_onnx_dir = base_dir / "onnx_models"
    out_onnx_dir.mkdir(parents=True, exist_ok=True)
    target_onnx = out_onnx_dir / "best.onnx"
    
    import shutil
    shutil.copy2(exported_onnx, str(target_onnx))

    print("===================================================")
    print(" Export Complete!")
    print(f" ONNX Model saved to: {target_onnx}")
    print(" This file is ready for Horizon OpenExplorer Docker conversion (.bin) for RDK X5!")
    print("===================================================")

if __name__ == "__main__":
    export_to_onnx()
