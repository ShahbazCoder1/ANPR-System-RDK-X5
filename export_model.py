import sys
from pathlib import Path

def export_to_onnx():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Please run setup_env.bat first!")
        sys.exit(1)

    base_dir = Path(__file__).parent.resolve()
    weights_path = base_dir / "runs" / "detect_plate" / "weights" / "best.pt"

    if not weights_path.exists():
        print(f"[ERROR] Trained model file not found at: {weights_path}")
        print("Please train the model first by running: python train.py")
        sys.exit(1)

    print(f"Loading trained PyTorch model: {weights_path}")
    model = YOLO(str(weights_path))

    print("Exporting model to ONNX format (opset=11, simplified)...")
    onnx_path = model.export(
        format="onnx",
        opset=11,          # Compatible with Horizon Open Explorer toolchain
        simplify=True,
        dynamic=False      # Fixed batch size for optimal BPU quantization later
    )

    print("===================================================")
    print(" Export Complete!")
    print(f" ONNX Model saved to: {onnx_path}")
    print(" This file is ready for Horizon OE toolchain conversion (.bin) for RDK X5!")
    print("===================================================")

if __name__ == "__main__":
    export_to_onnx()
