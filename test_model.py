import argparse
import sys
import cv2
from pathlib import Path

def test_model():
    parser = argparse.ArgumentParser(description="Test trained YOLOv8 License Plate Detection Model")
    parser.add_argument("--mode", type=str, choices=["dataset", "image", "video"], default="dataset",
                        help="Test mode: 'dataset' (validation metrics), 'image' (single image), 'video' (video file)")
    parser.add_argument("--source", type=str, default="", help="Path to image or video file when mode is 'image' or 'video'")
    parser.add_argument("--weights", type=str, default="", help="Custom weights path (defaults to runs/detect_plate/weights/best.pt)")
    parser.add_argument("--conf", type=float, default=0.55, help="Confidence threshold for detection (default: 0.55)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Please run setup_env.bat first!")
        sys.exit(1)

    base_dir = Path(__file__).parent.resolve()
    
    # Path to trained model
    weights_path = Path(args.weights) if args.weights else base_dir / "runs" / "detect_plate" / "weights" / "best.pt"
    
    if not weights_path.exists():
        print(f"[ERROR] Trained model file not found at: {weights_path}")
        print("Please train the model first by running python train.py")
        sys.exit(1)

    print(f"Loading trained model: {weights_path}")
    model = YOLO(str(weights_path))

    if args.mode == "dataset":
        data_yaml = base_dir / "License Plate Detection" / "data.yaml"
        print(f"Evaluating model on test dataset split: {data_yaml}...")
        metrics = model.val(data=str(data_yaml), split="test")
        print("\n================ Validation Metrics ================")
        print(f"mAP50    : {metrics.box.map50:.4f}")
        print(f"mAP50-95 : {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall   : {metrics.box.mr:.4f}")
        print("====================================================")

    elif args.mode == "image":
        if not args.source or not Path(args.source).exists():
            print("[ERROR] Please specify a valid --source path to an image file.")
            sys.exit(1)
        
        print(f"Running detection on image: {args.source} with confidence threshold: {args.conf}")
        results = model.predict(source=args.source, conf=args.conf, save=True)
        print(f"Results saved to: {results[0].save_dir}")

    elif args.mode == "video":
        if not args.source or not Path(args.source).exists():
            print("[ERROR] Please specify a valid --source path to a video file.")
            sys.exit(1)

        print(f"Processing video: {args.source} with confidence threshold: {args.conf}...")
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video file: {args.source}")
            sys.exit(1)

        out_dir = base_dir / "runs" / "video_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"detected_{Path(args.source).name}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=args.conf, verbose=False)
            annotated_frame = results[0].plot()

            out.write(annotated_frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        out.release()
        print(f"\n[SUCCESS] Processed video saved to: {out_path}")

if __name__ == "__main__":
    test_model()
