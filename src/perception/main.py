import argparse
from pathlib import Path

from .config import PerceptionConfig
from .video_processor import VideoProcessor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process driving MP4 files with Person 1 perception."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("result/outputs"))
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--segmentation-model")
    parser.add_argument("--lane-model")
    parser.add_argument("--traffic-sign-model")
    parser.add_argument("--traffic-light-model")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    if not args.input_dir.exists():
        print(f"ERROR: input directory does not exist: {args.input_dir}")
        return 1
    videos = sorted(args.input_dir.glob("*.mp4"))
    if not videos:
        print(f"No MP4 files found in {args.input_dir}")
        return 0
    config = PerceptionConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        segmentation_model_path=args.segmentation_model,
        lane_model_path=args.lane_model,
        traffic_sign_model_path=args.traffic_sign_model,
        traffic_light_model_path=args.traffic_light_model,
        confidence=args.confidence,
        iou=args.iou,
        image_size=args.imgsz,
        device=args.device,
        save_visualizations=args.visualize,
    )
    try:
        processor = VideoProcessor(config)
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    for video in videos:
        print(f"Processing: {video.name}")
        try:
            output = processor.process_video(video)
            if output:
                print(f"Output: {output}")
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"ERROR: failed to process {video.name}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
