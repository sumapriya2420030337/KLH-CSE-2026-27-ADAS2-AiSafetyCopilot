from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_settings
from .perception.config import PerceptionConfig
from .perception.detector import YoloDetector
from .pipeline import run_video, write_outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Person 1 perception and Person 2 tracking/prediction."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("result/outputs"))
    parser.add_argument(
        "--prediction-mode", choices=("baseline", "lstm"), default="baseline"
    )
    parser.add_argument("--frame-skip", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--progress-interval", type=int)
    args = parser.parse_args()
    settings = load_settings(args.config)
    settings.output_path = args.output_dir
    if args.frame_skip is not None:
        settings.video.frame_skip = max(1, args.frame_skip)
    if args.imgsz is not None:
        settings.video.imgsz = max(32, args.imgsz)
    if args.max_frames is not None:
        settings.video.max_frames = max(1, args.max_frames)
    if args.progress_interval is not None:
        settings.video.progress_interval = max(1, args.progress_interval)
    settings.prediction.mode = args.prediction_mode
    if args.prediction_mode == "baseline":
        settings.prediction.model_path = None
    detector_config = PerceptionConfig(
        model_path=args.model, image_size=settings.video.imgsz
    )
    detector = YoloDetector(
        detector_config.model_path,
        detector_config.confidence,
        detector_config.iou,
        detector_config.image_size,
        detector_config.resolved_device(),
        detector_config.object_classes,
    )
    outputs = run_video(
        args.video,
        lambda frame, frame_id, timestamp: detector.track_structured(
            frame, frame_id, timestamp
        ),
        settings,
        args.output_dir
        / f"{args.video.stem}_tracked{'_lstm' if args.prediction_mode == 'lstm' else ''}.mp4",
    )
    suffix = "_lstm" if args.prediction_mode == "lstm" else ""
    write_outputs(outputs, args.output_dir / f"{args.video.stem}_tracking{suffix}.json")
    print(f"Processed {len(outputs)} frames; prediction mode: {args.prediction_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
