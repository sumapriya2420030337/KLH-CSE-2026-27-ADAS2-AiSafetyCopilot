from __future__ import annotations

import argparse
import csv
from pathlib import Path

REQUIRED_TRACK_COLUMNS = {
    "id",
    "frame",
    "x",
    "y",
    "width",
    "height",
    "xVelocity",
    "yVelocity",
}


def _recording_id(path: Path) -> str:
    return path.name.removesuffix("_tracks.csv")


def _frame_rate(meta_path: Path, fallback: float) -> float:
    if not meta_path.exists():
        return fallback
    with meta_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle), {})
    return float(row.get("frameRate", fallback))


def prepare_file(
    tracks_path: Path,
    writer: csv.DictWriter,
    recording_index: int,
    fallback_frame_rate: float,
) -> int:
    recording = _recording_id(tracks_path)
    meta_path = tracks_path.with_name(f"{recording}_recordingMeta.csv")
    frame_rate = _frame_rate(meta_path, fallback_frame_rate)
    with tracks_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_TRACK_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{tracks_path} is missing HighD columns: {sorted(missing)}"
            )
        count = 0
        for row in reader:
            vehicle_id = int(row["id"])
            track_id = recording_index * 1_000_000 + vehicle_id
            x = float(row["x"])
            y = float(row["y"])
            width = float(row["width"])
            height = float(row["height"])
            velocity_x = float(row["xVelocity"])
            velocity_y = float(row["yVelocity"])
            writer.writerow(
                {
                    "track_id": track_id,
                    "source_id": recording,
                    "class_id": 2,
                    "class_name": "vehicle",
                    "timestamp": float(row["frame"]) / frame_rate,
                    "frame_id": int(row["frame"]),
                    "center_x": x + width / 2.0,
                    "center_y": y + height / 2.0,
                    "velocity_x": velocity_x,
                    "velocity_y": velocity_y,
                    "speed": (velocity_x**2 + velocity_y**2) ** 0.5,
                }
            )
            count += 1
    return count


OUTPUT_COLUMNS = [
    "track_id",
    "source_id",
    "class_id",
    "class_name",
    "timestamp",
    "frame_id",
    "center_x",
    "center_y",
    "velocity_x",
    "velocity_y",
    "speed",
]


def prepare_highd(
    input_dir: Path, output_path: Path, frame_rate: float = 25.0
) -> dict[str, int | str]:
    tracks_files = sorted(input_dir.glob("*_tracks.csv"))
    if not tracks_files:
        raise FileNotFoundError(
            f"no HighD *_tracks.csv files found in {input_dir}; download the official HighD dataset there"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for index, tracks_path in enumerate(tracks_files, start=1):
            total_rows += prepare_file(tracks_path, writer, index, frame_rate)
    return {
        "recordings": len(tracks_files),
        "observations": total_rows,
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert official HighD files to the Person 2 trajectory CSV format."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/highd/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/trajectories/highd.csv")
    )
    parser.add_argument("--frame-rate", type=float, default=25.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing prepared dataset: {args.output}")
    try:
        print(prepare_highd(args.input_dir, args.output, args.frame_rate))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
