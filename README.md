# AI Safety Copilot for Autonomous Driving

A Computer Vision and Deep Learning-Based Predictive Driver Assistance System Using CARLA Simulation.

---

## 📌 Project Overview
The **AI Safety Copilot** continuously monitors driving environments within the CARLA Autonomous Driving Simulator[cite: 1]. Moving beyond traditional reactive ADAS, this system leverages Computer Vision and Deep Learning to predict hazards, assess dynamic collision risks, and provide proactive, explainable safety recommendations[cite: 1].

---

## 🎯 Key Objectives
* **Object & Environment Perception:** Build a real-time computer vision pipeline to detect vehicles, pedestrians, lane boundaries, traffic signs, and traffic lights[cite: 1].
* **Tracking & Trajectory Prediction:** Implement multi-object tracking and movement forecasting for surrounding road users[cite: 1].
* **Dynamic Risk Assessment:** Analyze traffic scenarios to project collision risks and trigger timely safety alerts[cite: 1].
* **Explainable AI (XAI):** Integrate an XAI decision module delivering transparent, human-understandable reasoning for every decision[cite: 1].

---

## 📄 Abstract
This project evaluates deep learning perception and prediction algorithms by combining high-fidelity simulated sensor outputs from the CARLA Simulator with standard real-world autonomous driving benchmark datasets[cite: 1]. Unlike conventional driver-assistance systems that act only after detecting immediate hazards[cite: 1], the proposed architecture integrates multi-object tracking, trajectory prediction, dynamic risk assessment, and Explainable AI (XAI)[cite: 1]. By leveraging simulation-based environments, the framework provides a robust, hardware-free testbed for developing proactive autonomous driving safety mechanisms[cite: 1].

---

## 📊 Datasets

* **Dataset Overview:**
  * **CULane Dataset:** A large-scale benchmark for urban lane detection containing complex traffic scenarios, severe illumination shifts, and diverse lane structures.
  * **BDD100K Dataset:** A diverse, multi-task driving dataset featuring 100,000 video clips captured across varied weather conditions, times of day, and urban environments. Used for object detection, lane marking detection, and drivable area analysis[cite: 1].
  * **CARLA Telemetry:** Simulated multi-modal sensor streams (RGB cameras, Depth maps, LiDAR, IMU/GNSS)[cite: 1].

* **Downloadable Links:**
  * [Download CULane Dataset (Kaggle Repository)](https://www.kaggle.com/datasets/manideep1108/culane)
  * [Download BDD100K Dataset (Berkeley DeepDrive Official)](https://bdd-data.berkeley.edu/)

---

## 🔬 Research & References

* **Risk Assessment in Autonomous Driving: A Comprehensive Survey of Risk Sources, Methodologies, and System Architectures — [Springer Article Link](https://link.springer.com/article/10.1007/s43684-025-00112-1)
* **Promoting CAV Deployment by Enhancing the Perception Phase of Autonomous Driving Using Explainable AI — [Purdue e-Pubs Download Link](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1041&context=ccat)
* **

---

## 👥 Team Members

| Name | Roll No |
| :--- | :--- |
| **RAYAPUREDDI RISHI SRICHARAN** | 2420030638 |
| **CHITTARI SUMA PRIYA VARMA**   | 2420030337 |
| **TEJASWI KANNEKANTI**          | 2420030517 |

## Person 1 + Person 2 Prototype

The working backend path is:

`video -> YOLO detections -> Pydantic Detection -> Ultralytics ByteTrack -> lifecycle/occlusion state -> image-space motion -> trajectory prediction -> JSON/video output`

Tracking uses the maintained ByteTrack implementation bundled with `ultralytics==8.4.154`, invoked through `YOLO.track(..., tracker="bytetrack.yaml", persist=True)`. The project requires `lap>=0.5.12` for its association solver. The root [bytetrack.yaml](bytetrack.yaml) configures high-confidence association, low-confidence recovery, new-track activation, matching, and the lost-track buffer. `TrackManager` adds the project lifecycle, velocity, history, and Pydantic contract around those genuine tracker IDs.

Run the pipeline with the existing `.venv`:

```powershell
.\.venv\Scripts\python.exe -m src.main --video data/videos/example.mp4 --model yolo11s.pt --prediction-mode baseline
```

For CPU-friendly development, process every fifth source frame, use a smaller YOLO input, cap the run at 100 processed frames, and log progress every 10 processed frames:

```powershell
python -m src.main --video data/videos/driving.mp4 --model yolo11n.pt --prediction-mode baseline --frame-skip 5 --imgsz 416 --max-frames 100 --progress-interval 10
```

`frame_skip`, `imgsz`, `max_frames`, and `progress_interval` can also be set under `video` in `config.yaml`. Defaults are full frame-rate processing (`frame_skip: 1`), `imgsz: 416`, no frame limit, and progress every 10 processed frames. Source frame IDs and timestamps are retained when frames are skipped, and velocity remains based on elapsed source-video time. Use `frame_skip: 1` and omit `max_frames` for final full-frame-rate runs; increase `imgsz` when higher detector detail is needed.

`baseline` is a constant-velocity development predictor and is not a trained LSTM. Use `--prediction-mode lstm` after configuring a trained checkpoint in `config.yaml`. The prediction horizon is calculated as `fps * horizon_seconds`; the default is 3 seconds. Tracking thresholds, history length, and model settings are in `config.yaml`.

Train a checkpoint only after placing a selected nuScenes/HighD-derived trajectory file at the configured path:

```powershell
.\.venv\Scripts\python.exe -m src.trajectory.train --dataset data/trajectories.csv --output-dir artifacts/trajectory
```

The loader accepts CSV or flat JSON with `track_id`, `timestamp`, `frame_id`, `center_x`, and `center_y`, plus optional class, velocity, and speed fields. It groups strictly by track ID, orders observations temporally, splits by trajectory, and fits normalization only on training trajectories. The current repository has no such dataset locally, so training, ADE, FDE, validation loss, and test loss are not yet measured. The command fails clearly rather than substituting the driving video or fabricating metrics.

For HighD, place the official raw files without renaming them under `data/highd/raw/`. The adapter expects files such as `01_tracks.csv` and `01_recordingMeta.csv`, where the tracks file contains the official HighD columns `id`, `frame`, `x`, `y`, `width`, `height`, `xVelocity`, and `yVelocity`. Prepare a separate normalized dataset with:

```powershell
.\.venv\Scripts\python.exe -m src.trajectory.prepare_highd
```

This writes `data/trajectories/highd.csv`, never overwrites raw files, preserves temporal order, converts box coordinates to centers, computes timestamps and speed, and namespaces vehicle IDs across recordings. Training splits by recording source to avoid leakage between adjacent windows. No HighD files are currently present in this workspace, so this command currently stops with a missing-input message.

The LSTM checkpoint and scaler are saved under `artifacts/trajectory/`. LSTM video inference requires both files and raises an actionable error when either is missing. Baseline and LSTM outputs use separate names: `*_tracking.json`/`*_tracked.mp4` versus `*_tracking_lstm.json`/`*_tracked_lstm.mp4`.

Start the contract API with:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --reload
```

Endpoints:

- `GET /health`
- `POST /tracking/frame`, accepting and returning `TrackingFrameOutput`

### Person 3 Integration Contract

Person 3 consumes `TrackingFrameOutput` from `src.tracking.schemas`. Each frame contains `frame_id`, `timestamp`, `tracked_objects`, and `predictions`. Each tracked object includes persistent `track_id`, class, bounding box, center, image-space velocity in pixels/second, speed, lifecycle `status` (`active` or `occluded`), and `missed_frames`. Each prediction contains the same `track_id`, `horizon_seconds`, and future `(x, y)` image coordinates. Physical speed, collision risk, uncertainty, and safety recommendations are intentionally outside this module.
