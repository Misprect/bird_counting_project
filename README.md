---

# Bird Counting & Weight Estimation System

**Computer Vision · YOLOv8 · FastAPI**

---

## Project Overview

This project implements an **end-to-end bird (poultry) counting system** using **YOLOv8 object detection and tracking**.
The system processes video inputs to:

- Detect birds (COCO class: bird)
- Track **unique birds across frames**
- Count **birds per frame**
- Estimate **approximate bird weight**
- Generate **annotated videos**
- Produce structured **JSON analytics**
- Provide a **FastAPI backend** for video uploads and processing

The solution is optimized for **real-world poultry farm footage** where noise (eggs, roofs, cages) and long video durations are common.

---

## Key Features

- ✅ YOLOv8-based bird detection
- ✅ Persistent tracking with unique IDs
- ✅ Frame-wise bird count
- ✅ Total unique bird count
- ✅ Bounding-box-based weight estimation
- ✅ Region-based frame cropping (noise reduction)
- ✅ Frame skipping for performance
- ✅ Annotated output videos
- ✅ JSON-based analytical output
- ✅ REST API for video uploads (FastAPI)
- ✅ **Accuracy & stability analysis with crop info**

---

## System Architecture

```
Video Upload
    ↓
Frame Preprocessing (crop + skip)
    ↓
YOLOv8 Detection & Tracking
    ↓
Bird Analyzer (counts + IDs)
    ↓
Weight Estimation
    ↓
Annotated Video + JSON Output
```

---

## Project Folder Structure

```
BIRD_COUNTING_PROJECT/
│
├── app/
│   ├── __init__.py
│   ├── analyzer.py          # Bird counting & tracking analytics
│   ├── detector_tracker.py  # YOLOv8 detection + tracking + cropping
│   ├── weight.py            # Weight estimation logic
│   ├── utils.py             # Drawing & visualization helpers
│   └── main.py              # FastAPI backend
│
├── data/
│   └── videos/              # Raw input videos
│
├── uploads/
│   ├── poultry1.mp4
│   ├── annotated_poultry1.mp4
│   ├── poultry1_results.json
│   └── ...
│
├── outputs/
│   ├── detections/           # Offline test JSONs
│   └── experiment/           # Accuracy & experiment results (with crop info)
│
├── experiments/
│   └── accuracy_check.py     # Stability & coverage analysis (includes crop heights)
│
├── test_detector.py          # Local testing script
├── requirements.txt
├── yolov8n.pt                # YOLOv8 weights
└── README.md
```

---

## Installation & Setup

### 1. Create virtual environment (recommended)

```bash
conda create -n birdcount python=3.10
conda activate birdcount
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Backend (FastAPI)

Start the server from project root:

```bash
uvicorn app.main:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## Uploading & Processing a Video

### Endpoint

```
POST /process-video/
```

### Input

- Video file (`.mp4`)

### Output

- Annotated video (`annotated_<name>.mp4`)
- JSON analytics (`<name>_results.json`)

Both are saved automatically to:

```
uploads/
```

---

## JSON Output Structure

```json
{
  "summary": {
    "total_unique_birds": 5,
    "total_frames_with_birds": 120,
    "frame_wise_counts": { ... }
  },
  "frames": {
    "0": [
      {
        "track_id": 1,
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.87,
        "estimated_weight_kg": 1.4
      }
    ]
  }
}
```

---

## Performance Optimizations

### 🔹 Frame Cropping

- Removes **top 30%** of frame to eliminate:

  - Eggs
  - Roofs
  - Hatchery noise

### 🔹 Frame Skipping

- Processes **every 5th frame**
- Reduces compute while preserving tracking accuracy

### 🔹 Persistent Tracking

- Unique IDs maintained across frames using YOLOv8 tracking

---

## Weight Estimation Logic

Bird weight is estimated using **bounding box area**:

```python
weight ≈ bbox_area × scale_factor
```

Provides an **approximate but consistent estimate** suitable for analytics (not medical use).

---

## Accuracy & Stability Analysis

Run:

```bash
python experiments/accuracy_check.py
```

### What it does:

- Loads all JSON results from `outputs/detections/`
- Computes:

  - Detection coverage (frames with birds / total frames)
  - Track persistence (frames per bird ID)
  - Average birds per frame
  - **Original vs cropped frame heights** (top 30% crop removed)

- Saves per-video accuracy JSON in:

```
outputs/experiment/
```

### Example output (JSON):

```json
{
  "video": "poultry3_results.json",
  "total_frames": 18612,
  "frames_with_birds": 13200,
  "detection_coverage": 0.71,
  "track_lengths": {
    "1": 5000,
    "2": 4800
  },
  "average_birds_per_frame": 4.5,
  "original_frame_height": 720,
  "cropped_frame_height": 504
}
```

---

## Local Testing (Without API)

```bash
python test_detector.py
```

Useful for:

- Debugging detection
- Verifying tracking
- Offline experimentation

---

## What This Project Demonstrates

- Real-world CV pipeline design
- Efficient long-video processing
- Modular, production-ready code
- REST API integration
- Practical ML engineering decisions
-Public poultry videos were used for testing. Sample inputs and outputs are included in the repository.
---

## Future Improvements

- Bird re-identification across camera cuts
- Real-world calibrated weight models
- GPU batch inference
- Frontend dashboard
- Multi-camera support

---

## Author

**Aryaman Jain**
B.Tech CSE (AI/ML)
Focus: Computer Vision, Deep Learning, Applied AI
Submitted as an ML Intern.

---
