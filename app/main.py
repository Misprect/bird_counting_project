import os
import cv2
import json
import shutil
from fastapi import FastAPI, UploadFile, File

from app.detector_tracker import BirdDetectorTracker
from app.analyzer import BirdAnalyzer
from app.utils import draw_bbox
from app.weight import BirdWeightEstimator

# ---------------- APP SETUP ----------------
app = FastAPI(title="Bird Counting & Weight Estimation API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Process every Nth frame (performance tuning)
FRAME_SKIP = 5


@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    # ---------------- SAVE UPLOADED VIDEO ----------------
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ---------------- OUTPUT PATHS ----------------
    name, _ = os.path.splitext(file.filename)
    annotated_path = os.path.join(UPLOAD_DIR, f"annotated_{name}.mp4")
    json_path = os.path.join(UPLOAD_DIR, f"{name}_results.json")

    # ---------------- INITIALIZE COMPONENTS ----------------
    detector = BirdDetectorTracker(
        conf_thresh=0.35,
        iou_thresh=0.5,
        crop_top_ratio=0.3
    )
    analyzer = BirdAnalyzer()
    weight_estimator = BirdWeightEstimator()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"error": "Unable to open video file"}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        return {"error": "Unable to write output video"}

    frame_idx = 0           # actual video frame index
    processed_frames = 0    # frames actually analyzed

    # ---------------- PROCESS VIDEO ----------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # -------- FRAME SKIPPING --------
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        detections = detector.process_frame(frame)

        # Add weight + draw boxes
        for det in detections:
            det["estimated_weight_kg"] = weight_estimator.estimate(det["bbox"])
            draw_bbox(
                frame,
                det["bbox"],
                det["track_id"],
                det["confidence"]
            )

        analyzer.update(frame_idx, detections)
        writer.write(frame)

        processed_frames += 1
        frame_idx += 1

    cap.release()
    writer.release()

    # ---------------- SAVE JSON ----------------
    results = {
        "summary": analyzer.get_summary(),
        "frames": analyzer.get_frame_detections()
    }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- RESPONSE ----------------
    return {
        "message": "Processing complete",
        "total_frames": frame_idx,
        "processed_frames": processed_frames,
        "frame_skip": FRAME_SKIP,
        "unique_birds": results["summary"]["total_unique_birds"],
        "annotated_video": annotated_path,
        "results_json": json_path
    }
