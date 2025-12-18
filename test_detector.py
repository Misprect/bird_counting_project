import cv2
import json
import os

from app.detector_tracker import BirdDetectorTracker
from app.analyzer import BirdAnalyzer

# ---------------- CONFIG ----------------
VIDEO_PATH = "data/videos/poultry3.mp4"
OUTPUT_DIR = "outputs/detections"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "poultry3_results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Crop top 30% (to remove eggs / roof / noise)
detector = BirdDetectorTracker(
    conf_thresh=0.35,
    iou_thresh=0.5,
    crop_top_ratio=0.3
)

analyzer = BirdAnalyzer()
cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0

# ---------------- PROCESS VIDEO ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.process_frame(frame)

    # Update analyzer for both metrics
    analyzer.update(frame_idx, detections)

    if detections:
        print(f"Frame {frame_idx} detections: {detections}")

    frame_idx += 1

cap.release()

# ---------------- SAVE RESULTS ----------------
results = {
    "summary": analyzer.get_summary(),
    "frames": analyzer.get_frame_detections()
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

# ---------------- REPORT ----------------
print("\n✅ Analysis complete")
print("Total frames processed:", frame_idx)
print("Unique birds detected:", results["summary"]["total_unique_birds"])
print("Results saved to:", OUTPUT_JSON)
