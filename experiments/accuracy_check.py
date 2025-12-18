import os
import json
from collections import defaultdict
import cv2

# ---------------- PROJECT ROOT ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # experiments folder
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DETECTIONS_DIR = os.path.join(ROOT_DIR, "outputs", "detections")
EXPERIMENT_DIR = os.path.join(ROOT_DIR, "outputs", "experiment")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# ---------------- LIST JSON FILES ----------------
if not os.path.exists(DETECTIONS_DIR):
    raise FileNotFoundError(f"Detections directory not found: {DETECTIONS_DIR}")

json_files = [f for f in os.listdir(DETECTIONS_DIR) if f.endswith(".json")]

if not json_files:
    raise FileNotFoundError(f"No JSON files found in {DETECTIONS_DIR}")

# ---------------- PROCESS EACH FILE ----------------
CROP_TOP_RATIO = 0.3  # Must match the detector's crop_top_ratio

for file_name in json_files:
    result_path = os.path.join(DETECTIONS_DIR, file_name)

    with open(result_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", {})
    if not frames:
        print(f"No frames found in {file_name}, skipping...")
        continue

    track_lengths = defaultdict(int)
    frame_counts = []

    for frame_id, detections in frames.items():
        frame_counts.append(len(detections))
        for det in detections:
            track_lengths[det["track_id"]] += 1

    total_frames = len(frames)
    frames_with_birds = sum(1 for c in frame_counts if c > 0)
    avg_birds_per_frame = sum(frame_counts) / total_frames if total_frames else 0

    # ---------------- FRAME SIZE / CROP INFO ----------------
    orig_height = None
    cropped_height = None
    video_name = file_name.replace("_results.json", "")
    video_path = os.path.join(ROOT_DIR, "data", "videos", f"{video_name}.mp4")
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            orig_height = frame.shape[0]
            cropped_height = int(orig_height * (1 - CROP_TOP_RATIO))
            print(f"\n🎬 Video: {video_name}")
            print(f"Original frame height: {orig_height}")
            print(f"Cropped frame height (top {CROP_TOP_RATIO*100:.0f}% removed): {cropped_height}")
        cap.release()

    # ---------------- PRINT METRICS ----------------
    print(f"\n📊 Accuracy / Stability Analysis for {file_name}")
    print("----------------------------------")
    print("Total processed frames:", total_frames)
    print("Frames with birds:", frames_with_birds)
    print("Detection coverage:", round(frames_with_birds / total_frames, 2) if total_frames else 0)

    print("\nTrack persistence (frames per bird):")
    for tid, count in track_lengths.items():
        print(f" Bird {tid}: {count} frames")

    print("\nAverage birds per frame:", round(avg_birds_per_frame, 2))

    # ---------------- SAVE RESULTS ----------------
    accuracy_results = {
        "video": file_name,
        "total_frames": total_frames,
        "frames_with_birds": frames_with_birds,
        "detection_coverage": round(frames_with_birds / total_frames, 2) if total_frames else 0,
        "track_lengths": dict(track_lengths),
        "average_birds_per_frame": round(avg_birds_per_frame, 2),
        "original_frame_height": orig_height,
        "cropped_frame_height": cropped_height
    }

    output_json = os.path.join(EXPERIMENT_DIR, f"{os.path.splitext(file_name)[0]}_accuracy.json")
    with open(output_json, "w") as f:
        json.dump(accuracy_results, f, indent=2)

    print(f"✅ Accuracy results saved to: {output_json}")
