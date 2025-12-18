from ultralytics import YOLO

class BirdDetectorTracker:
    def __init__(self, conf_thresh=0.4, iou_thresh=0.5, crop_top_ratio=0.3):
        self.model = YOLO("yolov8n.pt")
        self.conf = conf_thresh
        self.iou = iou_thresh
        self.crop_top_ratio = crop_top_ratio

    def process_frame(self, frame):
        h, w, _ = frame.shape

        # --------- APPLY CROP (TOP REMOVAL) ----------
        crop_start_y = int(h * self.crop_top_ratio)
        cropped_frame = frame[crop_start_y:h, :]

        # --------- RUN YOLO ON CROPPED FRAME ----------
        results = self.model.track(
            cropped_frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            classes=[14]  # COCO bird
        )

        detections = []

        if results[0].boxes.id is not None:
            for box, tid, conf in zip(
                results[0].boxes.xyxy,
                results[0].boxes.id,
                results[0].boxes.conf
            ):
                x1, y1, x2, y2 = box.tolist()

                # --------- MAP BACK TO ORIGINAL FRAME ----------
                detections.append({
                    "track_id": int(tid),
                    "bbox": [
                        x1,
                        y1 + crop_start_y,
                        x2,
                        y2 + crop_start_y
                    ],
                    "confidence": float(conf)
                })

        return detections
