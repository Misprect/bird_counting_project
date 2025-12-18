from collections import defaultdict

class BirdAnalyzer:
    def __init__(self):
        self.unique_birds= set()
        self.frame_counts= defaultdict(int)
        self.frame_detections= defaultdict(list)

    def update(self, frame_idx, detections):
        """
        detections: list of detects from detector_tracker
        """
        self.frame_counts[frame_idx]= len(detections)

        for det in detections:
            tid= det["track_id"]
            self.unique_birds.add(tid)
            self.frame_detections[frame_idx].append(det)

    def get_summary(self):
        return {
            "total_unique_birds": len(self.unique_birds),
            "total_frames_with_birds": sum(
                1 for c in self.frame_counts.values() if c > 0
            ),
            "frame_wise_counts": dict(self.frame_counts)
        }
 
    def get_frame_detections(self):
        return dict(self.frame_detections)