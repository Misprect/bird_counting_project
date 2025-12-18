class BirdWeightEstimator:
    """
    Simple heuristic:
    Bird weight is proportional to bounding box area.
    This is NOT biological accuracy, only estimation.
    """

    def __init__(self, scale_factor=0.0008):
        # scale_factor tuned empirically
        self.scale_factor = scale_factor

    def estimate(self, bbox):
        """
        bbox: [x1, y1, x2, y2]
        returns estimated weight in kg
        """
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        weight = area * self.scale_factor
        return round(weight, 3)
