import cv2
def draw_bbox(
    frame,
    bbox,
    track_id,
    confidence=None,
    weight=None
):
    """
    Draw bounding box with track ID, confidence and weight (optional)
    """
    x1, y1, x2, y2 = map(int, bbox)
    # ---------------- DRAW BBOX ----------------
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ---------------- LABEL TEXT ----------------
    label_parts = [f"ID {track_id}"]

    if confidence is not None:
        label_parts.append(f"{confidence:.2f}")

    if weight is not None:
        label_parts.append(f"{weight:.2f}kg")

    label = " | ".join(label_parts)

    # ---------------- LABEL BACKGROUND ----------------
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    cv2.rectangle(
        frame,
        (x1, y1 - th - 8),
        (x1 + tw + 6, y1),
        (0, 255, 0),
        -1
    )

    # ---------------- LABEL TEXT ----------------
    cv2.putText(
        frame,
        label,
        (x1 + 3, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )
def draw_crop_line(frame, crop_top_ratio):
    """
    Visualize cropped area (for debugging / demo)
    """
    h, w, _ = frame.shape
    crop_y = int(h * crop_top_ratio)

    cv2.line(
        frame,
        (0, crop_y),
        (w, crop_y),
        (0, 0, 255),
        2
    )
    cv2.putText(
        frame,
        "CROP LINE",
        (10, crop_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )
