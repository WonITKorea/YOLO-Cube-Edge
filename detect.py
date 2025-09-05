import cv2
from ultralytics import YOLO

# Define rectangular ROIs as (x1, y1, x2, y2)
ROIS = [
    (10, 510, 450, 710),  # ROI 0
    (500, 510, 780, 710), # ROI 1
    (880, 510, 1270, 710),  # ROI 2
    (10, 300, 450, 500),  # ROI 3
    (500, 300, 780, 500),  # ROI 4
    (880, 300, 1270, 500),  # ROI 5
    (10, 10, 450, 290),  # ROI 6
    (500, 10, 780, 290),  # ROI 7
    (880, 10, 1270, 290)  # ROI 8
]

ROI_COLOR_DEFAULT = (255, 0, 0)   # Blue
ROI_COLOR_TRIGGERED = (0, 255, 255) # Green

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        "format=(string)NV12, framerate=(fraction){}/1 ! ".format(framerate) +
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

def rect_iou(a, b):
    # a,b are (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter if (area_a + area_b - inter) > 0 else 1e-6
    return inter / union

def main():
    # Load TensorRT engine or .pt; specify task if needed
    model = YOLO("YOLOCUBE.engine", task="detect")

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Camera open. Starting inference...")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame.")
                break

            # Run inference
            results = model(frame, verbose=False)
            
            # Access the first result object
            r = results[0]

            # Track ROI triggers
            roi_triggered = [False] * len(ROIS)

            # --- THIS IS THE CORRECTED SECTION ---
            # Extract detections
            for box in r.boxes:
                # Step 1: Get the tensor for the box's coordinates
                coords_tensor = box.xyxy
                # Step 2: Convert to list and select the first (and only) element
                coords_list = coords_tensor.tolist()[0]
                # Step 3: Now map to int safely
                x1, y1, x2, y2 = map(int, coords_list)

                det_rect = (x1, y1, x2, y2)

                # Check against all ROIs
                for i, roi in enumerate(ROIS):
                    if rect_iou(det_rect, roi) > 0:
                        roi_triggered[i] = True

            # Annotate detections
            annotated = r.plot()

            # Draw ROIs with color based on trigger
            for i, roi in enumerate(ROIS):
                x1, y1, x2, y2 = roi
                color = ROI_COLOR_TRIGGERED if roi_triggered[i] else ROI_COLOR_DEFAULT
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ROI {i} {'TRIGGER' if roi_triggered[i] else ''}"
                cv2.putText(annotated, label, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Optional: print which ROIs triggered this frame
            triggered_indices = [i for i, triggered in enumerate(roi_triggered) if triggered]
            if triggered_indices:
                print(f"Action: ROI(s) {triggered_indices} triggered")

            cv2.imshow("YOLO Multi-ROI", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

