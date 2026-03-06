import time
import cv2
import cvzone
from ultralytics import YOLO

# Minimum confidence to show a box
confidence_threshold = 0.1

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Load YOLO model
model = YOLO("models/best.pt")
classNames = ["fake", "real"]

prev_frame_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    new_frame_time = time.time()

    # Get predictions
    results = model(img, stream=True, verbose=False)

    for r in results:
        # Convert all boxes to numpy arrays
        if len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()  # [[x1, y1, x2, y2], ...]
        confs = r.boxes.conf.cpu().numpy()  # [conf1, conf2, ...]
        clss = r.boxes.cls.cpu().numpy()    # [cls1, cls2, ...]

        for x1, y1, x2, y2, conf, cls in zip(boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], confs, clss):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)
            conf = float(conf)

            if conf > confidence_threshold:
                color = (0, 255, 0) if classNames[cls] == "real" else (0, 0, 255)

                # Draw corner rectangle
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorC=color, colorR=color)

                # Draw label
                cvzone.putTextRect(
                    img,
                    f'{classNames[cls].upper()} {int(conf*100)}%',
                    (max(0, x1), max(35, y1)),
                    scale=2,
                    thickness=2,
                    colorR=color,
                    colorB=color
                )

    # Display FPS
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        cv2.putText(img, f"FPS: {int(fps)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    prev_frame_time = new_frame_time

    cv2.imshow("Real/Fake Detection", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()