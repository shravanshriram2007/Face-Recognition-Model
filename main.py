import time
import cv2
import cvzone
from ultralytics import YOLO

confidence_threshold = 0.01  # <-- lowered for debugging

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("runs/detect/train/weights/best.pt")
classNames = ["fake", "real"]

prev_frame_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    new_frame_time = time.time()

    results = model(img, stream=True, verbose=True, imgsz=640)  # verbose=True to see raw output

    for r in results:
        if len(r.boxes) == 0:
            print("No boxes detected this frame")  # debug
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()

        for x1, y1, x2, y2, conf, cls in zip(boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], confs, clss):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)
            conf = float(conf)

            # Ensure valid box dimensions
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                print(f"Invalid box: ({x1},{y1},{x2},{y2})")  # debug
                continue

            if conf > confidence_threshold:
                color = (0, 255, 0) if classNames[cls] == "real" else (0, 0, 255)
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(
                    img,
                    f'{classNames[cls].upper()} {int(conf*100)}%',
                    (max(0, x1), max(35, y1)),
                    scale=2, thickness=2,
                    colorR=color, colorB=color
                )

    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        cv2.putText(img, f"FPS: {int(fps)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    prev_frame_time = new_frame_time

    cv2.imshow("Real/Fake Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()