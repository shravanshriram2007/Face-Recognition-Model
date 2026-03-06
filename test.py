from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")
results = model(r"E:\Coding\Eye\YOLOtest\testing.jpg",conf = 0.1)
print(results[0].boxes)
img = results[0].plot()

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()