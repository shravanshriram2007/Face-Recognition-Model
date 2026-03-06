from time import time
import os
import cv2

# -------------------- CONFIG --------------------
classID = 0  # 0 = fake, 1 = real

outputFolderPath = 'Dataset/all'
confidence = 1.3  # scaleFactor for Haar
minNeighbors = 5
blurThreshold = 35
camWidth, camHeight = 640, 480
floatingPoint = 6
# ------------------------------------------------

os.makedirs(outputFolderPath, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

# Load Haar Cascade
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    success, img = cap.read()
    if not success:
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        imgGray,
        scaleFactor=confidence,
        minNeighbors=minNeighbors
    )

    listBlur = []
    listInfo = []

    for (x, y, w, h) in faces:

        imgFace = img[y:y + h, x:x + w]

        if imgFace.size == 0:
            continue

        # Blur detection
        blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

        if blurValue > blurThreshold:
            listBlur.append(True)
        else:
            listBlur.append(False)

        # Normalize for YOLO format
        ih, iw, _ = img.shape
        xc, yc = x + w / 2, y + h / 2

        xcn = round(xc / iw, floatingPoint)
        ycn = round(yc / ih, floatingPoint)
        wn = round(w / iw, floatingPoint)
        hn = round(h / ih, floatingPoint)

        listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, f"Blur:{blurValue}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0), 2)

    # Save image + label
    if faces is not None and len(faces) > 0:
        if all(listBlur) and listBlur != []:
            timeNow = str(time()).replace('.', '')
            imgPath = f"{outputFolderPath}/{timeNow}.jpg"
            labelPath = f"{outputFolderPath}/{timeNow}.txt"

            cv2.imwrite(imgPath, img)

            with open(labelPath, 'w') as f:
                for info in listInfo:
                    f.write(info)

            print("Saved:", timeNow)

    cv2.imshow("Data Collection - Haar", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()