import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys


def detectShush(frame, location, ROI, cascade):
    mouths = cascade.detectMultiScale(ROI, 1.05, 14, 0, (2, 2))
    for (mx, my, mw, mh) in mouths:
        mx += location[0]
        my += location[1]
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
    return len(mouths)


def detect(frame, faceCascade, mouthsCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # possible frame pre-processing:
    rows, cols = gray_frame.shape
    total = 0
    for i in range(0, rows):
        for j in range(0, cols):
            total += gray_frame[i, j]

    avg = total / (rows * cols)
    dummy_frame = frame
    if avg > 190 or avg < 65:
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        dummy_frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    blur_frame = cv2.GaussianBlur(dummy_frame, (5, 5), 0)

    faces = faceCascade.detectMultiScale(
        blur_frame, 1.061, 13, 0 | cv2.CASCADE_SCALE_IMAGE, (30, 30))
    detected = 0
    for (x, y, w, h) in faces:
        # ROI for mouth
        x1 = x
        h2 = int(h*2/3)
        y1 = y + h2
        mouthROI = dummy_frame[y1:y+h, x1:x1+w]
        if detectShush(frame, (x1, y1), mouthROI, mouthsCascade) == 0:
            detected += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(faces) == 0:
        result = detectShush(frame, (0, 0), frame, mouthsCascade)
        if result == 0:
            detected += 1
            cv2.rectangle(frame, (0, 0), (cols, rows), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (0, 0), (cols, rows), (0, 255, 0), 2)

    return detected


def run_on_folder(cascade1, cascade2, folder):
    if (folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    windowName = None
    totalCnt = 0
    for f in files:
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            totalCnt += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCnt


def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showframe = True
    while (showframe):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            break
        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showframe = False

    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1 +
              "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
    if (len(sys.argv) == 2):  # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, mouth_cascade, folderName)
        print("Total of ", detections, "detections")
    else:  # no arguments
        runonVideo(face_cascade, mouth_cascade)