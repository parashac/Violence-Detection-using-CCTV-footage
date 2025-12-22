import os
import cv2

path = "D:/Major project/dataset/SCVD/video_clips/Frames/violence"
resize = (224, 224)

for folder in os.listdr(path):
    video = os.path.join(path,folder)
    if not os.path.isdir(video):
        continue

    for frame in sorted(os.listdir(video)):
        frame = os.path.join(video, frame)

        if not frame.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img = cv2.imread(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.equalizeHist(gray)
        cv2.imwrite(frame, gray)
print(" All frames processed.")