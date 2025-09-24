import cv2
import argparse
import os

def detect(frame, classifier):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objs = classifier.detectMultiScale(gray, 1.3, 5)
    return objs

frame_count = 0

classifier1_catch = 0

classifier2_catch = 0

classifier3_catch = 0

parser = argparse.ArgumentParser()

parser.add_argument("--video", help="path to the video", default="./video/video.mp4")

parser.add_argument("--classifier1", help="using classifier", default="./998_763_14_35.xml")

parser.add_argument("--classifier2", help="using classifier", default="./haarcascade_fullbody.xml")

parser.add_argument("--classifier3", help="using classifier", default="./car.xml")

args = parser.parse_args()

path = args.video

pathC1 = args.classifier1

pathC2 = args.classifier2

pathC3 = args.classifier3

cap = cv2.VideoCapture(path)

fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./video/output.avi', fourcc, fps, (640, 360))

if not os.path.exists("./frames"):
    os.makedirs("./frames")

if(cap.isOpened() == False):
    print("Error opening video stream or file")

classifier1 = cv2.CascadeClassifier(pathC1)

classifier2 = cv2.CascadeClassifier(pathC2)

classifier3 = cv2.CascadeClassifier(pathC3)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        objs = detect(frame, classifier1)
        res = frame.copy()
        for (x, y, w, h) in objs:
            classifier1_catch += 1
            cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        objs = detect(frame, classifier2)
        for (x, y, w, h) in objs:
            classifier2_catch += 1
            cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 255), 2)

        objs = detect(frame, classifier3)
        for (x, y, w, h) in objs:
            classifier3_catch += 1
            cv2.rectangle(res, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
        cv2.imshow("Press q to quit", res)
        out.write(res)
        key = cv2.waitKey(16) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            frame_filename = f"./frames/frame_{frame_count:04d}.jpg"
            frame_count += 1
            cv2.imwrite(frame_filename, res)
    else:
        break
    
print(f"classifier: {pathC1} count: {classifier1_catch:06d}\n")
    
print(f"classifier: {pathC2} count: {classifier2_catch:06d}\n")
    
print(f"classifier: {pathC3} count: {classifier3_catch:06d}\n")
