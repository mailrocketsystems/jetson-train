import os
import sys
import cv2
import datetime 
import imutils

res = input("\r\nPlease enter model name: ")

dirLst = os.listdir("data")
if res in dirLst:
    print("\r\nError. Model directory already exists. Please choose some other name or remove the current directory\r\n")
    sys.exit() 

print("Making directory structure")
os.makedirs("data/{}/Annotations".format(res))
os.makedirs("data/{}/ImageSets/Main".format(res))
os.makedirs("data/{}/JPEGImages".format(res)) 

cap = cv2.VideoCapture('videos/testvideo.mp4')

# Adjust saveImg value as per your choice
# High value will have less images saved
# Low value will have large images saved 
saveImg = 50  

imgCnt = 0

frameCnt = 0
fCnt = 0
startTime = datetime.datetime.now()
while True:
    ret, frame = cap.read()
    if not ret:
        break 
    frame = imutils.resize(frame, width=800)
    frameCnt = frameCnt + 1
    fCnt = fCnt + 1

    if frameCnt == saveImg:
        frameCnt = 0
        imgCnt = imgCnt + 1
        cv2.imwrite("data/{}/JPEGImages/file{}.png".format(res, imgCnt), frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

endTime = datetime.datetime.now()
diff = endTime - startTime
sec = diff.seconds
fps = round(fCnt/sec, 3)
print("Total FPS {}".format(fps))
print("Total Img saved {}".format(imgCnt))





