import os
import random
import sys
import cv2
import datetime 
import imutils

res = input("\r\nPlease enter model name: ")

dirLst = os.listdir("data")
if res in dirLst:
    print("\r\nError. Model directory already exists. Please choose some other name or delete the current directory\r\n")
    sys.exit() 

print("Making directory structure")
os.makedirs("data/{}/Annotations".format(res))
os.makedirs("data/{}/ImageSets/Main".format(res))
os.makedirs("data/{}/JPEGImages".format(res)) 

def rrmdir(path):
    for entry in os.scandir(path):
        if entry.is_dir():
            rrmdir(entry)
        else:
            os.remove(entry)
    os.rmdir(path)


fileLst = os.listdir('videos')
fileLst.remove('readme.txt')
if len(fileLst) == 0:
    print("No test video found in videos dir.")
    rrmdir("data/{}".format(res))
    sys.exit(0)
elif len(fileLst) > 1:
    print("Multiple videos found in videos dir. Please enter the name of the video to use\r")
    print(fileLst)
    vname = input("Please enter the full name of the video to use for dataset with extension(.mp4): ")
elif len(fileLst) == 1:
    vname = fileLst[0]
    

cap = cv2.VideoCapture('videos/{}'.format(vname))

# Adjust saveImg value as per your choice
# High value will have less images saved
# Low value will have large images saved 
saveImg = 10

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

if frameCnt == 0:
    print("Unable to process video. Error reading video file. Exiting...\r\n")
    rrmdir("data/{}".format(res))
    sys.exit(0)

endTime = datetime.datetime.now()
diff = endTime - startTime
sec = diff.seconds
fps = round(fCnt/sec, 3)
print("Total FPS {}".format(fps))
print("Total Img saved {}".format(imgCnt))

allFiles = os.listdir("data/{}/JPEGImages".format(res))

testNum = int(imgCnt * 0.10)
testFileLst = []
while True:
    ap = random.choice(allFiles)
    if ap not in testFileLst:
        testFileLst.append(ap)
        if len(testFileLst) == testNum:
            break

trainFileLst = list(set(testFileLst).symmetric_difference(set(allFiles)))

f = open("data/{}/ImageSets/Main/test.txt".format(res), 'w+')
for test in testFileLst:
    test = test.split(".")
    f.write(test[0] + "\n")
f.close()
f = open("data/{}/ImageSets/Main/val.txt".format(res), 'w+')
for test in testFileLst:
    test = test.split(".")
    f.write(test[0] + "\n")
f.close()

f = open("data/{}/ImageSets/Main/train.txt".format(res), 'w+')
for train in trainFileLst:
    train = train.split(".")
    f.write(train[0] + "\n")
f.close()
f = open("data/{}/ImageSets/Main/trainval.txt".format(res), 'w+')
for train in trainFileLst:
    train = train.split(".")
    f.write(train[0] + "\n")
f.close()








