import os 
import matplotlib.pyplot as plt

res = input("\r\nPlease enter model name: ")

files = os.listdir("models/{}".format(res))
fileLst = []
for file in files:
    if "mb" in file:
        fileLst.append(file)

fileLst.sort(key=lambda x: int(x.split("-")[3]))

x = []
y = []

for file in fileLst:
    file = file.split("-")
    loss = file[5]
    lossNum = loss.split(".pth")
    y.append(round(float(lossNum[0]), 3))
    x.append(file[3])

bestChkpt = ""
runOnce = True 
prevLoss = 0.0
for file in fileLst:
    origFile = file
    file = file.split("-")
    loss = file[5]
    lossNum = loss.split(".pth")
    if runOnce:
        prevLoss = lossNum
        runOnce = False
    else:
        if lossNum < prevLoss:
            prevLoss = lossNum
            bestChkpt = origFile #file[0] + file[1] + file[2] + file[3] + file[4] + file[5]




print("Bes Checkpoint: {}".format(bestChkpt))
plt.plot(x, y)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training result")
plt.show()

