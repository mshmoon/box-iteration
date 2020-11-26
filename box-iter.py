# coding=utf-8
# version:Python3.5
# date:2020-10
# author: Ma Shuhao
import os
import time
import cv2 as cv
import numpy as np

S=10
count=0
sun=100
view=100
noise=100
W=150*10
H=130*10
bigBox=(0,0,W,H)
colorSet=([255,0,255],[255,0,255],[255,255,255],[255,255,255],[125,125,125],[255,0,255],[255,255,255])

def plotDemo(totalIndex):
    global count
    if not os.path.exists("./image"):
        os.mkdir(("./image"))
    img=np.zeros((W,H,3))
    cv.imwrite("img.png",img)
    temp = []
    for item in totalIndex:
        t = item[0][0] * objBox.rowNum + item[0][1]
        temp.append(int(t))
    totalIndex = [temp]
    for i,mK in enumerate(totalIndex):
        img = cv.imread("img.png")
        for j,box in enumerate(objBox.box):
            x,y=mK[j]//objBox.rowNum,mK[j]%objBox.rowNum
            mBox = objBox.decodingBox(x,y,j)
            mBox=[x*S for x in mBox[0]]
            color=colorSet[j]
            cv.rectangle(img=img, pt1=(int(mBox[1]), int(mBox[0])), pt2=(int(mBox[3]), int(mBox[2])),color=color, thickness=20)
        for j, item in enumerate(mK):
            x, y = item // objBox.rowNum, item % objBox.rowNum
            cv.circle(img, (y * 10, x * 10), 1, color, thickness=20)
        cv.imwrite("./image/" + str(0) + ".png", img)
        count += 1

class Box:
    def __init__(self,space,box,solution,boxNum,factor=3):
        self.step=S
        self.space=space
        self.box=box
        self.rowNum=space[0]//self.step
        self.colNum=space[1]//self.step
        self.boxNum=boxNum
        self.factor=factor
        self.solutionNum=solution
        self.centerPointIndex = None
        self.currentBox = None
        self.tempKeyList = None
        self.flag1=None
        self.flag2 = None
        self.adjMat = None

    def calcuIll(self,i, j):
        return (self.rowNum - i) / self.rowNum

    def calcuNoise(self,i, j):
        return i / self.rowNum

    def calcuView(self,i, j):
        return 0

    def processBox(self):
        boxAtt = np.array(
           [
            [1.0, 0.2, 0.0],
            [0.5, 0.2, 0.0],
            [0.5, 0.2, 0.0],
            [1.0, 0.2, 0.0],
            [0.5, 0.5, 0.0],
            [0.2, 1.0, 0.0],
            [0.2, 1.0, 0.0]
           ]
        )
        return boxAtt

    def processSpace(self):
        spacePointAtt = np.zeros((self.rowNum * self.colNum, self.factor))
        funcList=[self.calcuIll,self.calcuNoise,self. calcuView]
        for i in range(self.rowNum):
            for j in range(self.colNum):
                for numFactor in range(self.factor):
                    spacePointAtt[i * self.colNum + j][numFactor] = funcList[numFactor](i, j)
        return spacePointAtt

    def calcuScore(self,boxAtt,spacePointAtt):
        totalScoreBox = []
        scoreMat = boxAtt @ spacePointAtt.T
        for i,score in enumerate(scoreMat):
            totalScoreBox.append(score.reshape(self.rowNum,self.colNum))
        return np.array(totalScoreBox)

    def initCenter(self):
        centerArray=np.zeros((self.boxNum,1,2))
        for i,box in enumerate(self.box):
            width,height=box[0]/self.step/2,box[1]/self.step/2
            centerArray[i][0][0], centerArray[i][0][1] = np.random.randint(width - 1, self.rowNum - width,
                                                                           1), np.random.randint(height - 1,
                                                                                                 self.colNum- height, 1)
            centerArray[i][0][0], centerArray[i][0][1]=self.rowNum//2,self.colNum//2
        return centerArray

    def calcuBoundary(self,centerArray):
        boxBoundary = []
        for i, center in enumerate(centerArray):
            centerX, centerY = center[0][0], center[0][1]
            boxWidth, boxHeight = self.box[i][0], self.box[i][1]
            leftUpX, leftUpY, = centerX - boxWidth // self.step // 2, centerY - boxHeight // self.step // 2
            rightDownX, rightDownY = centerX + boxWidth // self.step // 2, centerY + boxHeight // self.step // 2
            boxBoundary.append([leftUpX,leftUpY,rightDownX,rightDownY])
        return boxBoundary

    def decodingBox(self,center_x,center_y,i):
        boxList=[]
        x1 = center_x - self.box[i][0] //self.step / 2
        x2 = center_x + self.box[i][0] //self.step / 2
        y1 = center_y - self.box[i][1] //self.step / 2
        y2 = center_y + self.box[i][1] //self.step / 2
        boxList.append([x1, y1, x2, y2])
        return boxList

    def calcuIOU(self, box1, box2):
        iou = 0
        xmin1, ymin1, xmax1, ymax1 = box1[0]
        for item in box2:
            xmin2, ymin2, xmax2, ymax2 = item[0]
            s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
            s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
            xmin = max(xmin1, xmin2)
            ymin = max(ymin1, ymin2)
            xmax = min(xmax1, xmax2)
            ymax = min(ymax1, ymax2)
            w = max(0, xmax - xmin)
            h = max(0, ymax - ymin)
            area = w * h
            iou += area / (s1 + s2 - area)
        return iou

    def boxIoU(self,centerArray,boxBoundary):
        step=2
        iouMat=np.zeros((self.boxNum,self.rowNum,self.colNum))
        for i, center in enumerate(centerArray):
            boxX, boxY = self.box[i][0] // self.step // 2, self.box[i][1] // self.step // 2
            x=int(center[0][0])
            y=int(center[0][1])
            lowY = y-step*boxY
            highY = y+step*boxY
            lowX = x-step*boxX
            highX = x+step*boxX
            for j in range(lowX, highX, 1):
                    for k in range(lowY, highY, 1):
                        if j>=boxX and j<=self.rowNum-boxX and k>=boxY and k<=self.colNum-boxY:
                            currBox=self.decodingBox(j,k,i)
                            otherBoxList=[]
                            for m in range(self.boxNum):
                                if i==m:
                                    continue
                                otherJ,otherK=centerArray[m][0][0],centerArray[m][0][1]
                                temp=self.decodingBox(otherJ,otherK,m)
                                otherBoxList.append(temp)
                            iouMat[i][j][k] = self.calcuIOU(currBox,otherBoxList)
                        else:
                            continue
        return iouMat

    def calcuCenter(self,centerArray,boxBoundary,totalScoreBox,iouMat=None):
        meanVal=[]
        meanIoU=[]
        for i,center in enumerate(centerArray):
            if iouMat is None:
                tempX = int(centerArray[i][0][0])
                tempY = int(centerArray[i][0][1])
                meanVal.append(totalScoreBox[i][tempX][tempY])
            else:
                tempX=int(centerArray[i][0][0])
                tempY=int(centerArray[i][0][1])
                meanVal.append(totalScoreBox[i][tempX][tempY])
                meanIoU.append(iouMat[i][tempX][tempY])
        if iouMat is None:
            return meanVal
        else:
            return [meanVal,meanIoU]

    def calcuDist(self,centerArray):
        self.adjMat = np.array(
            [
                [ 0, -1,  1,  1, -1, -1, -1],
                [-1,  0,  1,  1, -1, -1, -1],
                [ 1,  1,  0,  1, -1, -1, -1],
                [ 1,  1,  1,  0,  1, -1, -1],
                [-1, -1, -1,  1,  0,  1,  1],
                [-1, -1, -1, -1,  1,  0,  1],
                [-1, -1, -1, -1,  1,  1,  0],
            ]
        )

        totalList=[]
        self.adjMat[self.adjMat == -1] = 00
        distMat=np.zeros((self.boxNum,1))
        for i ,arr in enumerate(self.adjMat):
            tempList0=[]
            for j,ele in enumerate(arr):
                if ele==1:
                    tempList0.append(i)
                    tempList0.append(j)
            tempList1=list(set(tempList0))
            tempList1.sort(key=tempList0.index)
            totalList.append(tempList1)
        for i,item0 in enumerate(totalList):
            tempList = []
            tempList.extend(item0)
            for j,item1 in enumerate(totalList):
                tempList.extend(item1)
                for k,item2 in enumerate(totalList):
                    if set(tempList)==item2:
                        pass
        return distMat

    def updateCenterScore(self,centerArray,boxBoundary,meanVal,ioutMat,meanIoU,boxTh,flag):
        step=2
        self.IOU=0
        updateCenter=np.zeros((self.boxNum,1,2))
        for i, center in enumerate(centerArray):
            if flag==True:
                if boxTh % self.boxNum != i:
                    updateCenter[i][0][0] = centerArray[i][0][0]
                    updateCenter[i][0][1] = centerArray[i][0][1]
                    continue
            x = int(center[0][0])
            y = int(center[0][1])
            iou,error, coor, score,dist= 0, 100, [x,y],[],[]
            boxX, boxY = self.box[i][0] // self.step // 2, self.box[i][1] // self.step // 2
            lowY = y -  step*boxY
            highY = y +  step*boxY
            lowX = x -  step*boxX
            highX = x +  step*boxX
            for j in range(lowX, highX, 1):
                for k in range(lowY, highY, 1):
                    if j >= boxX and j <= self.rowNum - boxX and k >= boxY and k <= self.colNum - boxY:
                        tempError = meanVal[i] - totalScoreBox[i][j][k]
                        tempIoU = meanIoU[i] - ioutMat[i][j][k]
                        if flag==True:
                            # 此处必须<=和>=
                            if tempError <= error and tempIoU>= iou:
                                coor=[j,k]
                                error = tempError
                                iou=tempIoU
                                score.append(coor)
                        else:
                            if tempError < error:
                                coor = [j, k]
                                error = tempError
                                score.append(totalScoreBox[i][j][k])
                    else:
                        continue
            updateCenter[i][0][0]=(coor[0]-centerArray[i][0][0])//4+centerArray[i][0][0]
            updateCenter[i][0][1]=(coor[1]-centerArray[i][0][1])//4+centerArray[i][0][1]
        return updateCenter

    def gloablAdj(self, centerArray, boxBoundary, meanVal, ioutMat, meanIoU, boxTh, flag):
        boxX1Y1X2Y2=[]
        for i,center in enumerate(centerArray):
            boundingBox=self.decodingBox(center[0][0], center[0][1], i)
            x1,y1,x2,y2=boundingBox[0][0],boundingBox[0][1],boundingBox[0][2],boundingBox[0][3]
            boxX1Y1X2Y2.append([x1,y1,x2,y2])
        freeCenter=[]
        for m in range(self.space[0]//self.step):
            for n in range(self.space[1]//self.step):
                flag=True
                for coor in boxX1Y1X2Y2:
                    x1,y1,x2,y2=coor[0],coor[1],coor[2],coor[3]
                    if (x1<=m and m<=x2) and (y1<=n and n<=y2):
                        flag=False
                        break
                if flag:
                    freeCenter.append([m,n])
        return freeCenter

    def meanSift(self,centerArray,scoreMat):
        epoch=20
        Flag=True
        for n in range(epoch):
            boxBoundary=self.calcuBoundary(centerArray)
            iouMat = self.boxIoU(centerArray, boxBoundary)
            ret=self.calcuCenter(centerArray,boxBoundary,totalScoreBox,iouMat)
            meanVal,meanIoU=ret[0],ret[1]
            if n>=15:
                 Flag=True
            centerArray=self.updateCenterScore(centerArray,boxBoundary,meanVal,iouMat,meanIoU,n,flag=Flag)
            plotDemo(centerArray)
        freeCenter=self.gloablAdj(centerArray,boxBoundary,meanVal,iouMat,meanIoU,n,flag=Flag)
        return freeCenter

    def showScoreMat(self,boxAtt,spacePointAtt):
        """
        :function:可调用显示每个box得分矩阵
        """
        scoreMat = boxAtt @ spacePointAtt.T
        for i in range(np.shape(scoreMat)[0]):
            tempVal = scoreMat[i]
            print(tempVal.reshape(self.rowNum, self.colNum))

def generateVideo():
    """
    :function:生成结果用于动画展示
    """
    fps = 4
    size = (400, 400)
    path = "./image/"
    filelist = os.listdir(path)
    video = cv.VideoWriter("video.avi", cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    for item in filelist:
        if item.endswith(".png"):
            item = path + item
            img = cv.imread(item)
            img = cv.resize(img, (400, 400))
            video.write(img)
    video.release()
    cv.destroyAllWindows()

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='space')
    args = parser.parse_args()
    space=(W,H)
    box = ((120 * 10, 50 * 10),(36 * 10, 28 * 10), (40 * 10, 38 * 10), (30 * 10, 16 * 10), (30 * 10, 18 * 10), (70 * 10, 72 * 10),
           (20 * 10, 16 * 10),)
    boxNum=np.shape(box)[0]
    objBox=Box(space,box,solution=200,boxNum=len(box),factor=3)
    boxAtt=objBox.processBox()
    spacePointAtt=objBox.processSpace()
    totalScoreBox=objBox.calcuScore(boxAtt,spacePointAtt)
    initCenter=objBox.initCenter()
    plotDemo(initCenter)
    time.sleep(3)
    totalIndex=objBox.meanSift(initCenter,totalScoreBox)
    #generateVideo()
