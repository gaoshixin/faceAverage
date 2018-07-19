#!/usr/bin/env python

# Copyright (c) 2016 Satya Mallick <spmallick@learnopencv.com>
# All rights reserved. No warranty, explicit or implicit, provided.


import os
import cv2
import numpy as np
import math
from PIL import Image
#import sys

# 从目录中读取点的txt文件
def readPoints(path) :
    # 创建一个空数组，存放点
    pointsArray = []

    #遍历目录中所有txt文件，并独处里边的点
    for filePath in os.listdir(path):
        
        if filePath.endswith(".txt"):
            
            #创建一个点数组
            points = []        
            
            # 读取点
            with open(os.path.join(path, filePath)) as file :#os.path.join是拼接路径的函数
                for line in file :
                    x, y = line.split()
                    points.append((int(x), int(y)))
            
            # 存储读取的点
            pointsArray.append(points)
            
    return pointsArray

# 从文件中读取图片
def readImages(path) :
    
    #创建空列表装图片的数组
    imagesArray = []
    
    #遍历目录，读取图片
    for filePath in os.listdir(path):
        if filePath.endswith(".jpg"):
            # Read image found.
            img = cv2.imread(os.path.join(path,filePath))

            # Convert to floating point
            img = np.float32(img)/255.0

            # Add to array of images
            imagesArray.append(img)
            
    return imagesArray
                
# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
#先用两个进行相似变换，构造出假的第三组点，然后用来计算出一个仿射变换矩阵
def similarityTransform(inPoints, outPoints) :
    #根据相似变换，制造一对虚拟的点
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)
  
    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
    
    inPts.append([np.int(xin), np.int(yin)])
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
    
    outPts.append([np.int(xout), np.int(yout)])
    #初始三组点：[[504, 558], [886, 556], [693, 226]]，目标：[[180, 200], [420, 200], [300, -7]]
    #estimateRigidTransform计算二维点对之间的最优仿射变换矩阵，shape=[2x3]
    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
    
    return tform


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect);
   
    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]));

   
    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList();

    # Find the indices of triangles in the points array

    delaunayTri = []
    
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        

    
    return delaunayTri


def constrainPoint(p, w, h) :
    p =  ( min( max( p[0], 0 ) , w - 1 ) , min( max( p[1], 0 ) , h - 1 ) )
    return p;

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect



if __name__ == '__main__' :
    
    path = 'C:/Users/Administrator/Desktop/face_recognition/FaceAverage/test/'
    
    # Dimensions of output image
    w = 600;
    h = 600;

    # Read points for all images
    allPoints = readPoints(path)

    # Read all images
    images = readImages(path)
    
    # 眼睛位置，左眼角位于距左边0.3w，距顶部1/3处，右眼角位于距右边0.3w，距顶部1/3处。
    eyecornerDst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ]
    
    imagesNorm = []
    pointsNorm = []
    
    # 定义8个边界点。用来画德劳内三角形
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ]);
    
    # Initialize location of average points to 0s初始化局部平均点，全为0
    pointsAvg = np.array([(0,0)]* ( len(allPoints[0]) + len(boundaryPts) ), np.float32())
    n = len(allPoints[0])

    numImages = len(images)
    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    
    for i in range(0, numImages):

        points1 = allPoints[i]
        # Corners of the eye in input image得到眼睛的位置36、45
        eyecornerSrc  = [ allPoints[i][36], allPoints[i][45] ] 
        
        # 通过眼睛对应的位置，计算仿射变换矩阵
        tform = similarityTransform(eyecornerSrc, eyecornerDst)
        
        # 进行仿射变换，图像，矩阵，输出尺寸，所有图片尺寸一致，并且眼睛在同一位置，但是其他面部特征不在同一位置
        #需要使用德劳内三角
        img = cv2.warpAffine(images[i], tform, (w,h))

        # Apply similarity transform on points
        #将points1列表转换成一个三维数组
        points2 = np.reshape(np.array(points1), (68,1,2))       
        #对三维点进行相似变换
        points = cv2.transform(points2, tform)
        #转成二维数组
        points = np.float32(np.reshape(points, (68, 2)))
        
        # 加入8个边界点，为了构造德劳内三角形
        points = np.append(points, boundaryPts, axis=0)
        
        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / numImages
        pointsNorm.append(points)
        imagesNorm.append(img)
    

    # Delaunay triangulation计算德劳内三角形
    rect = (0, 0, w, h)
    dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))#76个点构成的142个三角形
    #print(dt[141][2])
    #print (pointsNorm[5])
    # Output image
    output = np.zeros((h,w,3), np.float32())
    # Warp input images to average image landmarks将输入图像标志点转换成平均图像标志点
    for i in range(0, len(imagesNorm)) :
        img = np.zeros((h,w,3), np.float32())
        # Transform triangles one by one 转移delaunay三角形
        for j in range(0, len(dt)) :
            tin = []
            tout = []
            
            for k in range(0, 3) :                
                pIn = pointsNorm[i][dt[j][k]]
                pIn = constrainPoint(pIn, w, h)
                
                pOut = pointsAvg[dt[j][k]]
                pOut = constrainPoint(pOut, w, h)
                
                tin.append(pIn)
                tout.append(pOut)
            
            warpTriangle(imagesNorm[i], img, tin, tout)

        # Add image intensities for averaging增加图像的平均强度
        output = output + img
    # Divide by numImages to get average 除以numImages得到平均值
    output = output / numImages

    # Display result
    #cv2.imshow('image', output)
    #cv2.waitKey(0)
    cv2.imwrite(path+'output.jpg',255*output)
