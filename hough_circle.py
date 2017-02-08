#PENDING TASKS FOR FURTHER OPTIMIZATION:
#1. hMat grows 2*int(radMax)+1 for all radiuses; i.e., all "radius" dimensions are growing to max radius size;
#   Enlarge each matrix only by its corresponding radius size
#2. A given (a,b) point currently only votes for themselves; try voting for "neighboring" pixels and evaluate performance
#3. Conduct "neighborhood suppression" of local maximas in hMat and evaluate performance

import cv2
import numpy as np
import math

#Load the three-channel and the grayscale version of the image
img_filename = ''
img = cv2.imread(img_filename)
img_gray=cv2.imread(img_filename,0) #Load grayscale version of same image
img_smooth = cv2.GaussianBlur(img_gray,(7,7),7) #'Smooth' using Gaussian filter of kernel size 7x7 and s.d.=7; use if Gaussian noise is present in image

#Calculate the Sobel matrices, the gradient matrix, and the Canny edge matrix
sobelX = cv2.Sobel(img_gray,cv2.CV_8U,1,0,ksize=7)  #Sobel gradient along X axis
sobelY = cv2.Sobel(img_gray,cv2.CV_8U,0,1,ksize=7)  #Sobel along Y axis
gradTheta = np.arctan2(sobelY,sobelX)   #Gradient direction matrix: arctan(y/x)
img_edges = cv2.Canny(img_gray,50,170)    #Detect image edges using canny operator
(threshold, img_bw) = cv2.threshold(img_edges, 25, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)    #Convert image to binary

#Get the appropriate matrices for the following operations:
radMin = 10 #Minimun radius for consideration
radMax = 40 #Maximum radius for consideration
radRes = 1 #Resolution or the bin size for the radius
radRange = int(math.ceil(radMax-radMin+1)/radRes)   #Gives the number of bins
radMultiplier = np.arange(radMin,radMin+(radRange*radRes),radRes).reshape((radRange,1,1)) #This is to multiply each radius-dimension with seperate radius value
gradTheta = np.tile(np.arctan2(sobelY,sobelX),(radRange,1,1))   #Tile the gradTheta matrix
voters = np.tile(img_bw>0,(radRange,1,1))  #Boolean matrix to keep track of which y,x cells get to vote; only cells with positive values vote

#Setup X and Y matrices and calculate A and B
index = np.indices(img_bw.shape) #Returns matrices with row and column index values
X = np.tile(index[1],(radRange,1,1)) #Sets column values to represent x-index in the image; tile adds third dimension of value radRange
Y = np.tile(index[0],(radRange,1,1)) #Sets row values
A = X - (radMultiplier * np.cos(gradTheta)) #Calculate A using the standard formula
A = A[voters].astype(int)   #Remove non-voters and convert to integer type from float
B = Y - (radMultiplier * np.sin(gradTheta)) #Same as A
B = B[voters].astype(int)

#Setup accumulator matrix and adjust other matrices so they have appropriate (i.e. non-negative) values for matrix indexing
hMat = np.zeros((radRange,img_bw.shape[0]+2*int(radMax)+1,img_bw.shape[1]+2*int(radMax)+1))    #hMat can have (a,b) circle center values "outside" (x,y) range
A = A + (2*radMax) #Positive scaling so that there are no negative values in A; done for indexing in hMat;later reversed
B = B + (2*radMax)
radMultiplier = ((radMultiplier - radMin)/radRes).astype(int) #Adjusting for indexing purposes; re-adjusted to original state below
hIndex = tuple((radMultiplier,B,A)) #The corresponding tuple values in (B,A) will provide (row,col) index in hMat
hMat[hIndex]+=1 #update accumulator

#Locating peaks
peakCount = 10  #Number of Hough peaks we want to detect in the Hough Accumulator
peakIndex = np.argsort(np.ravel(hMat))[::-1][:peakCount]    #Get the first peakCount number of "flat" hMat indices of the largest values
peakRYX = np.unravel_index(peakIndex, (hMat.shape)) #Unravel the "ravelled" peakIndex to get the Y,X,R coordinate tuple of the largest n (=peakCount) values

#Undoing the adjustments made above for indexing = final values of radius, y, and x
peakR = (peakRYX[0] + radMin) * radRes  #Radius adjustment
peakY = peakRYX[1] - (2*radMax) #Y Adjustment
peakX = peakRYX[2] - (2*radMax) #X Adjustment

#Draw the circles using the located peaks
for i in range(peakCount):
    cv2.circle(img, (peakX[i],peakY[i]), peakR[i], (0,255,0))

cv2.imshow('hMat',img)
cv2.waitKey()
cv2.destroyAllWindows()