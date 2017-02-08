import cv2
import numpy as np
import math
from math import sin,cos,radians

img_filename = ''   #Name of the image over which to perform line detection
img = cv2.imread(img_filename)
img_smooth = cv2.GaussianBlur(img,(7,7),3)  #Gaussian filter to make image smooth
img_edges = cv2.Canny(img_smooth[:,:,2],50,150) #Calculate edge using Canny operator
(threshold, img_bw) = cv2.threshold(img_edges, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)   #Convert image to binary (black and white)_

#Function arguments for hough_lines_acc and other default values
thetaBinSize = 2
rhoMax = math.hypot(img_bw.shape[0]-1,img_bw.shape[1]-1)
rhoBinSize = 1
tMax=89 #Max theta (degrees)
tMin = -90

def hough_lines_acc(image,rhoRes=rhoBinSize,thetaMin=tMin,thetaMax=tMax): #thetaMax+1 because range has open interval on the right
    rhoBinCount = 2*math.ceil((rhoMax) / rhoRes)+1
    thetaBinCount = math.ceil((thetaMax-thetaMin)/thetaBinSize)+1

    H = np.zeros((rhoBinCount, thetaBinCount))
    rhoArray = np.arange(-rhoMax,rhoMax,rhoBinSize)
    thetaArray = np.arange(thetaMin,thetaMax+1,thetaBinSize)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y,x]<=0:
                continue
            else:
                for theta in range(thetaMin,thetaMax+1,thetaBinSize):
                    rho = x * cos(radians(theta)) + y * sin(radians(theta)) #Standard polar coordinate system representation of a line
                    thetaindex = math.ceil((theta + thetaMax + 1) / thetaBinSize)   #Adjust by adding thetaMax+1 to account for negative values; readjusted later
                    rhoindex = math.ceil((rho+rhoMax) / rhoRes) #Adjust by adding rhoMax to account for negative values; readjusted later
                    H[rhoindex,thetaindex] += 1 #Increment by 1 at the appropriate cell in the accumulator matrix

    return H,thetaArray,rhoArray

hMat,tMat,rMat = hough_lines_acc(img_bw)

#The following method detects the peaks in the accumulator array
def hough_peaks(hMatrix,numPeaks=1,thresFactor=0.4,nHoodFactor=(1,1)):
    threshold = (np.max(hMatrix) * thresFactor)
    peaks = np.array(np.where(hMatrix>threshold)) #Returns a 2xN matrix: First row is 'y' (i.e. row index) and second is 'x' coordinate in image

    if peaks.shape[1]>numPeaks: #If total peaks found is greater than the desired then sort and select the largest n (=numPeaks)
        intensity = np.zeros((1,peaks.shape[1]))    #intensity is the number of votes; used for sorting
        intensity += hMatrix[peaks[0],peaks[1]]
        PITable = np.append(peaks,intensity,axis=0) #Append intensity values to the peak table to use for sorting
        PITable_Sorted = PITable[:,np.argsort(PITable[PITable.shape[0]-1,:])] #Sort all columns based on the values of the last row (i.e. intensity)
        peaks = PITable_Sorted[0:2,0:numPeaks] #Slice off the intensity values and return sorted Hough Matrix (hMatrix) coordinates
    return peaks

peakMat = hough_peaks(hMat,numPeaks=10)
print(peakMat)

#Convert position in Hough matrix into theta and rho values
thetaVals = np.radians((peakMat[1] * thetaBinSize) - tMax - 1)  #Convert theta values from peak matrix into radians
rhoVals = (peakMat[0] * rhoBinSize) - rhoMax    #Reversing the adjustment made above
thetaRhoMat = np.array((thetaVals,rhoVals))

#Draw lines over the image using the peaks located
for i in range(thetaRhoMat.shape[1]):

    theta = thetaRhoMat[0,i]
    rho = thetaRhoMat[1,i]
    x1 = int(rho)
    y1 = int((-cos(theta)/sin(theta))*x1 + (rho/sin(theta))) if theta!=0 else 0
    x2 = img_bw.shape[1]-1 if theta!=0 else int(rho)
    y2 = int((-cos(theta)/sin(theta))*x2 + (rho/sin(theta))) if theta!=0 else img_bw.shape[0]-1

    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()