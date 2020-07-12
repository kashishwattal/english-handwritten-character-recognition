import numpy as np
import pandas as pd
import cv2
import os

bin_n = 9 # Number of bins

def deskew(img):
        gray = cv2.bitwise_not(img)
        thresh = cv2.threshold(gray, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
                angle = -(90 + angle)
        else:
                angle = -angle
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)   
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:16,:16], bins[16:,:16], bins[:16,16:], bins[16:,16:] 
    mag_cells = mag[:16,:16], mag[16:,:16], mag[:16,16:], mag[16:,16:] 
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

path=r'C:\Users\Desktop\extracted_images2\\'
filenames=os.listdir(path)

flabels=np.load('flabels.npy')
fvectors=np.load('fvectors.npy')
flabels.reshape(len(filenames),1)
naya=np.hstack((flabels,fvectors))
np.random.shuffle(naya)#shuffle the feature vector set
flabels=naya[0:,0]
fvectors=naya[0:,1:].astype('float')
maximum=np.amax(fvectors)#scaling the features
fvectors/=maximum#to 0-1
training_points=fvectors[0:int(0.80*len(filenames))]
training_labels=flabels[0:int(0.80*len(filenames))]
test_points=fvectors[int(0.80*len(filenames)):]
test_labels=flabels[int(0.80*len(filenames)):]
#training_points=fvectors
#training_labels=flabels
k=3
accuracy=0
for x in range(len(test_labels)):
    diff=(np.tile(test_points[x],(len(training_points),1)))-training_points
    square=diff**2
    sumsqr=np.sum(square,axis=1)
    sqrt=sumsqr**0.5
    ind=sqrt.argsort()
    a=list()
    for i in range(0,k):
        a.append(training_labels[ind[i]])
    cnt=list()
    for y in a:
        cnt.append(a.count(y))
    
    m=cnt.index(max(cnt))
    label=a[m]
    if label==test_labels[x]:
        accuracy+=1
    print(x)  

print(float(accuracy/len(test_labels)))



