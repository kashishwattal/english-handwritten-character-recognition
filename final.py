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

path=r'C:\Users\Kashish\Desktop\images2\\'#images3 for NIST database
filenames=os.listdir(path)
fvectors=np.zeros((len(filenames),36))#feature vector
flabels=np.zeros((len(filenames),1)).astype('int')#feature label
asc=65
for i in range(len(filenames)):
        var=path+filenames[i]
        img=cv2.imread(var,0)
        r=deskew(img)#deskew the image
        h = cv2.adaptiveThreshold(r,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)#binarize the image
        im2, contours, hierarchy = cv2.findContours(h,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#find contours
        x, y, width, height = cv2.boundingRect(contours[1])#to find the area
        roi = r[y:y+height, x:x+width]#to be cropped
        roi=cv2.resize(roi,(32,32))
        blur = cv2.bilateralFilter(roi,9,75,75)#filtering
        hist=hog(blur)#hog features
        hist=hist.reshape(1,36)
        fvectors[i]=hist
        if (i%55==0 and i!=0):#each character has 55 sample images(4000 for the NIST database)
                asc+=1
        flabels[i]=asc#labels contain ascii of characters they represent
        print(i)

#np.save('flabels',flabels)
#np.save('fvectors',fvectors)
#flabels=np.load('flabels.npy')
#fvectors=np.load('fvectors.npy')
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
k=1
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



