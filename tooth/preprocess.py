import os
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
imgset=[]
temp=np.zeros((100,1))
# temp2=np.zeros((170,1))
temp3=np.zeros((100,1))
for i in os.listdir('jpgdata'):
    img=cv2.imread(os.path.join('jpgdata',i),cv2.IMREAD_GRAYSCALE)
    H,W=img.shape
    top=(600-H)//2
    bottom=600-H-(600-H)//2
    left=(512-W)//2
    right=512-W-(512-W)//2
    padding =cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
    print(padding.shape)
    img=padding[360:460,150:380]
    corner =cv2.cornerHarris(img, 3,0,0.05)
    dst=cv2.dilate(corner,None)
    img[dst>0.03*dst.max()] = 0
    a,b=np.where(dst>0.03*dst.max())
    new=dst>0.03*dst.max()
    height=np.unique(a)
    width=np.unique(b)
    Y=np.zeros((len(width)))
    reg=LinearRegression()
    for ind,i in enumerate(width):
        index=np.where(b==i)
        Y_list=a[index]
        Y_median=round(np.median(Y_list))
        Y[ind]=Y_median
    Y=Y.reshape(-1,1)
    width=width.reshape((-1,1))
    reg.fit(width,Y)
    coef=float(reg.coef_)
    inter=float(reg.intercept_)
    for j in range(img.shape[1]):
        img[round(j*coef+inter),j]=255
    temp=np.hstack((temp,cv2.Canny(padding[360:460,150:380],1,200)))
    # temp2=np.hstack((temp2,padding[330:500,150:380]))
    temp3=np.hstack((temp3,img))
    # cv2.imshow('test',padding)
    # cv2.waitKey(0)
    # imgset.append(padding[330:500,150:380])
result=np.vstack((temp,temp3))
plt.imshow(temp3,cmap='gray')
plt.show()
