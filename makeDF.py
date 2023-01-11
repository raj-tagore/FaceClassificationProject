import cv2
import numpy as np
import pandas as pd
import os

a=cv2.imread(r"D:\Padhai\img_align_celeba\000001.jpg",0)
a=cv2.resize(a,(50,50))
a = a.reshape(1, 2500)
#m=pd.DataFrame(a)
dfm = pd.DataFrame(a)

#images = []


for i in range(2, 437):
    if os.path.exists(r"D:\Padhai\facedata\faces"+str(i)+".jpg") == True:
        img = cv2.imread(r"D:\Padhai\facedata\faces"+str(i)+".jpg", 0)
        img = cv2.resize(img, (50, 50))
        img = img.reshape(1, 2500)

        df = pd.DataFrame(img)
        dfm=pd.concat([dfm, df])


    else :
        continue

#data = {"images" : images, "MorF" : MorF}
#df = pd.DataFrame(data)
#print(df)
dfm.to_csv(r'D:\Padhai\FaceDetectionProject2\trainData6.csv')

#cv2.imshow("", a)