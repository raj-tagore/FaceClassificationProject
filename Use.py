from Analysis import model2
import cv2
import pandas as pd

b=cv2.imread(r"D:\Padhai\facedata2\faces72.jpg",0)
b=cv2.resize(b,(50,50))
b = b.reshape(1, 2500)
#m=pd.DataFrame(a)
dfm2 = pd.DataFrame(b)
dfm2.insert(0, "unnamed", 0)
print(dfm2)
y = model2.predict(dfm2)
print(y)