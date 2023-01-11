import pandas as pd
import cv2

a=cv2.imread(r"D:\Padhai\img_align_celeba\000001.jpg",0)
a.shape
a=cv2.resize(a,(50,50))
m=pd.DataFrame(a)
cv2.imshow("Example",a)
cv2.waitKey(0)
cv2.destroyAllWindows()
