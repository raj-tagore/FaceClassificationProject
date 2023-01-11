import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.utils import check_array
import matplotlib.pyplot as plt

df = pd.read_csv("trainData6.csv")
df2 = pd.read_csv("Book1.csv")
print (df)
df2 = df2.astype("category")

xl, xt, yl, yt = train_test_split(df, df2)
print(xl.shape, xt.shape)


# here xt = x test and xl = x  learn, yp = y predicted
model = LogisticRegression()
model.fit(xl, yl)
yp = model.predict(xt)
print(confusion_matrix(yt, yp))
print(accuracy_score(yt, yp))

model2 = RandomForestClassifier()
model2.fit(xl, yl)
yp = model2.predict(xt)
print(confusion_matrix(yt, yp))
print(accuracy_score(yt, yp))

