import sklearn
import pandas as pd
import matplotlib.pyplot as plt
df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")
df_test=df_test.iloc[:,2:]
from sklearn.model_selection import train_test_split
x=df_train.iloc[:,:-1]
y=df_train.iloc[:,24]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
print(mse)
y_pred2 = clf.predict(df_test)
df_test1=pd.read_csv("test.csv")
df_test2=list(df_test1['soldierId'].astype(int))
y_pred3=list(y_pred2)
final = pd.DataFrame({'soldierId': df_test2,'bestSoldierPerc': y_pred3})
final.to_csv('submission1.csv')
print(final)

