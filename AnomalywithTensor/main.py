import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from Process_input_data import *
name_tags=['BP up','BP down','SpO2','Body temp','Age' ,'Gender','Height','Y']
Data_frame = pd.read_csv("Wellness ML algo.csv")
temp=pd.get_dummies((Data_frame["Gender"]))
Data_frame.pop('Gender')
Data_frame["M"]=temp["M"].to_list()
Data_frame["F"]=temp["F"].to_list()
temp_2=Data_frame.pop('Y')
Data_frame["Y"]=temp_2.to_list()
T=Data_frame.to_numpy()
MMs=MinMaxScaler()
MMs.fit_transform(T)
ss=StandardScaler()
ss.fit_transform(T)
X=T[0:T[:,1].size,0:T[1].size-1]
Y=T[:,T[1].size-1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=T[:,1].size-1,random_state=4)
ans=IsolationForest(contamination=0.38)
ans.fit(X_train)
X_new= list(input("Please input the parameters : ").strip().split())[:7]
X_new = process_input_data(X_new)
predictions = ans.predict(X_new)
def chnge(p):
    if p==1:
        return 0
    else:
        return 1
val=map(chnge,predictions)
print(list(val))