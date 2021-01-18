from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn import preprocessing
def process_input_data(X_new):
    name_tags = ['BP up', 'BP down', 'SpO2', 'Body temp', 'Age', 'Gender', 'Height']
    df=pd.DataFrame([X_new],columns=name_tags)
    temp = pd.get_dummies((df["Gender"]))
    if df.iloc[0,5]=="F":
        df["F"] = 1
        df["M"]=0
    else:
        df["M"] = 1
        df["F"]=0
    df.pop('Gender')
    X_new=df.to_numpy()
    MMs=MinMaxScaler()
    MMs.fit_transform(X_new)
    ss=StandardScaler()
    ss.fit_transform(X_new)
    return  X_new