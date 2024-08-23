from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import csv


def data_split(data, look_back=1):
    x, y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        x.append(a)
        y.append(data[i + look_back, 0])
    return np.array(x), np.array(y)

def predict(df, model):
    df = pd.DataFrame(df)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(df)
    scale_data = scaler.transform(df)
    X, Y = data_split(scale_data, look_back=18)
    x = X[-1]
    pred = model.predict(x.reshape((1, 1, 18)))
    pred = scaler.inverse_transform(pred).flatten().astype('int')
    return pred



def getPrediction(data):
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    # df_SAS = df.iloc[:,3]
    df_SAS = df[2]
    # df_SAX = df.iloc[:,2]
    df_SAX = df[3]
    current_path = os.path.dirname(__file__)
    model_SAS = load_model(os.path.join(current_path, './model_spr.h5'))
    model_SAX = load_model(os.path.join(current_path, './model_nspr.h5'))
    SAS = predict(df_SAS, model_SAS)
    SAX = predict(df_SAX, model_SAX)
    return SAX[0], SAS[0]

if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    sas, sax =  getPrediction(data)
    print(sas, sax)
