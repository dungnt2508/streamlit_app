import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(data, time_steps=50):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i-time_steps:i, 0])
        y.append(data_scaled[i, 0])

    return np.array(X), np.array(y), scaler
