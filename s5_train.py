

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def get_stock_data(symbol, start="2020-01-01", end="2025-02-19"):
    data = yf.download(symbol, start=start, end=end)
    return data[['Close']]


def prepare_data(data, time_steps=50):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i - time_steps:i, 0])
        y.append(data_scaled[i, 0])

    return np.array(X), np.array(y), scaler


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_lstm(symbol):
    print(f"🔄 Đang tải dữ liệu cho {symbol}...")
    data = get_stock_data(symbol)

    print("🔄 Đang chuẩn bị dữ liệu...")
    X, y, scaler = prepare_data(data)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    print("🔄 Đang xây dựng mô hình...")
    model = build_lstm_model((X.shape[1], 1))

    print("🔄 Đang huấn luyện mô hình...")
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)

    model.save(f"{symbol}_lstm_model.h5")
    joblib.dump(scaler, f"{symbol}_scaler.pkl")

    print(f"✅ Mô hình {symbol} đã được lưu!")



train_lstm('AAPL')