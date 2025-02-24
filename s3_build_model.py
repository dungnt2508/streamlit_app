import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import s1_getdata as s1
import s2_prepare_data as s2
import joblib

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
    data = s1.get_stock_data(symbol)
    X, y, scaler = s2.prepare_data(data)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_lstm_model((X.shape[1], 1))

    model.fit(X, y, epochs=50, batch_size=32, verbose=1)

    model.save(f"{symbol}_lstm_model.h5")
    joblib.dump(scaler, f"{symbol}_scaler.pkl")

    return model, scaler

