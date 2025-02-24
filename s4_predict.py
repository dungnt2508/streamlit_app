from tensorflow.keras.models import load_model
import joblib
import s1_getdata as s1
import s2_prepare_data as s2
import numpy as np

def predict_stock(symbol, days=1):
    model = load_model(f"{symbol}_lstm_model.h5")
    scaler = joblib.load(f"{symbol}_scaler.pkl")

    data = s1.get_stock_data(symbol)
    X, _, _ = s2.prepare_data(data)
    X = X[-1].reshape(1, X.shape[1], 1)

    predictions = []
    for _ in range(days):
        pred = model.predict(X)
        predictions.append(pred[0][0])
        X = np.append(X[:, 1:, :], np.array(pred).reshape(1, 1, 1), axis=1)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions).flatten()

    return predictions
