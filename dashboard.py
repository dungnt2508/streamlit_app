import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import s4_predict as s4


st.set_page_config(page_title="AI Stock Prediction", layout="wide")

st.title("📈 AI Dashboard Dự Báo Giá Cổ Phiếu")

# Nhập mã cổ phiếu
symbol = st.text_input("Nhập mã cổ phiếu (VD: AAPL)", "AAPL")

# Chọn số ngày dự báo
days = st.slider("Số ngày dự báo", 1, 30, 5)

# Button dự đoán
if st.button("Dự đoán"):
    with st.spinner("Đang dự đoán..."):
        predictions = s4.predict_stock(symbol, days)

    # Vẽ biểu đồ dự báo
    fig, ax = plt.subplots()
    ax.plot(predictions, marker='o', linestyle='-', color='blue', label='Dự báo')
    ax.set_title(f"Dự báo giá cổ phiếu {symbol} trong {days} ngày tới")
    ax.legend()

    st.pyplot(fig)

    # Hiển thị dữ liệu dự đoán
    df = pd.DataFrame({"Ngày": [datetime.today().date() + pd.Timedelta(days=i) for i in range(days)],
                       "Giá dự báo": predictions})
    st.dataframe(df)
