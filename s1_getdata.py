import yfinance as yf
import pandas as pd

def get_stock_data(symbol, start="2020-01-01", end="2025-01-01"):
    data = yf.download(symbol, start=start, end=end)
    return data
